"""
Prompt templates for the VLM4D dynamic-branch three-stage pipeline.

Stages (mirroring the static pipeline):
    [1] Decomposer  -> structured JSON  (entity, ref-frame, archetype, scope)
    [2] Planner     -> DSL plan picking dynamic-memory tools (this MVP routes
                       through a deterministic mapping per archetype)
    [3] Reasoner    -> answers the original question using decomposition +
                       executor evidence + video frames
"""

VLM4D_DECOMPOSER_SYSTEM = """\
You decompose dynamic-video reasoning questions into a structured JSON object.
Do not answer the original question.
Return ONLY valid JSON; no markdown fences, no commentary.
"""

VLM4D_DECOMPOSER_PROMPT = """\
You are decomposing a multiple-choice question about MOTION in a dynamic video.

[Question]
{question_description}

[Options]
{options}

[Task]
Return a JSON object with exactly these fields.

{{
  "moving_entity": "the noun phrase of the thing whose motion/event the question is about (or 'camera' for ego-motion questions). Copy as faithfully as possible from the question.",
  "reference_frame": one of:
       - "camera"   (the question is screen-relative or says 'from the camera perspective')
       - "object:<noun>"   (the question says 'from <noun>'s own perspective' or 'with respect to <noun>')
       - "world"   (rarely needed; absolute world directions)
  ,
  "query_archetype": one of:
       - "DIRECTION"        (left / right / up / down / forward / backward)
       - "DEPTH_CHANGE"     (towards / away / closer / farther / approaching / receding)
       - "ROTATION_SENSE"   (clockwise vs counter-clockwise; revolutions)
       - "COUNT_EVENTS"     (how many times / how many of)
       - "PRESENCE_CHECK"   (does X happen / is X present)
       - "HAND_OR_SIDE"     (which hand / which side)
  ,
  "temporal_scope": one of:
       - "whole_video"
       - "first_half"
       - "last_half"
       - "around_event:<short event description>"
  ,
  "counting_constraint": null OR a short string like "to the left" / "with right foot" describing the extra filter on counted events. Only set when archetype is COUNT_EVENTS.,
  "expected_answer_shape": brief hint such as "direction word from options" / "non-negative integer" / "yes/no" / "object name from options".
}}

Rules:
- If the camera is the moving entity (e.g. "which direction did the camera pan"), set moving_entity = "camera".
- For questions that say "from his/her/its own perspective", reference_frame must be "object:<entity>".
- Do not add fields beyond those listed. Do not output explanation.

Return JSON only.
"""


# ---------------------------------------------------------------------------
# Reasoner — sees decomposition + executor evidence + sampled frames
# ---------------------------------------------------------------------------
VLM4D_REASONER_SYSTEM = """\
You answer a multiple-choice question about a dynamic video.
You are given:
  - the original question and options,
  - a structured decomposition (already extracted by a planner),
  - quantitative evidence derived from a Flow3r-based 3D reconstruction of
    the video (camera ego-motion summary, scene extent),
  - sampled video frames in time order.

Use the decomposition to focus your analysis. Cross-check your judgement
against the evidence — in particular, do not confuse camera ego-motion
with object motion. Give the answer in the format <answer>LETTER</answer>.
"""


def build_vlm4d_reasoner_prompt(
    *,
    question: str,
    options_block: str,
    decomposition: dict,
    evidence_block: str,
    use_cot: bool = True,
    is_egocentric: bool = False,
    include_rubric: bool = True,
) -> str:
    decomp_str = (
        f"- moving_entity     : {decomposition.get('moving_entity', '?')}\n"
        f"- reference_frame   : {decomposition.get('reference_frame', '?')}\n"
        f"- query_archetype   : {decomposition.get('query_archetype', '?')}\n"
        f"- temporal_scope    : {decomposition.get('temporal_scope', '?')}\n"
        f"- counting_constraint : {decomposition.get('counting_constraint', None)}\n"
        f"- expected_answer_shape : {decomposition.get('expected_answer_shape', '?')}\n"
    )
    archetype = (decomposition.get("query_archetype") or "").upper()
    # Per-archetype micro-rubric — keep tight
    if is_egocentric:
        # In ego4d-style video the camera IS the actor's head: ego-motion is the actor's
        # body/head motion, not a confounder. The rubric leans into this.
        rubric = {
            "DIRECTION": (
                "1) In egocentric video the camera is the actor's head — its panning/tilting "
                "IS the actor turning their head.\n"
                "2) For 'which direction did the person turn / look', read off the camera ego-motion "
                "from the evidence and pick the matching option.\n"
                "3) For object-manipulation directions ('which way did the person move X'), inspect "
                "X's pixel motion in the frame relative to the reference object stated in the decomposition."
            ),
            "DEPTH_CHANGE": (
                "1) Examine the entity's apparent SIZE across frames.\n"
                "2) Camera moving forward (per evidence) means stationary objects look like they "
                "come 'towards' the actor; combine with the size cue."
            ),
            "ROTATION_SENSE": (
                "1) Pick a tracking landmark on the rotating object.\n"
                "2) Classify clockwise vs counter-clockwise from the actor's viewpoint."
            ),
            "COUNT_EVENTS": (
                "1) Enumerate each candidate event briefly.\n"
                "2) Apply the counting_constraint; drop events that fail it.\n"
                "3) If no event qualifies and the option set includes 0, answer 0."
            ),
            "PRESENCE_CHECK": (
                "Verify whether the queried event is observable in the frames."
            ),
            "HAND_OR_SIDE": (
                "1) Identify which hand/side from the frames directly.\n"
                "2) IGNORE the camera-motion evidence for this archetype; it's irrelevant.\n"
                "3) If the option set includes 'both hands' or 'no hands', actively verify before excluding."
            ),
        }.get(archetype, "Analyse the frames step by step.")
    else:
        # Allocentric (third-person) video: camera ego-motion is a confounder that must be
        # subtracted when judging object motion.
        rubric = {
            "DIRECTION": (
                "1) Note the camera ego-motion from the evidence — if the camera is panning, mentally subtract that.\n"
                "2) Track the moving entity across the frames; estimate its pixel motion direction in the reference frame stated above.\n"
                "3) Pick the option that matches; prefer screen-relative (left/right) over depth (towards/away) if both are in options."
            ),
        "DEPTH_CHANGE": (
            "1) Examine the entity's APPARENT SIZE across frames (closer = larger).\n"
            "2) If both screen (left/right) and depth (towards/away) options exist, depth wins only when size change is clear.\n"
            "3) If camera is moving forward/backward (per evidence), adjust: a static object will look 'towards' if camera moves toward it."
        ),
        "ROTATION_SENSE": (
            "1) Pick a tracking landmark on the rotating object.\n"
            "2) Observe its angular position across 3–5 frames; classify clockwise vs counter-clockwise from the camera POV.\n"
            "3) If counting revolutions, count return-to-start events only."
        ),
        "COUNT_EVENTS": (
            "1) Enumerate each candidate event with a one-line note: 'event #k at frame ~t: <description>'.\n"
            "2) Apply the counting_constraint (e.g., 'to the left' / 'with right foot') and DROP events that fail it.\n"
            "3) Return the remaining count. If the option set includes 0 and no event qualifies, answer 0."
        ),
        "PRESENCE_CHECK": (
            "1) Verify the queried event/object is observable.\n"
            "2) Reject the 'no <entity> in video' option only after confirming you've looked at all frames."
        ),
        "HAND_OR_SIDE": (
            "1) Identify which side (left or right) of the actor in the frame.\n"
            "2) If the actor faces away from the camera, the visible left side IS the actor's left side."
        ),
        }.get(archetype, "Analyse the frames step by step.")

    head = (
        "[Task]\n"
        "Answer the dynamic-video question below using the decomposition and evidence.\n\n"
        "[Question]\n" + question + "\n\n"
        "[Options]\n" + options_block + "\n\n"
        "[Decomposition]\n" + decomp_str + "\n"
        "[Evidence from 3D dynamic memory]\n" + evidence_block + "\n\n"
    )
    if include_rubric:
        head += "[Reasoning rubric for this archetype]\n" + rubric + "\n"
    if use_cot:
        head += (
            "\n[Format]\nReason concisely, then output the final answer on its own line: "
            "<answer>LETTER</answer>\n"
        )
    else:
        head += "\n[Format]\nOutput only <answer>LETTER</answer>.\n"
    return head
