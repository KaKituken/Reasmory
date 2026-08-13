import json
import re
from typing import Any, Dict


VALID_PART2_MOTIONS = {
    "turn left",
    "turn right",
    "turn up",
    "turn down",
    "turn back",
    "step forward",
    "step backward",
    "step left",
    "step right",
    "step up",
    "step down",
}

INVALID_ARBITRARY_ALIASES = {
    "unspecified",
    "unknown",
    "not specified",
    "not given",
    "none",
    "null",
    "n/a",
    "na",
}


class DecompositionVerificationError(Exception):
    pass


def extract_json_text(text: str) -> str:
    match = re.search(r"```json\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_decomposition_output(text: str) -> Dict[str, Any]:
    json_text = extract_json_text(text)
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise DecompositionVerificationError(
            f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc


def verify_decomposition_structure(decomposition: Dict[str, Any]) -> None:
    if not isinstance(decomposition, dict):
        raise DecompositionVerificationError("The decomposition output must be a JSON object.")

    for key in ("Part1", "Part2", "Part3"):
        if key not in decomposition:
            raise DecompositionVerificationError(f"Missing required top-level key `{key}`.")

    part1 = decomposition["Part1"]
    part2 = decomposition["Part2"]
    part3 = decomposition["Part3"]

    if not isinstance(part1, dict):
        raise DecompositionVerificationError("`Part1` must be an object.")
    if not isinstance(part2, list):
        raise DecompositionVerificationError("`Part2` must be a list.")
    if not isinstance(part3, dict):
        raise DecompositionVerificationError("`Part3` must be an object.")

    for key in ("position", "orientation"):
        if key not in part1 or not isinstance(part1[key], str):
            raise DecompositionVerificationError(f"`Part1.{key}` must be a string.")
        normalized_value = part1[key].strip().lower()
        if normalized_value in INVALID_ARBITRARY_ALIASES:
            raise DecompositionVerificationError(
                f"`Part1.{key}` uses `{part1[key]}`, but unspecified viewpoint fields must be written "
                'exactly as `"arbitrary"`.'
            )

    for idx, step in enumerate(part2):
        if not isinstance(step, dict):
            raise DecompositionVerificationError(f"`Part2[{idx}]` must be an object.")
        motion = step.get("motion")
        grounding = step.get("grounding")
        if not isinstance(motion, str):
            raise DecompositionVerificationError(f"`Part2[{idx}].motion` must be a string.")
        if motion not in VALID_PART2_MOTIONS:
            raise DecompositionVerificationError(
                f"`Part2[{idx}].motion` must be one of {sorted(VALID_PART2_MOTIONS)}, got `{motion}`."
            )
        if not isinstance(grounding, str):
            raise DecompositionVerificationError(f"`Part2[{idx}].grounding` must be a string.")

    if "final_question" not in part3 or not isinstance(part3["final_question"], str):
        raise DecompositionVerificationError("`Part3.final_question` must be a string.")
    final_question_lower = part3["final_question"].lower()
    contain_direction = part3.get("contain_direction")
    if contain_direction is not None and not isinstance(contain_direction, str):
        raise DecompositionVerificationError("`Part3.contain_direction` must be a string or null.")
    reference_entity = part3.get("reference_entity")
    if reference_entity is not None and not isinstance(reference_entity, str):
        raise DecompositionVerificationError("`Part3.reference_entity` must be a string or null.")
    target_object = part3.get("target_object")
    if target_object is not None:
        if not isinstance(target_object, list) or not all(isinstance(x, str) for x in target_object):
            raise DecompositionVerificationError("`Part3.target_object` must be a list of strings or null.")

    for idx, step in enumerate(part2):
        grounding = step.get("grounding")
        if not isinstance(grounding, str):
            continue
        grounding_lower = grounding.strip().lower()
        if grounding_lower and grounding_lower in final_question_lower:
            raise DecompositionVerificationError(
                f"`Part2[{idx}].grounding` appears verbatim inside `Part3.final_question`. "
                "This usually means the motion/setup was not properly decomposed out of the final ask."
            )


def format_decomposer_feedback(previous_output: str, error_text: str) -> str:
    return (
        "Your previous decomposition failed verification.\n\n"
        f"Verifier error:\n{error_text}\n\n"
        "Please regenerate the full decomposition JSON.\n"
        "Rules:\n"
        "- Output JSON only.\n"
        "- Keep the keys `Part1`, `Part2`, and `Part3`.\n"
        f"- Every `Part2.motion` must be one of: {sorted(VALID_PART2_MOTIONS)}.\n"
        "- Do not combine multiple actions into one motion string.\n\n"
        "Previous output:\n"
        f"{previous_output}"
    )
