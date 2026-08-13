# runtime.py
import os
import threading
import torch
from transformers import Sam2Processor, Sam2Model, AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import Qwen3VLForConditionalGeneration
from pi3.models.pi3 import Pi3
from pi3.models.pi3x import Pi3X
from pi3.utils.basic import load_images_as_tensor
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
from sam3.eval.postprocessors import PostProcessImage

try:
    from .spatial_memory import SpatialMemory
except:
    from spatial_memory import SpatialMemory

class Runtime:
    def __init__(self):
        self._lock = threading.Lock()

        # heavy singletons
        self.pi3 = None
        self.pi3x_metric = None
        self.sam2_model = None
        self.sam2_processor = None
        self.sam3_model = None
        self.sam3_processor = None
        self.gd_dino_processor = None
        self.gd_dino = None
        self.qwen3_vl_processor = None
        self.qwen3_vl_model = None

        # per-session cache: session_id -> spatial_memory dict
        self.session_mem = {}

    def load_spatial_memory_cache(self, session_id: str, cache_path: str):
        if session_id in self.session_mem:
            return self.session_mem[session_id]
        memory = SpatialMemory.load(cache_path)
        self.session_mem[session_id] = memory
        return memory
    def ensure_sam2(self):
        with self._lock:
            if self.sam2_model is not None:
                return
            sam2_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
            self.sam2_model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(sam2_device)
            self.sam2_processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")

    def ensure_sam3(self, device=None):
        # SAM3 tokenizer asset. Override the checkout location with REASMORY_SAM3_ROOT
        # (defaults to a `third_party/sam3` sibling of this repository).
        _sam3_root = os.environ.get("REASMORY_SAM3_ROOT", "").strip()
        if not _sam3_root:
            _sam3_root = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))))),
                "third_party", "sam3",
            )
        bpe_path = os.path.join(_sam3_root, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
        with self._lock:
            if self.sam3_model is not None:
                return
            if device is not None:
                sam3_device = device
            else:
                sam3_device = "cuda:2" if torch.cuda.device_count() > 1 else "cuda:0"
            # import ipdb; ipdb.set_trace()
            self.sam3_model = build_sam3_image_model(bpe_path=bpe_path, device=sam3_device)
            self.sam3_processor = Sam3Processor(self.sam3_model, device=sam3_device)
            self.sam3_transform = ComposeAPI(
                transforms=[
                    RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
                    ToTensorAPI(),
                    NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
            self.sam3_postprocessor = PostProcessImage(
                max_dets_per_img=-1,       # if this number is positive, the processor will return topk. For this demo we instead limit by confidence, see below
                iou_type="segm",           # we want masks
                use_original_sizes_box=True,   # our boxes should be resized to the image size
                use_original_sizes_mask=True,   # our masks should be resized to the image size
                convert_mask_to_rle=False, # the postprocessor supports efficient conversion to RLE format. In this demo we prefer the binary format for easy plotting
                detection_threshold=0.5,   # Only return confident detections
                to_cpu=True,
            )

    def ensure_pi3(self, device=None):
        with self._lock:
            if self.pi3 is not None:
                return
            if device is not None:
                pi3_device = device
            else:
                pi3_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
            self.pi3 = Pi3.from_pretrained("yyfz233/Pi3").to(pi3_device).eval()
            self.pi3.device = pi3_device

    def ensure_pi3x_metric(self, device=None):
        with self._lock:
            if self.pi3x_metric is not None:
                return
            if device is not None:
                pi3x_device = device
            else:
                pi3x_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
            self.pi3x_metric = Pi3X.from_pretrained("yyfz233/Pi3X").to(pi3x_device).eval()
            if hasattr(self.pi3x_metric, "disable_multimodal"):
                self.pi3x_metric.disable_multimodal()
            self.pi3x_metric.device = pi3x_device

    def ensure_grounding_dino(self):
        with self._lock:
            if self.gd_dino is not None:
                return
            grounding_dino_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
            gd_dino_path = "IDEA-Research/grounding-dino-base"
            self.gd_dino_processor = AutoProcessor.from_pretrained(gd_dino_path)
            self.gd_dino = AutoModelForZeroShotObjectDetection.from_pretrained(gd_dino_path).to(grounding_dino_device)

    def ensure_qwen3_vl(self, model_path=None, device=None):
        with self._lock:
            if self.qwen3_vl_model is not None:
                return
            if model_path is None:
                model_path = os.environ.get("QWEN3_VL_RECOVERY_MODEL_PATH", "Qwen/Qwen3-VL-4B-Instruct")
            if device is None:
                device = os.environ.get("QWEN3_VL_RECOVERY_DEVICE")
            if device is None:
                device = "cuda:3" if torch.cuda.device_count() > 3 else ("cuda:0" if torch.cuda.is_available() else "cpu")

            torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
            model_kwargs = {
                "torch_dtype": torch_dtype,
            }
            if device.startswith("cuda"):
                model_kwargs["attn_implementation"] = "flash_attention_2"

            self.qwen3_vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                **model_kwargs,
            ).eval().to(device)
            self.qwen3_vl_processor = AutoProcessor.from_pretrained(model_path)

    def ensure_spatial_memory(self, session_id: str, image_paths: list, construct_3d_spatial_memory_fn):
        """
        image_paths: 多张输入视图（来自消息中的 image items）
        construct_3d_spatial_memory_fn: 你现有的 construct_3d_spatial_memory（可稍微改成用 self.pi3）
        """
        if session_id in self.session_mem:
            return self.session_mem[session_id]

        # heavy init once
        self.ensure_pi3()

        images = load_images_as_tensor(image_paths)
        # import ipdb; ipdb.set_trace()
        # This function might be called in different threads, so we need to ensure thread safety when calling the construct_3d_spatial_memory_fn if it uses self.pi3
        with self._lock:
            position, confidence, camera_trajectory, intrinsics, global_up = construct_3d_spatial_memory_fn(images, pi3=self.pi3)

        mem = SpatialMemory(
            rgb_images=images,
            position_3d=position,
            confidence=confidence,
            camera_trajectory=camera_trajectory,
            intrinsics=intrinsics,
            global_up=global_up
        )
        self.session_mem[session_id] = mem

    def ensure_metric_spatial_memory(self, session_id: str, image_paths: list, construct_3d_spatial_memory_fn):
        if session_id in self.session_mem:
            return self.session_mem[session_id]

        self.ensure_pi3x_metric()

        images = load_images_as_tensor(image_paths)
        with self._lock:
            position, confidence, camera_trajectory, intrinsics, global_up = construct_3d_spatial_memory_fn(
                images,
                pi3x=self.pi3x_metric,
            )

        mem = SpatialMemory(
            rgb_images=images,
            position_3d=position,
            confidence=confidence,
            camera_trajectory=camera_trajectory,
            intrinsics=intrinsics,
            global_up=global_up,
        )
        self.session_mem[session_id] = mem

    @property
    def already_initialized(self):
        return self.pi3 is not None and self.sam2_model is not None and self.sam2_processor is not None

    def cleanup_session(self, session_id: str):
        if session_id in self.session_mem:
            del self.session_mem[session_id]
        torch.cuda.empty_cache()
