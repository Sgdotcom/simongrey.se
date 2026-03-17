from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

DEFAULT_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"


@dataclass
class ReplacementConfig:
    mask_threshold: float = 0.55
    feather_px: int = 21
    bbox_smoothing: float = 0.75
    min_mask_area_ratio: float = 0.002
    model_asset_path: str | None = None


class PersonReplacer:
    def __init__(self, config: ReplacementConfig | None = None) -> None:
        self.config = config or ReplacementConfig()
        self._segmentor = self._build_segmenter(self.config.model_asset_path)

    def close(self) -> None:
        self._segmentor.close()

    def _build_segmenter(self, model_asset_path: str | None) -> mp_vision.ImageSegmenter:
        model_path = _resolve_model_path(model_asset_path)
        options = mp_vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            output_confidence_masks=True,
        )
        return mp_vision.ImageSegmenter.create_from_options(options)

    def process_video(
        self,
        source_image_path: str | Path,
        target_video_path: str | Path,
        output_video_path: str | Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        source_rgba = self._prepare_source_person(source_image_path)
        cap = cv2.VideoCapture(str(target_video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open target video: {target_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 24.0
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        output_path = Path(output_video_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            cap.release()
            raise ValueError(f"Could not open output video writer: {output_video_path}")

        smoothed_bbox: np.ndarray | None = None
        frame_index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                person_mask = self._segment(frame)
                person_binary = person_mask > self.config.mask_threshold
                min_mask_area = int(frame.shape[0] * frame.shape[1] * self.config.min_mask_area_ratio)
                bbox = _largest_bbox(person_binary, min_area=max(1, min_mask_area))

                if bbox is not None:
                    current_bbox = np.array(bbox, dtype=np.float32)
                    if smoothed_bbox is None:
                        smoothed_bbox = current_bbox
                    else:
                        smooth = np.clip(self.config.bbox_smoothing, 0.0, 0.98)
                        smoothed_bbox = (smooth * smoothed_bbox) + ((1.0 - smooth) * current_bbox)
                    frame = self._composite(frame, source_rgba, smoothed_bbox, person_binary)

                writer.write(frame)
                frame_index += 1
                if progress_callback:
                    progress_callback(frame_index, total_frames)
        finally:
            cap.release()
            writer.release()

        return output_path

    def _prepare_source_person(self, source_image_path: str | Path) -> np.ndarray:
        source = cv2.imread(str(source_image_path))
        if source is None:
            raise ValueError(f"Could not read source image: {source_image_path}")

        source_mask = self._segment(source)
        source_binary = source_mask > self.config.mask_threshold
        bbox = _largest_bbox(source_binary, min_area=50)
        if bbox is None:
            raise ValueError("No person was detected in the source image.")

        feathered_mask = _feather(source_binary.astype(np.float32), self.config.feather_px)
        x, y, w, h = bbox
        person_crop = source[y : y + h, x : x + w]
        alpha_crop = feathered_mask[y : y + h, x : x + w]
        return np.dstack([person_crop.astype(np.float32), alpha_crop.astype(np.float32)])

    def _segment(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._segmentor.segment(image)
        if not result.confidence_masks:
            return np.zeros(frame_bgr.shape[:2], dtype=np.float32)
        mask = np.squeeze(result.confidence_masks[0].numpy_view()).astype(np.float32)
        if mask.shape != frame_bgr.shape[:2]:
            mask = cv2.resize(mask, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        return np.clip(mask, 0.0, 1.0)

    def _composite(
        self,
        frame_bgr: np.ndarray,
        source_rgba: np.ndarray,
        bbox: np.ndarray,
        target_person_binary: np.ndarray,
    ) -> np.ndarray:
        frame_h, frame_w = frame_bgr.shape[:2]
        x = int(bbox[0])
        y = int(bbox[1])
        w = max(1, int(bbox[2]))
        h = max(1, int(bbox[3]))
        x = int(np.clip(x, 0, frame_w - 1))
        y = int(np.clip(y, 0, frame_h - 1))
        w = int(np.clip(w, 1, frame_w - x))
        h = int(np.clip(h, 1, frame_h - y))

        src_bgr = source_rgba[:, :, :3]
        src_alpha = source_rgba[:, :, 3]
        src_h, src_w = src_bgr.shape[:2]

        # Scale to cover target area and preserve aspect ratio.
        scale = max(w / max(1, src_w), h / max(1, src_h))
        scaled_w = max(1, int(src_w * scale))
        scaled_h = max(1, int(src_h * scale))

        resized_bgr = cv2.resize(src_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)
        resized_alpha = cv2.resize(src_alpha, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        resized_alpha = _feather(resized_alpha, self.config.feather_px)

        target_center_x = x + (w // 2)
        target_center_y = y + (h // 2)
        left = target_center_x - (scaled_w // 2)
        top = target_center_y - (scaled_h // 2)
        right = left + scaled_w
        bottom = top + scaled_h

        frame_x1 = max(0, left)
        frame_y1 = max(0, top)
        frame_x2 = min(frame_w, right)
        frame_y2 = min(frame_h, bottom)
        if frame_x1 >= frame_x2 or frame_y1 >= frame_y2:
            return frame_bgr

        src_x1 = frame_x1 - left
        src_y1 = frame_y1 - top
        src_x2 = src_x1 + (frame_x2 - frame_x1)
        src_y2 = src_y1 + (frame_y2 - frame_y1)

        src_region = resized_bgr[src_y1:src_y2, src_x1:src_x2].astype(np.float32)
        alpha_region = resized_alpha[src_y1:src_y2, src_x1:src_x2].astype(np.float32)

        target_mask = target_person_binary[frame_y1:frame_y2, frame_x1:frame_x2].astype(np.float32)
        target_mask = _feather(target_mask, self.config.feather_px)
        alpha_region = np.clip(alpha_region * target_mask, 0.0, 1.0)
        alpha_3 = alpha_region[:, :, None]

        output = frame_bgr.astype(np.float32)
        frame_region = output[frame_y1:frame_y2, frame_x1:frame_x2]
        blended = (alpha_3 * src_region) + ((1.0 - alpha_3) * frame_region)
        output[frame_y1:frame_y2, frame_x1:frame_x2] = blended
        return np.clip(output, 0, 255).astype(np.uint8)


def _largest_bbox(mask: np.ndarray, min_area: int = 1) -> tuple[int, int, int, int] | None:
    mask_uint8 = (mask.astype(np.uint8) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < min_area:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    return int(x), int(y), int(w), int(h)


def _feather(mask: np.ndarray, feather_px: int) -> np.ndarray:
    kernel = max(1, int(feather_px))
    if kernel % 2 == 0:
        kernel += 1
    feathered = cv2.GaussianBlur(mask.astype(np.float32), (kernel, kernel), 0)
    return np.clip(feathered, 0.0, 1.0)


def _resolve_model_path(model_asset_path: str | None) -> Path:
    if model_asset_path:
        provided = Path(model_asset_path)
        if not provided.exists():
            raise ValueError(f"Model file does not exist: {provided}")
        return provided

    model_path = Path(__file__).parent / "models" / "selfie_segmenter.tflite"
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DEFAULT_MODEL_URL, model_path)
    return model_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replace a person in a video with a source image person.")
    parser.add_argument("--source-image", required=True, help="Path to source person image.")
    parser.add_argument("--target-video", required=True, help="Path to video where person is replaced.")
    parser.add_argument("--output-video", required=True, help="Path to write output video.")
    parser.add_argument("--mask-threshold", type=float, default=0.55, help="Segmentation confidence threshold.")
    parser.add_argument("--feather-px", type=int, default=21, help="Edge feather radius in pixels.")
    parser.add_argument("--bbox-smoothing", type=float, default=0.75, help="Bounding box smoothing factor.")
    parser.add_argument(
        "--min-mask-area-ratio",
        type=float,
        default=0.002,
        help="Minimum detected person mask area as ratio of frame area.",
    )
    parser.add_argument(
        "--model-asset-path",
        default=None,
        help="Optional local path to a MediaPipe segmentation model (.tflite).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = ReplacementConfig(
        mask_threshold=args.mask_threshold,
        feather_px=args.feather_px,
        bbox_smoothing=args.bbox_smoothing,
        min_mask_area_ratio=args.min_mask_area_ratio,
        model_asset_path=args.model_asset_path,
    )

    replacer = PersonReplacer(config)
    try:
        output = replacer.process_video(args.source_image, args.target_video, args.output_video)
        print(f"Saved output video to: {output}")
    finally:
        replacer.close()


if __name__ == "__main__":
    main()
