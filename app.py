from __future__ import annotations

from datetime import datetime
from pathlib import Path

import gradio as gr

from person_replacer import PersonReplacer, ReplacementConfig

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_person_replacement(
    source_image_path: str,
    target_video_path: str | dict,
    mask_threshold: float,
    feather_px: int,
    bbox_smoothing: float,
    min_mask_area_ratio: float,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str]:
    if not source_image_path:
        raise gr.Error("Please upload a source person image.")
    resolved_video_path = _resolve_video_path(target_video_path)
    if not resolved_video_path:
        raise gr.Error("Please upload a target video.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"replaced_{timestamp}.mp4"
    config = ReplacementConfig(
        mask_threshold=mask_threshold,
        feather_px=int(feather_px),
        bbox_smoothing=bbox_smoothing,
        min_mask_area_ratio=min_mask_area_ratio,
    )

    replacer = PersonReplacer(config)
    try:
        def _progress(frame_idx: int, total_frames: int) -> None:
            progress(min(frame_idx / max(1, total_frames), 1.0), desc="Processing video frames...")

        saved_path = replacer.process_video(
            source_image_path=source_image_path,
            target_video_path=resolved_video_path,
            output_video_path=output_path,
            progress_callback=_progress,
        )
    except Exception as exc:
        raise gr.Error(f"Processing failed: {exc}") from exc
    finally:
        replacer.close()

    details = (
        "Finished person replacement.\n"
        f"Output: {saved_path}\n"
        "Tip: Use a clear, front-facing source photo and a single-person target shot for best results."
    )
    return str(saved_path), details


def _resolve_video_path(video_value: str | dict | None) -> str | None:
    if isinstance(video_value, str):
        return video_value
    if isinstance(video_value, dict):
        return video_value.get("path") or video_value.get("video")
    return None


with gr.Blocks(title="Video Person Replacer") as demo:
    gr.Markdown(
        """
        # Video Person Replacer
        Upload an image of a person and a target video.
        The app detects the primary person in each video frame and composites the source person over them.
        """
    )

    with gr.Row():
        source_image = gr.Image(type="filepath", label="Source Person Image")
        target_video = gr.Video(label="Target Video")

    with gr.Accordion("Advanced Settings", open=False):
        mask_threshold = gr.Slider(0.1, 0.95, value=0.55, step=0.01, label="Mask Threshold")
        feather_px = gr.Slider(1, 61, value=21, step=2, label="Edge Feather (px)")
        bbox_smoothing = gr.Slider(0.0, 0.98, value=0.75, step=0.01, label="BBox Smoothing")
        min_mask_area_ratio = gr.Slider(
            0.0001,
            0.05,
            value=0.002,
            step=0.0001,
            label="Minimum Person Area Ratio",
        )

    run_btn = gr.Button("Replace Person", variant="primary")
    output_video = gr.Video(label="Output Video")
    output_text = gr.Textbox(label="Result", lines=4)

    run_btn.click(
        fn=run_person_replacement,
        inputs=[
            source_image,
            target_video,
            mask_threshold,
            feather_px,
            bbox_smoothing,
            min_mask_area_ratio,
        ],
        outputs=[output_video, output_text],
    )


if __name__ == "__main__":
    demo.launch()
