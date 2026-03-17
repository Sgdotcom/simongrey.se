# Video Person Replacer App

This repository now includes a Python app that replaces the primary person in a video with a person from a source image.

## How it works

1. The app segments the person in the source image.
2. For each frame in the target video, it segments the main person.
3. It tracks a smoothed bounding box for that person.
4. It composites the source person cutout onto the target area.

This is optimized for single-person scenes and gives best results when:

- The source image has a clearly visible full/upper body.
- The target video has one dominant person.
- Lighting and camera angle are not drastically different.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run web app (Gradio)

```bash
python3 app.py
```

If you are running on a remote/cloud machine and cannot open localhost from your own browser, run:

```bash
python3 app.py --host 0.0.0.0 --share
```

This prints a public Gradio URL you can open directly.

Then open the local Gradio URL in your browser, upload:

- **Source Person Image**
- **Target Video**

Click **Replace Person** and download/play the output video.

## Run from CLI

```bash
python3 person_replacer.py \
  --source-image path/to/person.jpg \
  --target-video path/to/input.mp4 \
  --output-video path/to/output.mp4
```

Optional quality controls:

- `--mask-threshold` (default `0.55`)
- `--feather-px` (default `21`)
- `--bbox-smoothing` (default `0.75`)
- `--min-mask-area-ratio` (default `0.002`)

## Notes

- Output videos are written to `outputs/` by default in the web app.
- Audio passthrough is not included in this baseline implementation.
- The segmentation model file is auto-downloaded to `models/selfie_segmenter.tflite` on first run.
