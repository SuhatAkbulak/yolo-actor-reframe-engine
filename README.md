# Actor Reframe Engine

**AI-powered horizontal-to-vertical video reframing** — turn landscape videos into TikTok, Reels, or Shorts format while keeping the main subject in frame.

ClipSage analyzes your video with pose detection and scene understanding, picks the main character, and crops each scene so the focus stays centered in a 9:16 vertical output. Original audio is preserved.

[![ClipSage Demo](https://img.youtube.com/vi/HMwCxGk5Eqk/maxresdefault.jpg)](https://youtu.be/HMwCxGk5Eqk)

*Click the image to watch the demo on YouTube.*

---

## Features

- **Main actor detection** — Scores people by size, screen position, and face visibility to identify the primary subject
- **Scene-aware cropping** — Detects scene changes via histogram comparison and chooses a stable crop per scene
- **Vertical 9:16 output** — Produces Reels/Shorts-ready video with H.264 video and AAC audio
- **YOLOv8 Pose** — Person tracking and keypoints for robust subject following

---

## How It Works

1. **Analysis**
   - Runs YOLOv8 Pose on every frame for person detection and tracking
   - Detects scene cuts using HSV histogram correlation
   - Assigns each person a score (area × centrality × face bonus) and selects the main actor
   - For each scene, picks the dominant person (with extra weight for the main actor)
   - Writes per-frame metadata to `video_analysis.json` (focus coordinates, faces, objects)

2. **Rendering**
   - Reads `video_analysis.json` and the original video
   - For each scene, uses the median focus X to define a fixed horizontal crop
   - Crops to 9:16 (height unchanged, width = height × 9/16) and writes frames
   - Runs FFmpeg to encode with `libx264` and mux original audio into `final_vertical_video_with_audio.mp4`

---

## Requirements

- **Python 3.8+**
- **CUDA** (optional, for GPU)
- **FFmpeg** (must be on `PATH` for final encode)

### Python dependencies

```
ultralytics
torch
opencv-python
numpy
tqdm
```

---

## Setup

```bash
pip install ultralytics torch opencv-python numpy tqdm
```

Ensure FFmpeg is installed and available in your terminal.

---

## Usage

1. **Input video**  
   Place your horizontal video as `input_video.mp4` in the project directory (or change `video_path` in the script).  
   The original script also shows a `wget` example for fetching a sample URL.

2. **Run analysis + render**  
   Execute the full script (e.g. `Reframe.py`). It will:
   - Run pose detection and scene detection
   - Save `video_analysis.json`
   - Crop and encode to `final_vertical_video_with_audio.mp4`

3. **Output**  
   - `video_analysis.json` — frame-level metadata (scenes, focus, main actor, faces/objects)  
   - `final_vertical_video_with_audio.mp4` — vertical 9:16 video with original audio

---

## Notes

- The script includes Colab-oriented snippets (`!wget`, `!pip`, `IPython.display`). For local use, remove or replace those with your own input/display logic.
- For long videos, consider processing in chunks or lowering input resolution to speed up analysis.
- Main actor ID and scene leaders are derived from tracking IDs; consistent tracking improves reframing quality.

---

## To Do

- **SAM (Segment Anything Model) integration** — Integrate Meta’s SAM for pixel-precise subject selection and masking. Allow users to pick focus by click or box prompt so the crop follows the chosen object/person across frames; improves focus quality beyond bounding-box heuristics.
- **LLM-guided focus selection** — Use a large language model to choose focus subjects from scene context, dialogue, or user intent (e.g. “follow the speaker” or “emphasize the person on the left”) instead of heuristic scoring alone.
- **Multi-crop planning in one run** — Plan several crop scenarios (e.g. different aspect ratios, focus targets, or safe zones) from a single analysis pass, so one `video_analysis.json` can drive multiple export presets without re-running detection.

---

## License

Use and modify as you like. If you use YOLO/Ultralytics, check their license terms.

---

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for pose estimation and tracking
- OpenCV and FFmpeg for video I/O and encoding
