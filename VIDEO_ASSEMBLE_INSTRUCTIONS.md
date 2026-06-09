Assemble a simple explanatory video from the generated slide images and a recorded narration.

1) Generate assets (already created by `generate_video_assets.py`):

```bash
python generate_video_assets.py
```

2) Record or generate narration audio (one file):
- You can record a voiceover matching `narration_script.txt` (recommended) and export to `narration.wav`.
- Alternatively, use a TTS service to create `narration.wav`.

3) Create a simple slideshow video using `ffmpeg` (needs ffmpeg installed):

- Create a `slides.txt` file with lines in this form (example):
  file 'video_assets/waveform.png'
  duration 4
  file 'video_assets/mfcc.png'
  duration 4
  file 'video_assets/training_history.png'
  duration 4
  file 'video_assets/confusion_matrix.png'
  duration 4

- Run these commands (adjust durations as desired):

```bash
ffmpeg -y -f concat -safe 0 -i slides.txt -vsync vfr -pix_fmt yuv420p slides.mp4
ffmpeg -y -i slides.mp4 -i narration.wav -c:v copy -c:a aac -shortest final_video.mp4
```

4) Tips
- If using Windows PowerShell, create `slides.txt` with a text editor and run the `ffmpeg` commands.
- Use higher durations for slide reading time if narration is longer.
