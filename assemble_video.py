import os
import subprocess
from gtts import gTTS
from imageio_ffmpeg import get_ffmpeg_exe

ASSETS_DIR = 'video_assets'
NARRATION_SCRIPT = 'narration_script.txt'
NARRATION_WAV = 'narration.wav'
OUTPUT_VIDEO = 'final_video.mp4'
SLIDES_FILE = 'slides.txt'
SLIDES_VIDEO = 'slides.mp4'

SLIDES = [
    os.path.join(ASSETS_DIR, 'waveform.png'),
    os.path.join(ASSETS_DIR, 'mfcc.png'),
    os.path.join(ASSETS_DIR, 'training_history.png'),
    os.path.join(ASSETS_DIR, 'confusion_matrix.png'),
]
DURATIONS = [4, 4, 4, 4]


def synthesize_narration(script_path, out_wav):
    with open(script_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tts = gTTS(text=text, lang='en')
    tts.save(out_wav)


def write_slides_file(slides, durations, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for image, duration in zip(slides, durations):
            f.write(f"file '{image}'\n")
            f.write(f"duration {duration}\n")
        # FFmpeg concat demuxer requirement: repeat last file at end
        f.write(f"file '{slides[-1]}'\n")


def run_ffmpeg(args):
    ffmpeg_exe = get_ffmpeg_exe()
    command = [ffmpeg_exe] + args
    print('Running:', ' '.join(command))
    subprocess.run(command, check=True)


def assemble(slides_txt, narration_wav, out_video):
    run_ffmpeg([
        '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', slides_txt,
        '-vsync', 'vfr',
        '-pix_fmt', 'yuv420p',
        SLIDES_VIDEO,
    ])
    if os.path.exists(narration_wav):
        run_ffmpeg([
            '-y',
            '-i', SLIDES_VIDEO,
            '-i', narration_wav,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            out_video,
        ])
    else:
        os.replace(SLIDES_VIDEO, out_video)


def main():
    if not os.path.exists(ASSETS_DIR):
        raise FileNotFoundError(f'{ASSETS_DIR} directory does not exist')

    if not os.path.exists(NARRATION_SCRIPT):
        raise FileNotFoundError(f'{NARRATION_SCRIPT} not found')

    if not os.path.exists(NARRATION_WAV):
        print('Synthesizing narration to', NARRATION_WAV)
        synthesize_narration(NARRATION_SCRIPT, NARRATION_WAV)
    else:
        print(NARRATION_WAV, 'already exists; using existing file.')

    print('Writing slides list to', SLIDES_FILE)
    write_slides_file(SLIDES, DURATIONS, SLIDES_FILE)

    print('Assembling slides into video...')
    assemble(SLIDES_FILE, NARRATION_WAV, OUTPUT_VIDEO)
    print('Video written to', OUTPUT_VIDEO)


if __name__ == '__main__':
    main()
