import subprocess
import sys

SCRIPT_ORDER = [
    "download_dataset.py",
    "extract_features.py",
    "train.py",
]


def run_script(script_name):
    print(f"\n=== Running {script_name} ===")
    subprocess.check_call([sys.executable, script_name])


def main():
    for script in SCRIPT_ORDER:
        run_script(script)

    print("\n=== Pipeline completed successfully! ===")
    print("You can now predict a new speech sample with:")
    print("  python predict.py <path_to_audio_file.wav>")


if __name__ == "__main__":
    main()
