import subprocess
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

scripts = [
    "background_capture_frames.py",
    "object_download.py",
    "object_remove_background.py",
    "object_augment.py",
    "combine_background_and_object.py"
]

for script in scripts:
    script_path = os.path.join(BASE_DIR, script)
    print(f"\n▶️ Запуск: {script_path}")
    subprocess.run(["python", script_path])
