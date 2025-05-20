import os

import cv2
from tqdm import tqdm
import config

VIDEO_FOLDER = os.path.join(config.DATA_FOLDER, "raw_videos")
EXTRACTED_FRAMES_FOLDER = os.path.join(config.DATA_FOLDER,
                                       "extracted_frames")
FRAMES_INTERVAL = 300


def extract_frames(video_path, output_folder, frame_interval=30):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    extracted_frame_count = 0

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a progress bar
    pbar = tqdm(total=total_frames, desc="Processing frames")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder,
                                          f"frame_{extracted_frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_frame_count += 1

        frame_count += 1
        pbar.update(1)  # Update the progress bar

    cap.release()
    pbar.close()  # Close the progress bar
    print(f"Extracted {extracted_frame_count} frames from the video.")


if __name__ == "__main__":
    for video_file in os.listdir(VIDEO_FOLDER):
        video_path = os.path.join(VIDEO_FOLDER, video_file)
        output_folder = os.path.join(EXTRACTED_FRAMES_FOLDER,
                                     os.path.splitext(video_file)[0])
        extract_frames(video_path, output_folder, FRAMES_INTERVAL)
