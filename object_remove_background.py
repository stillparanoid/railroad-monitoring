import os
from rembg import remove
from config import DATA_FOLDER
from PIL import Image
import io

# Path to the input directory containing images.
BATCH_INPUT_DIR = os.path.join(DATA_FOLDER, "raw_objects")

# Path to the output directory where processed images will be saved.
BATCH_OUTPUT_DIR = os.path.join(DATA_FOLDER, "prepared_objects")

# Define the image file extensions that the script will process.
SUPPORTED_FORMATS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

def remove_background_and_center(input_path, output_path):
    """
    Removes the background from a single image, centers the object, and saves the result.

    Args:
        input_path (str): Path to the input image.
        output_path (str): Path to save the output image with the background removed and object centered.
    """
    try:
        with open(input_path, "rb") as input_file:
            input_data = input_file.read()
        output_data = remove(input_data)

        # Load the output data into a PIL Image
        image = Image.open(io.BytesIO(output_data)).convert("RGBA")

        # Get the bounding box of the non-transparent pixels
        bbox = image.getbbox()
        if bbox:
            # Crop the image to the bounding box
            cropped_image = image.crop(bbox)
        else:
            # If bbox is None, the image is fully transparent
            cropped_image = image

        # Optional: If you want to resize the image to a specific size while keeping the object centered
        # For example, resize to 512x512
        # new_size = (512, 512)
        # centered_image = Image.new("RGBA", new_size, (0, 0, 0, 0))
        # cropped_image.thumbnail(new_size, Image.ANTIALIAS)
        # paste_position = (
        #     (new_size[0] - cropped_image.width) // 2,
        #     (new_size[1] - cropped_image.height) // 2
        # )
        # centered_image.paste(cropped_image, paste_position)
        # final_image = centered_image

        # If you want the image to be tightly cropped without additional padding
        final_image = cropped_image

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the final image in PNG format to preserve transparency
        final_image.save(output_path, format="PNG")
        print(f"Processed: {input_path} -> {output_path}")
    except Exception as e:
        print(f"Failed to remove background from {input_path}. Error: {e}")

def batch_remove(input_dir, output_dir):
    """
    Removes backgrounds from all supported images in a directory and its subdirectories,
    centers the extracted objects, and preserves the directory structure in the output directory.

    Args:
        input_dir (str): Path to the input directory containing images.
        output_dir (str): Path to save the output images with backgrounds removed and objects centered.
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    processed_count = 0
    skipped_count = 0

    # Walk through all subdirectories and files in the input directory
    for root, dirs, files in os.walk(input_dir):
        # Compute the relative path from the input directory
        relative_path = os.path.relpath(root, input_dir)
        # Determine the corresponding directory in the output directory
        output_root = os.path.join(output_dir, relative_path)

        for filename in files:
            if filename.lower().endswith(SUPPORTED_FORMATS):
                input_path = os.path.join(root, filename)
                # Change output format to PNG to preserve transparency
                base_filename, _ = os.path.splitext(filename)
                output_filename = base_filename + ".png"
                output_path = os.path.join(output_root, output_filename)
                remove_background_and_center(input_path, output_path)
                processed_count += 1
            else:
                print(f"Skipped (unsupported format): {os.path.join(root, filename)}")
                skipped_count += 1

    print("\nBatch processing completed.")
    print(f"Total images processed: {processed_count}")
    if skipped_count > 0:
        print(f"Total files skipped (unsupported formats): {skipped_count}")

def main():
    print("=== Batch Image Background Remover and Centering Tool ===\n")
    print(f"Input Directory : {BATCH_INPUT_DIR}")
    print(f"Output Directory: {BATCH_OUTPUT_DIR}\n")
    batch_remove(BATCH_INPUT_DIR, BATCH_OUTPUT_DIR)

if __name__ == "__main__":
    main()
