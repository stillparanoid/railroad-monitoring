# Railroad Monitoring with UAVs and Deep Learning

This repository contains the implementation of a system for monitoring railroad tracks using Unmanned Aerial Vehicles (UAVs) and deep learning techniques. The system processes aerial images captured by UAVs to detect and classify hazardous objects and anomalies on the tracks, including debris, vegetation encroachment, and other safety hazards, enabling proactive maintenance and risk mitigation.

## Project Components

The project consists of two main components:

### 1. Synthetic Data Generation

This component generates a comprehensive dataset by augmenting real-world video frames with synthetic hazardous objects. The resulting dataset is used to train the object detection model, ensuring robustness across diverse scenarios.

**Steps:**

- **Frame Extraction from Videos:** Extract frames from Creative Commons (CC) YouTube videos recorded from locomotive cabins, capturing railroad tracks and their surroundings.
- **Hazardous Object Image Retrieval:** Download images from Google containing potential hazards (e.g., cars, animals, broken signs, metal scrap, coal heaps) that may appear on railroad tracks.
- **Background Removal:** Remove backgrounds from the downloaded hazard images to isolate the objects, enabling seamless integration into the extracted frames.
- **Object Augmentation and Placement:** Place the isolated objects onto the frames with augmentations (e.g., scaling, rotation), recording their coordinates as ground truth for model training.

### 2. Object Detection

This component trains a Convolutional Neural Network (CNN) to detect and classify hazardous objects and anomalies within aerial images captured by UAVs.

**Steps:**

- **Dataset Preparation:** Use the synthetic dataset from the previous component, consisting of augmented frames with labeled hazardous objects.
- **Model Training:** Train a pretrained CNN (e.g., from ImageNet) using Keras, leveraging transfer learning to enhance performance with limited data.
- **Model Evaluation:** Validate the model’s accuracy and reliability using a separate dataset, ensuring effective detection of hazards.
- **Inference and Deployment:** Apply the trained model to perform real-time detection on new UAV-captured aerial images, supporting railroad monitoring efforts.

## Configuration

To configure the project environment, complete the following steps:

1. **Set Up a Virtual Environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
    ```

2. **Create the `config.ini` File:**

   In the root directory, create a `config.ini` file with the following structure:

   ```ini
   [DEFAULT]
   google_api_key = YOUR_GOOGLE_API_KEY
   custom_search_engine_id = YOUR_CUSTOM_SEARCH_ENGINE_ID
   path_to_data_folder = YOUR_PATH_TO_DATA_FOLDER
   ```
  
  Note: Obtain your Google API key and Custom Search Engine ID from the Google Cloud Console and Google Custom Search Engine, respectively.

## Usage

1. **Prepare the Data Folder:**

* Create a `categories.csv` file in the data folder specified in `config.ini` with this format:
  ```csv
   Object name,scale factor
   aluminum_can,0.05
   ...
   ```
  
   The scale factor adjusts the size of object images during augmentation.

* Create a `raw_videos` subfolder in the data folder and add videos for frame extraction.

2. **Extract Frames from Videos:**

   Run the following command to extract frames:

   ```bash
   python background_capture_frames.py
   ```

   Extracted frames will be saved in `<data_folder>/<video_name>/extracted_frames`.

   Example of result:

   <img src="examples/readme/data/extracted_frames/video_1/frame_000001.jpg" width="50%">
   
3. **Download Hazardous Object Images:**

   Run this command to retrieve object images from Google:

   ```bash
   python object_download.py
   ```

   Enter the object category and the number of images to download (1-50). Images will be saved in `<data_folder>/<category>/raw_objects`.
   
   Example of result:

   <img src="examples/readme/data/raw_objects/aluminum_can/image_2.jpg" width="50%">

5. **Remove Backgrounds from Object Images:** 

   Run this command to isolate objects by removing backgrounds:

   ```bash
   python object_remove_background.py
   ```

   Processed images will be saved in `<data_folder>/<category>/prepared_objects`.
   
   Example of result:

   <img src="examples/readme/data/prepared_objects/aluminum_can/image_2.png" width="25%">

5. **Augment and Place Objects on Frames:**

   Run this command to generate synthetic images by placing objects on frames:

   ```bash
   python combine_background_and_object.py
   ```

   Augmented frames will be saved in `<data_folder>/synthetic_images`.
   
   Example of result:

   <img src="examples/readme/data/synthetic_images/frame_000001_aluminum_can_image_2.png" width="50%">
