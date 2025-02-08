# Tennis Analysis Project

This project implements a tennis analysis system using machine learning, computer vision, and deep learning techniques as demonstrated in [this YouTube video](https://www.youtube.com/watch?v=L23oIHZE14w) by Abdullah Tarek. It utilizes YOLO for object detection (players and tennis balls), object trackers for tracking objects across frames, and potentially a custom convolutional neural network for court keypoint detection (implementation details may vary).

## Features

*   **Object Detection:** Uses YOLO11 (for general objects like players) and a fine-tuned YOLOv5 (for tennis balls) to detect objects in images and videos.
*   **Object Tracking:** Tracks detected players and tennis balls across frames using object trackers.
*   **Video Processing:** Uses OpenCV (cv2) to read, manipulate, and save video files.
*   **Data Analysis:** Analyzes detection data to develop features and derive insights, potentially including player speed, ball speed, distances covered, and shot analysis.

## Prerequisites

Before running the project, ensure you have the following installed:

*   **Python 3.7+**
*   **pip** (Python package installer)


*Important Notes regarding dependency versions:*

*   *Older OpenCV Version Recommended:* Using a newer version of OpenCV may lead to compatibility issues and errors due to breaking changes in the API. Stick to the exact version specified.

*   *NumPy:* Some computer vision libraries like OpenCV are highly dependent on NumPy. Make sure you install a NumPy version (like `numpy==1.26.4`) compatible with your OpenCV to avoid errors.

*   *Check CUDA and cuDNN versions:* Ensure that the CUDA and cuDNN versions match the compatibility requirements of PyTorch. You can find the compatible versions in the PyTorch documentation. Incorrect versions can cause errors when using GPU acceleration.

## Installation

1.  **Clone the repository:**

    ```
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment (recommended):**

    ```
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the required packages:**

    ```
    pip install -r requirements.txt
    ```


## Usage

1.  **Prepare your input video:** Place your tennis match video in the `input_videos` directory and rename it to `input_video.mp4` (or adjust the filename in `main.py`).

2.  **Run the analysis script:**

    ```
    python main.py
    ```

3.  **View the output:** The processed video with object detections, tracking, and keypoint overlays will be saved as `output_videos/video.avi`.

## Future Enhancements

*   Implement more advanced data analysis techniques to extract deeper insights from the video.
*   Improve the accuracy of the tennis ball detection model by using better models such as TrackNet.
*   Explore different object tracking algorithms to improve tracking performance.
*   Add support for different video formats.
*   Develop a user interface for easy analysis and visualization.
*   Incorporate techniques like homography to create a more accurate top-down view of the court.

