# Real-time Object & Emotion Detector with YOLO, DeepFace & Gradio

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project demonstrates real-time object detection using **YOLO (Ultralytics)** and facial emotion recognition using **DeepFace**, launched via a user-friendly **Gradio** interface. The application captures video from a webcam or file, performs detections, and displays the annotated video stream in a separate **OpenCV** window.

## ✨ Features

*   **Dual Detection:** Performs both general object detection (YOLO) and facial emotion recognition (DeepFace) in real-time.
*   **Gradio Launcher:** Modern, web-based GUI (runs locally) to configure detection parameters before launch.
    *   Select video source (detected webcams, video file upload, video file path).
    *   Choose pre-trained YOLO models (e.g., YOLOv8n, YOLOv8s) or specify a custom path.
    *   Select processing device (CPU or CUDA GPU).
    *   Adjust initial detection thresholds (confidence, IoU).
    *   Optionally enable/disable emotion detection at launch.
    *   Configure initial display options.
*   **OpenCV Output:** Displays the annotated video stream in a dedicated OpenCV window for clear visualization.
*   **GPU Acceleration:** Leverages NVIDIA GPUs via CUDA (if configured correctly with PyTorch) for significantly improved performance. CPU fallback available.
*   **Interactive Controls (OpenCV Window):**
    *   Toggle Info Overlay (`i`)
    *   Toggle Emotion Detection (`e`, if initially enabled)
    *   Cycle YOLO Class Filters (`f`)
    *   Adjust Confidence Threshold (`+`/`-`)
    *   Save Screenshot (`s`)
    *   Record Video Clip (`r`)
    *   Quit (`q` or `ESC`)

##  Demo

**(Strongly Recommended: Replace this text with an embedded GIF showcasing the Gradio UI launching the OpenCV window with detections!)**

<!-- Example: <img src="assets/demo.gif" alt="Demo GIF" width="700"/> -->

## 💻 Technologies Used

*   Python 3.x
*   OpenCV (`opencv-python`)
*   PyTorch (`torch`, `torchvision`, `torchaudio`) - CUDA version required for GPU support
*   Ultralytics YOLO
*   DeepFace
*   Gradio
*   NumPy

## 🛠️ Setup & Installation

Follow these steps carefully to set up the project environment.

**1. Prerequisites:**
    *   Python 3.8+ recommended.
    *   Git installed.
    *   NVIDIA GPU with suitable drivers installed (if you want GPU acceleration).

**2. Clone the Repository:**
    ```bash
    git clone https://github.com/Juliansgith/Vision-v3.git
    cd Vision-v3
    ```

**3. Create and Activate Virtual Environment:**
    *   It's highly recommended to use a virtual environment.
    *   **On macOS and Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   **On Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

**4. ⭐ Install PyTorch with CUDA (Crucial for GPU Performance):**
    *   This step is essential for leveraging your NVIDIA GPU. If you skip this or install the wrong version, detection will run significantly slower on the CPU.
    *   **Uninstall Previous Versions (Recommended):**
        ```bash
        pip uninstall torch torchvision torchaudio -y
        ```
    *   **Go to the Official PyTorch Website:** [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
    *   **Select Your Configuration:**
        *   *PyTorch Build:* Stable
        *   *Your OS:* Windows/Linux/macOS
        *   *Package:* Pip
        *   *Language:* Python
        *   *Compute Platform:* **Select the CUDA version compatible with your NVIDIA drivers** (e.g., CUDA 11.8, CUDA 12.1). Check your driver documentation if unsure. If you don't have an NVIDIA GPU, select `CPU`.
    *   **Copy the Generated Command:** The website will provide a command like `pip3 install torch ... --index-url ...`.
    *   **Run the Command in your Activated Virtual Environment.**
        ```bash
        # === EXAMPLE ONLY - GET YOUR COMMAND FROM PYTORCH WEBSITE ===
        # Example for CUDA 12.1 on Windows/Linux:
        # pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        # =============================================================
        ```

**5. Install Other Dependencies:**
    *   Once PyTorch (with CUDA if desired) is installed, install the rest from `requirements.txt`:
        ```bash
        pip install -r requirements.txt
        ```
    *   This will install `opencv-python`, `ultralytics`, `numpy`, `gradio`, and `deepface` (which includes TensorFlow by default).
    *   **Note:** The first time you run the application, `ultralytics` (YOLO) and `deepface` may download pre-trained model weights, which can take a few moments depending on your internet connection.

**6. Verify Installation (Optional but Recommended):**
    *   Check PyTorch and CUDA:
        ```bash
        python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}'); print(f'cuDNN Version: {torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A'}')"
        ```
        Ensure `CUDA Available` is `True` if you intended to install the GPU version.

## 🚀 Usage

1.  **Activate your virtual environment** (if not already active):
    *   Windows: `.\venv\Scripts\activate`
    *   Linux/macOS: `source venv/bin/activate`
2.  **Run the Gradio Launcher:**
    ```bash
    python gui_launcher.py
    ```
3.  **Open the Local URL:** The console will output a URL like `http://127.0.0.1:7860`. Open this in your web browser.
4.  **Configure Settings in Gradio:**
    *   Select your desired Webcam or provide a video file source.
    *   Choose the YOLO model.
    *   Select the Processing Device (CUDA if available, otherwise CPU).
    *   Check the box to "Enable Emotion Detection" if desired.
    *   Adjust other parameters as needed.
5.  **Start Detection:** Click the "Start Detection (in OpenCV window)" button.
6.  **View Output:**
    *   A **separate OpenCV window** will open, displaying the video stream with bounding boxes for detected objects and emotion labels (if enabled) near faces.
    *   Check the console output for initialization messages and potential warnings/errors.
7.  **Interact with OpenCV Window:** Use the keyboard controls listed below while the OpenCV window is active.
8.  **Quit:** Press `q` or `ESC` in the OpenCV window to close it and stop the detection process. The Gradio status will update once the process finishes.

## ⌨️ Keyboard Controls (in OpenCV Window)

*   `q` or `ESC`: Quit the detection window.
*   `s`: Save the current annotated frame to the `output/images/` directory.
*   `r`: Start/Stop recording the annotated video stream to `output/videos/`.
*   `+` / `=`: Increase YOLO confidence threshold by 0.05.
*   `-` / `_`: Decrease YOLO confidence threshold by 0.05.
*   `i`: Toggle the on-screen informational display (FPS, model, etc.).
*   `f`: Cycle through predefined YOLO class filters (e.g., All -> Person -> Car -> Person+Car -> ...).
*   `e`: Toggle real-time emotion detection ON/OFF (only works if "Enable Emotion Detection" was checked in the Gradio launcher).

## 🔧 Troubleshooting

*   **Low FPS / Slow Performance:** Almost certainly due to PyTorch not using CUDA. Double-check your PyTorch installation using the verification command in the Setup section. Ensure you selected the correct CUDA version on the PyTorch website and installed it correctly.
*   **Webcam Not Detected / Errors during Camera Check:** Some systems have specific camera drivers (like OBS virtual cams or depth sensors) that can cause errors or warnings with OpenCV's default detection. The script tries multiple backends (`default`, `DSHOW` on Windows), but if your desired camera isn't listed, ensure its drivers are installed and no other application is using it exclusively. Check the console output for specific errors.
*   **DeepFace Errors:** On first run, DeepFace downloads models. Ensure you have an internet connection. Errors might also relate to the TensorFlow backend installation or incompatibilities. Check the DeepFace documentation or GitHub issues if problems persist.
*   **OpenCV Window Doesn't Appear/Freeze:** This can sometimes happen due to conflicts between GUI backends or threading issues. Running the detection in a separate thread (as currently implemented) usually solves this. Ensure no other conflicting GUI libraries are being inadvertently used. Check console logs for errors immediately after starting detection.

## 🚀 Future Improvements

*   Implement a reliable "Stop" button in the Gradio UI (requires more complex thread/process management).
*   Re-attempt robust video streaming directly within the Gradio UI (e.g., using `gr.Image` or investigating `gr.Video` further).
*   Add object tracking (e.g., using DeepSORT or BoT-SORT with YOLO).
*   Allow selection of DeepFace face detector backend in Gradio UI.
*   Optimize performance further (e.g., frame skipping, model quantization).

## 🙏 Acknowledgements

*   **Ultralytics:** For their excellent YOLO implementations. [https://ultralytics.com/](https://ultralytics.com/)
*   **DeepFace:** For the comprehensive face analysis library. [https://github.com/serengil/deepface](https://github.com/serengil/deepface)
*   **OpenCV:** The backbone for video capture and image manipulation. [https://opencv.org/](https://opencv.org/)
*   **Gradio:** For the easy-to-use GUI framework. [https://www.gradio.app/](https://www.gradio.app/)
*   **PyTorch Team:** For the deep learning framework. [https://pytorch.org/](https://pytorch.org/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.