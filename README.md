# Real-time Multi-Analysis Detector with YOLO, DeepFace, SORT & ByteTrack

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project provides a comprehensive real-time detection system featuring:
*   **Object Detection:** Using **YOLO (Ultralytics)** with support for PyTorch (`.pt`) and TensorRT (`.engine`) models.
*   **Person Tracking:** Implementing **SORT (Simple Online and Realtime Tracking)** and **ByteTrack** to assign and maintain IDs for detected persons.
*   **Facial Analysis:** Leveraging **DeepFace** for emotion, age, and gender recognition.
*   **Association:** Detected faces and their analyses are associated with tracked persons based on IoU.
*   **YouTube URL Support:** Directly process videos from YouTube URLs (requires `yt-dlp` and `ffmpeg`).
*   **Gradio Launcher:** A user-friendly web-based GUI (runs locally) to configure and launch the detection pipeline.
*   **OpenCV Output:** The annotated video stream is displayed in a separate, interactive OpenCV window.

## ✨ Features

*   **Multi-Modal Analysis:** Combines object detection, person tracking, and detailed facial analysis.
*   **Versatile Input Sources:**
    *   Local video files (upload or path).
    *   Connected webcams (auto-detected with descriptive names).
    *   Directly from YouTube video URLs.
*   **Gradio Launcher:**
    *   User-friendly interface to select input source, YOLO model, and processing device.
    *   Fine-tune parameters for YOLO, SORT/ByteTrack, DeepFace, and display options.
    *   Toggle features like emotion/age-gender detection and person tracking IDs.
*   **Interactive OpenCV Output:**
    *   Displays YOLO detections, track IDs for persons, and associated facial analysis (emotion, age, gender) next to tracked individuals.
    *   Runtime controls for: Info Overlay (`i`), Emotion Detection (`e`), Age/Gender Detection (`a`), Person Tracking IDs (`t`), YOLO Class Filters (`f`), Confidence Threshold (`+`/`-`), Screenshot (`s`), Record (`r`), Quit (`q`/`ESC`).
*   **GPU Acceleration:** Optimized for NVIDIA GPUs via CUDA (for PyTorch) and TensorRT (for `.engine` models). CPU fallback.
*   **Organized Output:** Saves screenshots, recordings, and temporary video downloads to structured directories (`output/`, `temp_videos/`).

## 🎬 Demo

Watch a quick demonstration of the system in action:
[Real-time Multi-Analysis Detector Demo](https://www.youtube.com/watch?v=YzcawvDGe4Y)

*(This demo showcases the Gradio UI for configuration, followed by the OpenCV window displaying object detections, person track IDs, and associated facial analysis on a sample video.)*

## 💻 Technologies Used

*   Python 3.8+
*   OpenCV (`opencv-python`)
*   PyTorch (`torch`, `torchvision`, `torchaudio`) - CUDA version for GPU
*   Ultralytics YOLO
*   DeepFace (and its TensorFlow backend)
*   SORT (custom implementation with `filterpy` and `scipy`)
*   ByteTrack (integrated as a local submodule from YOLOX, requires `cython-bbox`)
*   `yt-dlp` (for YouTube URL processing)
*   FFmpeg (recommended for `yt-dlp` for optimal format handling and merging)
*   Gradio
*   NumPy

## 🛠️ Setup & Installation

Follow these steps carefully.

**1. Prerequisites:**
    *   Python 3.8 - 3.10 (TensorFlow 2.10 used by DeepFace has constraints).
    *   Git.
    *   NVIDIA GPU with suitable drivers (for GPU acceleration).
    *   C++ Compiler (required for `cython-bbox`, e.g., MSVC on Windows, GCC on Linux).
    *   **FFmpeg:** (Highly Recommended, especially for YouTube URL processing with `yt-dlp`). Download from [ffmpeg.org](https://ffmpeg.org/download.html) and ensure it's added to your system's PATH.

**2. Clone Repository:**
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git # Replace with your repo URL
    cd your-repo-name
    ```

**3. Create and Activate Virtual Environment:**
    *   (Highly Recommended)
    *   macOS/Linux: `python3 -m venv venv && source venv/bin/activate`
    *   Windows: `python -m venv venv && .\venv\Scripts\activate`

**4. ⭐ Install PyTorch with CUDA (Crucial for GPU with `.pt` models):**
    *   Go to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).
    *   Select: Stable, your OS, Pip, Python, **your CUDA version** (e.g., 11.8 or 12.1). If no NVIDIA GPU, select CPU.
    *   Run the generated `pip install torch...` command in your activated venv.
        ```bash
        # === EXAMPLE ONLY - GET YOUR COMMAND FROM PYTORCH WEBSITE ===
        # Example for CUDA 12.1:
        # pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        # =============================================================
        ```

**5. Install Other Dependencies (from `requirements.txt`):**
    ```bash
    pip install -r requirements.txt
    ```
    This installs `opencv-python`, `ultralytics`, `numpy<1.24`, `gradio`, `deepface` (which pulls `tensorflow==2.10.0`), `filterpy`, `scipy`, `protobuf<3.20`, `cython-bbox`, and `yt-dlp`.

    *   **Note:** First run may download model weights for YOLO and DeepFace.
    *   **`cython-bbox` troubleshooting:** If installation fails, ensure you have a C++ compiler set up correctly. On Windows, this might involve installing "Build Tools for Visual Studio". On Linux, `sudo apt-get install build-essential python3-dev`.

**6. (Optional) Build TensorRT Engines for YOLO:**
    *   If you have an NVIDIA GPU and want maximum YOLO performance, convert `.pt` models to TensorRT `.engine` files.
    *   Ensure TensorRT is installed (often comes with CUDA Toolkit or can be installed separately).
    *   Run the provided script:
        ```bash
        python transformTensorRT.py
        ```
        This will create `yolov8n.engine`, `yolov8x.engine`, etc., in your project root. The Gradio UI will list these.

**7. Verify Installation (Optional):**
    ```bash
    python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}'); import tensorflow as tf; print(f'TensorFlow: {tf.__version__}, GPU Available: {tf.config.list_physical_devices('GPU')}')"
    yt-dlp --version # Should output the yt-dlp version
    ffmpeg -version  # Should output FFmpeg version information
    ```
    Ensure CUDA/GPU is available for PyTorch and TensorFlow if you expect GPU acceleration, and that `yt-dlp` and `ffmpeg` are recognized.

## 🚀 Usage

1.  **Activate virtual environment.**
2.  **Run the Gradio Launcher:**
    ```bash
    python gui_launcher.py
    ```
3.  **Open the Local URL** (e.g., `http://127.0.0.1:7860` or `http://127.0.0.1:7861`) in your browser.
4.  **Configure Settings in Gradio:**
    *   **Select Source Type:** Choose between webcam, "Use Uploaded Video File", or "Use Video Path or URL".
    *   If using URL, paste the YouTube video URL or a direct video link.
    *   Select YOLO model (choose `.engine` for TensorRT speed if available).
    *   Adjust detection, tracking (SORT or ByteTrack), and facial analysis parameters.
    *   Enable/disable features as needed.
5.  **Start Detection:** Click the "🚀 Start Detection" button. (If using a YouTube URL, allow time for download).
6.  **View Output:** An OpenCV window will display the annotated video.
7.  **Interact:** Use keyboard controls in the OpenCV window.
8.  **Quit:** Press `q` or `ESC` in the OpenCV window.

**Command Line Interface (CLI):**
For non-GUI operation, use `main.py`:
```bash
# Example with webcam
python main.py --source 0 --model yolov8s.engine --enable-emotion --show-track-id 

# Example with local video file
python main.py --source "/path/to/your/video.mp4" --tracker-type bytetrack

# See 'python main.py --help' for all options
# Note: Direct YouTube URL processing via CLI would require similar yt-dlp logic in main.py