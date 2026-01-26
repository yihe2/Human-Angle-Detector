# Real-Time Human Angle Detection System

## Overview
This project implements a computer vision system that utilizes a standard webcam to detect human presence and calculate the angular deviation of the subject relative to the camera's center axis.

By leveraging **Google MediaPipe Pose** for high-fidelity landmark detection and **OpenCV** for real-time image processing, the system approximates the subject's center of mass (midpoint between hips) and computes the precise angle in degrees. This mimics the tracking capabilities found in hardware like the Microsoft Kinect but runs on standard consumer hardware.

## Key Features
* **Real-time Pose Estimation:** Detects 33 separate body landmarks at high frame rates.
* **Dynamic Angle Calculation:** Computes the angle (theta) between the camera's optical axis and the subject using a calibrated Horizontal Field of View (HFOV).
* **Visual Feedback:** Renders a skeletal overlay, a center reference line, and the live angle metric directly onto the video feed.

## Prerequisites
* **OS:** Windows, macOS, or Linux (Raspberry Pi OS 64-bit supported)
* **Python:** Version 3.8 or higher (Tested on 3.11)
* **Hardware:** Standard Laptop Webcam or USB Camera

## Installation & Setup

To ensure dependency stability and avoid conflicts with global Python packages, this project requires a **Virtual Environment**.

### 1. Clone or Initialize Project
Open your terminal and navigate to the project folder:
```powershell
cd path/to/your-project-folder
```

### 2. Create the Virtual Environment
Generate a clean isolated environment named `venv`:
```powershell
python -m venv venv
```

### 3. Activate the Environment
You must activate the environment every time you work on this project.

* **Windows (PowerShell):**
    ```powershell
    .\venv\Scripts\Activate
    ```
* **Mac/Linux:**
    ```bash
    source venv/bin/activate
    ```
*(You will see `(venv)` appear at the start of your command prompt)*

### 4. Install Dependencies
Install the specific versions of the libraries required to ensure compatibility between MediaPipe and NumPy.

```powershell
pip install mediapipe==0.10.14 opencv-python "numpy<2"
```

## Execution

Ensure your webcam is not being used by another application (Zoom, Teams, etc.).

1.  **Activate the environment** (if not already active):
    ```powershell
    .\venv\Scripts\Activate
    ```

2.  **Run the script:**
    ```powershell
    python MachineVision.py
    ```

3.  **Controls:**
    * Press **`q`** on your keyboard to quit the program and close the camera window.

## Methodology

### Mathematical Model
The angle calculation assumes a standard webcam Horizontal Field of View (HFOV) of **60°**.

1.  **Center Detection:** The system calculates the midpoint between the `LEFT_HIP` and `RIGHT_HIP` landmarks.
2.  **Pixel Offset:** It measures the distance (Delta x) between the person's center and the frame's center.
3.  **Angle Derivation:**
    Angle = (Delta x / Frame Width) * HFOV

### Dependencies

| Library | Purpose |
| :--- | :--- |
| **OpenCV (`opencv-python`)** | Image capture, frame manipulation, and graphical overlay rendering. |
| **MediaPipe (`mediapipe`)** | Machine Learning pipeline for pose estimation and landmark inference. |
| **NumPy (`numpy`)** | High-performance mathematical operations (Legacy version `<2.0` required). |

## Limitations
* **Single Person Only:** This version uses single-pose estimation. If multiple people are in the frame, it will track the most prominent person or flicker between them.