# OpenCV Camera Calibration Tool

A comprehensive Python-based camera calibration tool with a graphical user interface for calibrating cameras using OpenCV.

## Features

- **Camera Connection**: Connect to any available camera device
- **Flexible Image Capture**:
  - Single image capture
  - Timer-based automatic capture (configurable interval)
- **Real-time Feature Detection**: Visual feedback showing detected chessboard corners as semi-transparent dots that scale with image size
- **Image Review**:
  - List view of all captured images
  - Individual image viewer with detected corners highlighted
  - Reprojection error visualization with bar chart showing per-image errors and average error line
- **Multiple Calibration Models**:
  - Pinhole (Standard) camera model
  - Fisheye camera model
  - Omnidirectional (360) camera model
  - Pinhole with distortion coefficients
- **Results Display**: Comprehensive calibration results including:
  - Camera matrix
  - Distortion coefficients
  - Reprojection errors (mean, min, max, per-image)
  - Calibration metadata
- **Export**: Save calibration results as JSON file

## Installation

1. Install Python 3.7 or higher
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python camera_calibration_tool.py
```

2. **Setup**:
   - Select your camera from the dropdown
   - Click "Connect Camera"
   - Configure chessboard size (width x height) - this should match your physical chessboard pattern
   - Set square size in millimeters (the size of one square on your chessboard)

3. **Capture Images**:
   - Position the chessboard in front of the camera
   - You should see green dots appear on detected corners
   - Use "Capture Single Image" for manual capture
   - Or use "Start Timer Capture" to automatically capture images at set intervals
   - Capture at least 10-20 images from different angles and positions for best results

4. **Review & Calibrate**:
   - Click "Review Images & Calibrate" when you have enough images
   - Review each image and check the reprojection error chart
   - Select your calibration model
   - Click "Finish Calibration"

5. **View Results**:
   - Review the calibration results
   - Save results as JSON if needed

## Calibration Models

- **Pinhole (Standard)**: Standard camera model for regular cameras
- **Fisheye**: For fisheye/wide-angle lenses
- **Omni (360)**: For omnidirectional/360-degree cameras
- **Pinhole with Distortion Coefficients**: Standard model with full distortion modeling

## Tips for Best Results

1. Use a high-quality printed chessboard pattern
2. Ensure good lighting conditions
3. Capture images from various angles and distances
4. Make sure the entire chessboard is visible in each image
5. Aim for at least 10-15 images, more is better
6. Lower reprojection errors indicate better calibration (typically < 0.5 pixels is good)

## Output Format

The JSON export includes:
- Camera matrix (3x3)
- Distortion coefficients
- Rotation and translation vectors for each image
- Reprojection errors
- Calibration metadata

## Requirements

- Python 3.7+
- OpenCV 4.8+
- PyQt5
- NumPy
- Matplotlib

## License

This tool is provided as-is for camera calibration purposes.

