# Parking Detection Project

A deep learning-based parking spot detection system that automatically identifies and classifies parking spots as empty or occupied using a Convolutional Neural Network (CNN).

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Output](#output)

## 🎯 Overview

This project implements a complete parking detection pipeline that:
1. Allows manual annotation of parking spots using an interactive GUI
2. Trains a CNN model to classify parking spots as empty or occupied
3. Performs inference on parking lot images to detect available spots
4. Generates visual outputs and summary statistics

The system uses a custom-trained CNN model that processes grayscale images of parking spots (48x48 pixels) to determine their occupancy status.

## ✨ Features

- **Interactive Parking Slot Annotator**: GUI tool for manually drawing and annotating parking spots on images
- **Deep Learning Model**: Custom CNN architecture for parking spot classification
- **Automated Detection**: Batch processing of multiple parking spots in a single image
- **Visual Output**: Annotated images with color-coded spots (green for empty, red for occupied)
- **Summary Reports**: CSV export with parking statistics

## 📁 Project Structure

```
parking_detection_project/
│
├── data/
│   ├── matchbox_cars_parkinglot/
│   │   ├── empty/          # 908 training images of empty spots
│   │   └── occupied/       # 818 training images of occupied spots
│   ├── sample_image.png    # Test image for parking detection
│   └── archive.zip         # Archived data
│
├── models/
│   ├── emptyparkingspotdetectionmodel.h5  # Trained CNN model
│   └── yolov8n.pt          # YOLOv8 model (if used)
│
├── outputs/
│   ├── output_detected_parking.png        # Visual output with detections
│   └── parking_spot_summary.csv          # Summary statistics
│
└── src/
    └── parking_slot_annotator.ipynb       # Main Jupyter notebook
```

## 📦 Requirements

### Python Packages

- **Python 3.x**
- **OpenCV** (`cv2`) - Image processing
- **Keras/TensorFlow** - Deep learning framework
- **NumPy** - Numerical operations
- **Matplotlib** - Visualization
- **Tkinter** - GUI for annotation tool
- **Pandas** - Data handling and CSV export
- **scikit-learn** - Data splitting utilities

### Installation

Install the required packages using pip:

```bash
pip install opencv-python tensorflow keras numpy matplotlib pandas scikit-learn
```

Or create a requirements file and install:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Parking Slot Annotation

The project includes an interactive annotation tool to mark parking spots on images:

1. Open the Jupyter notebook: `src/parking_slot_annotator.ipynb`
2. Run the first cell to launch the annotation GUI
3. Click and drag on the image to draw parking spot rectangles
4. The coordinates are automatically saved
5. Use the "Clear Slots" button to reset annotations

### 2. Model Training

To train the CNN model:

1. Ensure training data is in `data/matchbox_cars_parkinglot/empty/` and `data/matchbox_cars_parkinglot/occupied/`
2. Run the training cell in the notebook
3. The model will be saved to `models/emptyparkingspotdetectionmodel.h5`

**Training Parameters:**
- Input size: 48x48 grayscale images
- Batch size: 64
- Epochs: 50
- Train/Test split: 80/20
- Validation split: 20%
- Optimizer: Adam
- Loss function: Categorical Crossentropy

### 3. Parking Detection

To detect parking spots in an image:

1. Load the trained model
2. Provide parking spot coordinates (manually annotated or pre-defined)
3. Run the detection cell
4. Results are displayed with:
   - Green rectangles: Empty spots
   - Red rectangles: Occupied spots
   - Text overlay showing total empty spots

### 4. Output Generation

The system automatically generates:
- **Visual Output**: `outputs/output_detected_parking.png` - Annotated image with detections
- **CSV Summary**: `outputs/parking_spot_summary.csv` - Statistics including:
  - Total spots
  - Empty spots count
  - Occupied spots count

## 🧠 Model Architecture

The CNN model uses the following architecture:

| Layer          | Type            | Details                     |
| -------------- | --------------- | --------------------------- |
| Input          | Image (48x48x1) | Grayscale, resized to 48x48 |
| Conv2D         | 32 filters      | 3x3 kernel, ReLU activation |
| Conv2D         | 64 filters      | 3x3 kernel, ReLU activation |
| MaxPooling2D   | 2x2 pool size   |                             |
| Dropout        | 25%             | Regularization              |
| Flatten        | —               |                             |
| Dense          | 128 neurons     | ReLU activation             |
| Dropout        | 50%             | Regularization              |
| Output (Dense) | 2 neurons       | Softmax for classification  |

### Training Workflow

1. Load grayscale images from training folders
2. Resize to 48×48 pixels
3. Label encoding: 0 = empty, 1 = occupied
4. Split into training and test sets (80/20)
5. Normalize pixel values to [0, 1]
6. Train with categorical crossentropy loss and Adam optimizer

## 📊 Results

The trained model achieves high accuracy in classifying parking spots. Example results from a test run:

- **Total Spots Detected**: 394
- **Empty Spots**: 115
- **Occupied Spots**: 279

The model demonstrates strong performance with validation accuracy typically above 95% after training.

## 🔧 Configuration

### Image Paths

Update the following paths in the notebook as needed:

- **Sample Image**: `../data/sample_image.png`
- **Training Data**: 
  - Empty spots: `../data/matchbox_cars_parkinglot/empty/`
  - Occupied spots: `../data/matchbox_cars_parkinglot/occupied/`
- **Model Path**: `../models/emptyparkingspotdetectionmodel.h5`
- **Output Directory**: `../outputs/`

### Customization

- **Image Size**: Modify the resize dimensions (currently 48x48) in both training and inference code
- **Model Architecture**: Adjust layers, filters, and dropout rates in the Sequential model
- **Training Parameters**: Modify batch size, epochs, and validation split as needed

## 📝 Notes

- The annotation tool uses Tkinter with matplotlib backend
- Ensure images are in supported formats (PNG, JPG, etc.)
- The model expects grayscale input images
- Coordinate format: (x1, y1, x2, y2) where (x1, y1) is top-left and (x2, y2) is bottom-right

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is open source and available for educational and research purposes.

---

**Note**: Make sure to have sufficient training data for optimal model performance. The current dataset includes 908 empty spot images and 818 occupied spot images.

