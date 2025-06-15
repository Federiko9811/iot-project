# IoT Posture Detection Project

## Project Overview
This project focuses on posture detection using IoT technology and machine learning. It includes a posture classifier and implements federated learning approaches for privacy-preserving model training.

## Project Structure

### Federated Learning (`/federated`)
Implementation of federated learning for distributed model training:
- `notebook.ipynb` - Jupyter notebook with implementation and documentation

### Datasets (`/datasets`)
Contains training data and sample images for the posture detection system:
- `train.csv` - Training dataset
- `good_posture_image.jpg` - Sample image of good posture
- `bad_posture_image.jpg` - Sample image of bad posture

### Posture Classifier (`/posture_classifier`)
The code used to create our dataset transforming the images in the csv file using the MediaPipe Landmarks extraction:
- `main.py` - Entry point for the posture detection system
- `posture_analyzer.py` - Core functionality for analyzing posture
- `settings.py` - Configuration settings for the application
- `/images` - Directory containing sample images for analysis or testing

