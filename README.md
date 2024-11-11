# Forensic Image Tampering Detection Tool

This project is a Flask-based web application for forensic image tampering detection. It allows users to upload images and analyze them using various techniques, including metadata extraction, Error Level Analysis (ELA), statistical analysis (RGB histograms, mean/median color values), and Discrete Cosine Transform (DCT) clone detection. This tool provides visualization of these analyses to detect potential tampering in images.

## Features

- **Metadata Extraction**: Extracts basic metadata information from the uploaded image.
- **Error Level Analysis (ELA)**: Identifies regions with varying compression levels to spot possible edits.
- **Statistical Analysis**: Shows RGB histograms, mean/median color values, and color standard deviations.
- **DCT-Based Forgery Detection**: Detects cloned regions within the image using Discrete Cosine Transform analysis.

## Directory Structure

- **app.py**: The main entry point to run the Flask server.
- **uploads/**: Directory where the uploaded images are stored.
- **static/uploads/**: Directory where the generated output images, such as ELA images, histograms, and heatmaps, are stored and served from. These are created based on user uploads and analysis.
- **templates/**: Contains HTML files used for rendering the web interface for the app.

## Usage

### Prerequisites

Ensure you have the following packages installed:
```bash
pip install flask opencv-python-headless matplotlib numpy Pillow scipy tensorflow
```

## Techniques Used

- **Metadata Extraction**: Extracts embedded metadata (EXIF) information, including camera model, timestamps, and geolocation. This can be useful to validate image authenticity.

- **Error Level Analysis (ELA)**: Highlights areas of an image that have different error levels, indicating possible alterations. ELA works by compressing the image and comparing the compressed version to the original.

- **Statistical Analysis**: Calculates and displays RGB histograms, mean, median, and standard deviation of color values, which helps in identifying unnatural color patterns indicative of tampering.

- **DCT-Based Forgery Detection**: Applies Discrete Cosine Transform to detect cloned or copied regions within an image. This method can detect block-based anomalies that may arise from copy-paste actions or other edits.

## Script Explanations

- **app.py**: The main script that initializes the Flask web application, handles routing, image uploads, and directs analysis functions to process uploaded images.

- **metadata_analysis.py**: Extracts metadata (EXIF data) from uploaded images, displaying attributes such as date, camera information, and GPS coordinates. This data can be useful in verifying authenticity.

- **ela_analysis.py**: Performs Error Level Analysis (ELA) by compressing the uploaded image and calculating differences to identify areas with abnormal compression levels. Such areas may indicate alterations or edits.

- **statistical_analysis.py**: Computes RGB histograms, color means, medians, and standard deviations to highlight color distribution anomalies, potentially signaling image manipulation.

- **dct_analysis.py**: Analyzes the image using Discrete Cosine Transform (DCT) to detect copied and pasted areas. It divides the image into blocks, calculates DCT coefficients, and identifies suspicious regions where DCT values deviate.

- **deep_fake_model.py**: Defines a Convolutional Neural Network (CNN) model using a pre-trained ResNet50 architecture. This model is designed to classify images as real or manipulated, training on extracted features and classifying them into fake or real.
