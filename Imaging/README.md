# Image Processing Pipeline for Endothelial Cell Analysis

This directory contains the Jupyter notebooks and related files for the processing and analysis of microscopy images of endothelial cells. The entire workflow is designed and optimized for the Google Colab environment to leverage its GPU capabilities, which are crucial for computationally intensive tasks like image denoising, projection, and segmentation.

## Workflow Overview

The image processing pipeline is divided into three main stages, each housed in its own respective folder. The typical workflow involves executing the notebooks in the following order:

### 1. Projection (`/Projection`)

**Purpose:** To convert 3D microscopy image stacks (Z-stacks) into 2D images.

This is the initial step where the raw 3D image data is processed. A maximum intensity projection is typically used to collapse the Z-stack into a single, focused 2D image. This reduces the complexity of the data and prepares it for the subsequent segmentation step. The notebooks in this folder are named according to the experimental conditions of the images they process (e.g., `Static-x20.ipynb`, `1.4Pa-x40.ipynb`).

### 2. Segmentation (`/Segmentation`)

**Purpose:** To identify and outline individual cells or other regions of interest.

This is the most computationally demanding part of the pipeline. The 2D projected images are fed into segmentation algorithms (often based on machine learning models like U-Net) to identify the boundaries of each cell. The effective execution of these notebooks relies heavily on GPU acceleration, which is why Google Colab was the chosen environment.

### 3. Analysis (`/Analysis`)

**Purpose:** To extract quantitative data from the segmented images.

Once the cells are segmented, the notebooks in this folder are used to perform quantitative analysis. This includes, but is not limited to:
- Calculating cell density (`Cell_density.ipynb`)
- Measuring the area of gaps (holes) in the cell monolayer (`Holes.ipynb`)
- Comparing morphological features across different experimental conditions (`Comparisons.ipynb`)

## Environment Note

**Google Colab:** All notebooks (`.ipynb`) were developed and are intended to be run in Google Colab. They often contain code to mount Google Drive for data access, with file paths structured accordingly (e.g., `/content/drive/MyDrive/...`).

To run these notebooks, it is highly recommended to use a GPU-enabled runtime in Google Colab. If you intend to run them in a local environment, you will need to:
1.  Install all necessary dependencies (e.g., TensorFlow, PyTorch, scikit-image, etc.).
2.  Ensure you have a compatible GPU and the necessary drivers (e.g., CUDA).
3.  Modify the file paths to match your local data storage structure.
