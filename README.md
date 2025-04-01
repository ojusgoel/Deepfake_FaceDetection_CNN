# Deepfake Face Detection

## Overview
This project aims to detect deepfake images using a Convolutional Neural Network (CNN) trained on real and fake face datasets. The model is implemented using PyTorch and trained on the `140k-real-and-fake-faces` dataset from Kaggle.

## Dataset
The dataset used for training and validation is sourced from Kaggle and contains a balanced set of real and fake face images.

## Model Architecture
The CNN model consists of multiple convolutional layers with batch normalization, max pooling, dropout layers for regularization, and fully connected layers. The final output layer uses a sigmoid activation function for binary classification (real or fake).

### Key Features:
- 10 convolutional layers with increasing filter sizes
- Batch normalization for stable training
- Dropout for regularization
- Fully connected layers for classification

## Data Augmentation
To improve the model's robustness and generalization capability, the following augmentations were applied during training:
- **Resizing** images to 224x224 pixels to ensure consistency.
- **Rotation** up to 20 degrees to account for different angles.
- **Horizontal flipping** with a 50% probability to increase diversity.
- **Brightness and contrast adjustments** to handle varying lighting conditions.
- **Gaussian blur** with a low probability to simulate real-world distortions.

## Installation
### Prerequisites:
- Python 3.x
- PyTorch
- torchvision
- Albumentations
- NumPy
- Matplotlib
- Kaggle API (for dataset download)
- scikit-learn

### Setup:
1. Clone the repository:
```
git clone https://github.com/ojusgoel/Deepfake_FaceDetection_CNN.git
cd Deepfake_FaceDetection_CNN
```
2. Download the dataset from Kaggle and extract it:
```
kaggle datasets download -d xhlulu/140k-real-and-fake-faces
```

## Training the Model
The model was trained for **6 epochs**, and the final performance on the validation set was:
- **Training Accuracy:** 94.06%
- **Validation Accuracy:** 96.93%
- **Training Loss:** 0.1536
- **Validation Loss:** 0.0838

The model was optimized using **cross-entropy loss** and the Adam optimizer, achieving stable convergence over the training period.

## Evaluation
After training, the model was evaluated on a separate test dataset, yielding the following results:
- **Test Loss:** 0.1102
- **Test Accuracy:** 95.77%
- **Precision:** 94.74%
- **Recall:** 97.20%
- **F1-Score:** 95.95%

These metrics indicate that the model performs exceptionally well in distinguishing between real and fake images. The **high recall (97.20%)** suggests that the model is highly sensitive to detecting fake images, which is crucial for deepfake detection. The **precision (94.74%)** ensures that false positives are minimized, and the **F1-score (95.95%)** confirms a strong balance between precision and recall.

## License
This project is licensed under the MIT License.

