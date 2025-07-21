# Deep-Learning-Cat-Dog-Classifier

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Project Overview

This project develops a robust **Deep Learning-based Cat and Dog Image Classifier** utilizing Convolutional Neural Networks (CNNs) and transfer learning. By accurately distinguishing between images of cats and dogs, this system showcases the power of deep learning for computer vision tasks and handles a substantial image dataset, aiming for high classification accuracy.

## ✨ Key Features & Technologies

- **Deep Learning Model:** Implements a state-of-the-art deep learning model using **TensorFlow/Keras**, likely leveraging pre-trained models (e.g., from TensorFlow Hub) for efficient and highly accurate image classification.
- **Comprehensive Data Handling:** Efficiently processes and prepares a large-scale image dataset of **25,000 images** of cats and dogs, including techniques for data loading, preprocessing, and augmentation to ensure model robustness and generalization.
- **Image Preprocessing & Augmentation:** Incorporates strategies for resizing, normalizing, and augmenting image data to optimize it for CNN training and prevent overfitting.
- **Robust Evaluation Metrics:** Employs standard machine learning metrics to thoroughly assess model performance, including accuracy, precision, recall, and a confusion matrix.
- **Predictive Functionality:** Provides the capability to classify new, unseen images as either 'cat' or 'dog'.
- **Libraries Used:**
  - `tensorflow`
  - `keras`
  - `numpy`
  - `Pillow` (PIL)
  - `matplotlib`
  - `scikit-learn` (for data splitting)
  - `opencv-python` (cv2)

## ⚙️ How It Works

The system operates through several key phases:

1. **Data Extraction & Preparation:** Extracts the `dogs-vs-cats.zip` dataset. Images are then loaded, resized to a consistent dimension, and normalized.
2. **Data Splitting:** Divides the dataset into training and validation/testing sets for proper model evaluation.
3. **Model Architecture & Transfer Learning:** Defines a CNN architecture, typically using a pre-trained model (from TensorFlow Hub), followed by custom classification layers.
4. **Model Training:** Trains the CNN model on the dataset, learning to identify patterns distinguishing cats from dogs.
5. **Model Evaluation:** Evaluates performance on the test set using metrics like accuracy, precision, recall, and a confusion matrix.
6. **Prediction:** Classifies new, unseen images into 'cat' or 'dog'.

## 📊 Dataset

Uses the popular **"Dogs vs. Cats"** dataset (`dogs-vs-cats.zip`) containing 25,000 images (12,500 cats + 12,500 dogs), ideal for training and evaluating image classification models.

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook (recommended)
- Install dependencies:

```bash
pip install tensorflow numpy Pillow matplotlib scikit-learn opencv-python jupyter
````

> If using TensorFlow Hub, also install:

```bash
pip install tensorflow_hub
```

### Installation

1. Clone the repository:

```bash
[git clone https://github.com/YOUR_GITHUB_USERNAME/Deep-Learning-Cat-Dog-Classifier.git
cd Deep-Learning-Cat-Dog-Classifier](https://github.com/Bishal-Nengminja/Deep-Learning-Cat-Dog-Classifier)
```

2. Download the `dogs-vs-cats.zip` dataset if not already included (due to size limits). Place it in the project root.

3. Launch the notebook:

```bash
jupyter notebook Cat_and_Dog_Image_Classifier.ipynb
```

4. Run all notebook cells to execute the full pipeline.

## 📈 Results and Performance

*(You can update this section after training your model)*
Example:

* Validation Accuracy: **92.4%**
* Test Accuracy: **91.7%**
* Precision, Recall, F1-Score, and Confusion Matrix shown in the notebook.

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repository, open issues, or submit pull requests.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 📞 Contact

**Your Name:** \[Bishal Nengminja]
**GitHub:** https://github.com/Bishal-Nengminja
**Email:** bishalnengminja61@gmail.com
