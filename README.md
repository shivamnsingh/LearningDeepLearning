# Learning Deep Learning

Notes and experiments from learning deep learning with TensorFlow/Keras. The repository currently centers on one project: classifying cat and dog images with convolutional neural networks and comparing several ways to improve them.

## Project: Cat vs. Dog classification

Notebook: [`CatVsDog/CatVsDog_Classification.ipynb`](CatVsDog/CatVsDog_Classification.ipynb)
(A PDF export of the Colab run is included alongside it.)

**Setup:** images are loaded from a Kaggle dataset through `kagglehub`, resized to 150x150, and fed to the models in batches of 32 using Keras `ImageDataGenerator`. The task is binary classification (cat vs. dog) trained with the Adam optimizer and binary cross-entropy loss.

### Experiments and findings

| Model | What it is | Result |
|-------|-----------|--------|
| Baseline CNN | 3 convolutional layers | ~79.6% test accuracy, clear overfitting (training accuracy ~95%, test loss rises after epoch 5) |
| Augmented CNN | Same architecture plus rotation, shift, zoom and flip augmentation and early stopping | ~88.2% test accuracy, test loss cut from 0.6477 to 0.2900, balanced precision and recall |
| VGG-style network from scratch | 6 convolutional layers, trained from random weights | Failed to converge: stuck at 50% accuracy (loss 0.6931), predicting a single class |
| VGG16 transfer learning | ImageNet-pretrained VGG16 (frozen) with a small new classifier | Included in the notebook; see the notebook for the run |

**Takeaways**

- Data augmentation and early stopping fixed most of the baseline's overfitting on a small training set (~8,000 images).
- Deep plain networks without normalization can fail to train from scratch on small datasets.
- Pretrained features are a natural fit here because ImageNet already contains many cats and dogs.

## Repository layout

```
CatVsDog/
  CatVsDog_Classification.ipynb
  CatVsDog_Classification.ipynb - Colab.pdf
d/
  code.ipynb            # empty scratch notebook
```

## Getting started

```bash
git clone https://github.com/shivamnsingh/LearningDeepLearning.git
cd LearningDeepLearning
pip install jupyter tensorflow numpy pandas matplotlib pillow scikit-learn kagglehub
jupyter notebook CatVsDog/CatVsDog_Classification.ipynb
```

Training CNNs is much faster on a GPU. The notebook was run on Google Colab, and the dataset download through `kagglehub` may require Kaggle credentials.
