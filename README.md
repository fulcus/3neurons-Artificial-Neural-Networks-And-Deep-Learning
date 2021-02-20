# Artificial Neural Networks and Deep Learning Competition

*Artificial Neural Network and Deep Learning* is course offered by Politecnico di Milano.
In this repository you can find the Jupyter Notebooks that we created for the competition in the academic year (2020/2021).

The competition was divided into three challenges, each one of them cover a different topic of the course:
 - Image Classification 
 - Image Segmentation
 - Visual Question Answering

## 1. Image Classification 
Competition and data available [here](https://www.kaggle.com/c/artificial-neural-networks-and-deep-learning-2020)
 - Purpose: Classification of images representing people with or without mask
 - 3 different classes (no one wears a mask, everyone wears a mask, some do and some don't)
 - 5614 images in the training set
 - 450 images in the test set
 - Evaluation: Multiclass Accuracy

## 2. Image Segmentation
Competition and data available [here](https://competitions.codalab.org/competitions/27176)
 - Purpose: Crop and weed segmentation
 - 3 different classes (weed, crop, background)
 - 2 different crop types (Mais and Haricot)
 - 4 widely different datasets of pictures and masks (coming from the [ROSE challenge](http://challenge-rose.fr/en/home/))
 - Evaluation: IoU, i.e. Intersection over Union = Area of Overlap / Area of Union

## 3. Visual Question Answering
Competition and data available [here](https://www.kaggle.com/c/anndl-2020-vqa)
 - input: An image and a question about the image
 - output: Answers belong to 3 possible categories: 'yes/no', 'counting' (from 0 to 5) and 'other' (e.g. colors, location, ecc.) answers. It's treated as a classification problem (where the class is the answer)
 - 58832 questions in training set 
 - 29333 total images (size: 400x700)
 - 6372 questions for testing
 - Evaluation: Multiclass Accuracy

## Team
[__Arianna Galzerano__](https://github.com/arigalzi), [__Francesco Fulco Gonzales__](https://github.com/fulcus), [__Alberto Latino__](https://github.com/albertolatino)
