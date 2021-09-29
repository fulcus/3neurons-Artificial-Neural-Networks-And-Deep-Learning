# Artificial Neural Networks and Deep Learning Competition

*Artificial Neural Network and Deep Learning* is course offered by Politecnico di Milano.
In this repository you can find the Jupyter Notebooks that we created for the competition in the academic year (2020/2021).

The competition was divided into three challenges, each one of them cover a different topic of the course:
 - Image Classification 
 - Image Segmentation
 - Visual Question Answering

## 1. Image Classification 
Competition and data available on [kaggle](https://www.kaggle.com/c/artificial-neural-networks-and-deep-learning-2020)

The goal of the challenge is to classify images of people wearing masks into one of three classes:
* Everyone in the image are wearing a mask 
* No one in the image is wearing a mask
* Someone in the image is not wearing a mask.

**Dataset**: 5614 images in the training set, 450 images in the test set

**Evaluation**: Multiclass Accuracy 95.33%

<img src="assets/1-challenge.jpg" width="470"/>

### The models

#### Simple CNN
The first approach we tried was a simple Convolutional Neural Network with 5 convolutional layers, each followed by a ReLU activation layer and a 2x2 MaxPooling. We also tried changing the structure of the model several times, adding and removing layers, and some regularization techniques such as Dropout and changing the activation function to a Leaky ReLU and noticed some minor improvements on the performance but still far from the desired outcome, as we only got a 54% training accuracy. Hence, we decided to switch to transfer learning.
#### Transfer Learning
By leveraging previously trained models and including them in the structure of our new model we were able to obtain a noticeably higher accuracy.
The first model we used was **VGG16**, which, after several tries and substantial tweaking surprised us with the astounding accuracy of 87%.
We also tried **VGG19** and made some changes to the structure of the layers: by thorough experimentation and inspired by famous architectures, we decided to add a Global Average Pooling layer right after the convolutional part of the network, substituting the previous MLP network. This reduced the number of parameters to be trained and obtained equally good results as we already got enough nonlinearity in the convolutional part. With this approach we reached an accuracy of 90%.
To further improve the performance we tried some newer models, **Inception** and **Xception**.
We then proceeded to tune the hyperparameters and structure of the model. Following are some of the parameters that mostly influenced it:
* the **learning rate** was changed from 1e-5 to 5e-5 and eventually lowered to 2e-5 to reduce
oscillations
* the **number of layers** of the fully connected network, in particular the addition of a **Dropout** layer seemed to slightly increase the model’s performance
* **image resizing** of height and width: we found that increasing from 256 to 512 boosted the accuracy

## 2. Image Segmentation
Competition and data available on [codalab](https://competitions.codalab.org/competitions/27176)
The goal of the challenge is to perform precise automatic crop and weed segmentation for the agricoltural sector.

The segmented objects can belong to one of three classes: weed, crop or background.
The images contained two different crop types: Mais or Haricot.

**Datasets**: integration of 4 widely different datasets of pictures and masks coming from the [ROSE challenge](http://challenge-rose.fr/en/home/)

**Evaluation**: Intersection over Union 62.34%

![equation](https://latex.codecogs.com/gif.latex?IoU=\frac{\text{Area&space;of&space;Overlap}}{\text{Area&space;of&space;Union}})


| <img src="assets/2-challenge-img.jpg" width="410"/> | <img src="assets/2-challenge-mask.png" width="410"/> |
|:---:|:---:| 
| Input image | Taget mask |
 

### Image processing

A major improvement came from changing image processing library from PIL to **CV2**. As a matter of fact, it seemed that due to the dimension of the images of the dataset, too much information was lost when processing the images with the former.
Since images had very large dimensions, using a resize function was another of the reasons of the significant loss of information. Through trial and error, we found that `2048x2048` was the best image size for the inputs, with the application of `CV2.INTER_AREA` as an interpolation method for downscaling and `CV2.BICUBIC` for upscaling. Another boost in performance came by applying the same random transformation to images and mask targets using custom dataset objects.

### The models

#### U-Net

The first approach we tried was a traditional U-Net which initially consisted of a stack of **Convolutional and Max Pooling** layers for the **encoder** part and **Upsampling and Convolutional** layers for the **decoder**. To get better precise locations, at every step of the decoder we used **skip connections** by concatenating the output of the upsampling layers with the feature maps from the encoder at the same level (u6=u6+c4; u7=u7+c3; u8 =u8+c2; u9=u9+c1) and after every concatenation we again applied two consecutive regular convolutions so that the model can learn to assemble a more precise output. We then tried changing the structure of the model several times, by adding batch normalization, and changing layers (for instance from Upsampling to Transpose Convolution in the decoder), and some regularization techniques such as **Dropout** and using the **Leaky ReLU** as activation function and noticed some improvements on the performance, reaching a meanIoU of 0.22 on the test dataset. We then tried a model using transfer learning.

#### Transfer Learning on Encoder

By leveraging previously trained models and including them in the structure of our new model we were able to obtain noticeably better results. In particular, as far as the encoder is concerned we tried transfer learning both from **VGG16** and **ResNet152**, and eventually chose the latter as it performed consistently better.
We then proceeded to tune the hyperparameters and the structure of the model. Following are some of the parameters that influenced it:
* the **learning rate** was changed from 1e-5 to 5e-5 and eventually to 1e-4 to reduce oscillations
* **Data Augmentation** parameters: a parameter which seemed to influence the accuracy was the preprocessing function designed for ResNet
* adding a **Learning Rate Plateau**, that allowed us to dynamically change the model’s learning rate whenever the val_loss parameter didn’t change.


## 3. Visual Question Answering
Competition and data available on [kaggle](https://www.kaggle.com/c/anndl-2020-vqa)

The goal of the challenge is to answer questions using the information provided by the corresponding image and question pair. 
The given input is an image and an associated question about it, and the output is an answer, belonging to one of three possible categories: 'yes/no', 'counting' (from 0 to 5) and 'other' (e.g. colors, location, ecc.) answers. It's treated as a classification problem (where the class is the answer).

**Dataset**: 58832 questions in training set, 29333 total images (size: 400x700), 6372 questions for testing

**Evaluation**: Multiclass Accuracy 64.76%

| <img src="assets/3-challenge.png" width="470"/> |
|:--| 
| **Q**: How many bikes?! <br> **A**: 1 |



### Preprocessing 
In the first section of the code we worked on collecting and reorganizing images and questions of the training dataset. Tokenization and padding of the questions was applied using tf.keras.preprocessing tools and the custom generator (`batch_generator`), which allows a correct generation of batches for the network, was developed. A further improvement in the performance could have been reached by incrementing the batch size to more than 70 (which was our final choice) which wasn’t possible to test due to exhaustion of GPU memory on Google Colab.

### The model

The structure of the model is inspired by [A. Agrawal, J. Lu, S. Antol, M. Mitchell, C. L. Zitnick, D. Batra, D. Parikh. VQA: Visual Question Answering.][1], which aims to solve the problem that we were given with an original architecture, which consists of two parts: one for the processing of the images and one for the processing of the texts, joined together with a `concatenate` layer.

#### Image processing model
Initially, we chose the **VGG16** model for transfer learning, and successively switched to **VGG19** which improved the performance of our model. All pre-trained layers were set as not trainable. We then tried to set the first layers as non-trainable, and to keep the others trainable. The idea behind this attempt was that the first layers of VGG would be already trained to capture the most generic features of the dataset’s images while the bottom layers had to be fine tuned to learn the specific features of our training dataset. However, this attempt hurt the performance, so we went back to the original settings.

#### Text processing model
In order to embed the questions, the model uses a **two layer LSTM** to encode the questions.
Initially we used LSTM layers and Dropout layers. In a second moment, we perceived that this part was less powerful than the Image processing part of the model. Therefore, for building a RNN with a longer range dependencies, we made the first LSTM layer **bidirectional**. This layer permits the propagation of input in the RNN.

#### Final model and fine tuning
In the final part of the implementation of the model, the image features are normalized. Then, both the question and image features are transformed to a common space and fused via element-wise multiplication, which is then passed through a **fully connected layer** followed by a **softmax** layer to obtain a distribution over answers.
We then proceeded to tune the hyperparameters and the structure of the model. Following are some of the parameters that influenced it:
* The compilation was pretty standard, using Adam optimizer and the Categorical_Crossentropy loss function (since we used one-hot representation for the target).
* The **learning rate** was changed from 1e-4 to 5e-4 and eventually to 1e-3 to reduce oscillations
* Adjustment in the percentage used in the **Dropout** layers
* **Leaky ReLu** as activation function in dense layers instead of standard ReLu
* addition of a **Learning Rate Plateau**, that allowed us to dynamically change the model’s learning rate whenever the val_loss parameter didn’t change.

[1]: https://arxiv.org/pdf/1505.00468v6.pdf

## Team
[__Arianna Galzerano__](https://github.com/arigalzi), [__Francesco Fulco Gonzales__](https://github.com/fulcus), [__Alberto Latino__](https://github.com/albertolatino)
