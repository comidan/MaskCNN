# MaskCNN
This model is meant to solve the classification problem of determining whether a group of people is wearing a mask, partially or not at all given an image as input.
 
No one is wearing a mask             |  All the people are wearing a mask  | Some people are wearing a mask
:-------------------------:|:-------------------------:|:-------------------------:
<img src="/MaskDataset/test/10296.jpg" width="213" height="137">  |  <img src="/MaskDataset/test/10148.jpg" width="213" height="137"> | <img src="/MaskDataset/test/10100.jpg" width="213" height="137">

All details are present in the report file in this repo, here there are just the main details of the Neural Network used.

## Dataset 
The dataset available is constituted of roughly 5000 images and its classes are balanced. The dataset was then
augmented by means of ImageDataGenerator class where we choose augmentation values (see Table 1) empirically
based on the result of the model on the validation set.
While the RGB images were of an original dimension of about 640 x 412, we choose to handle 256 x 256 RGB
images, so we can use a smaller model which we can actually train on our limited hardware, in batches of 16 (this
is the biggest value that fit in our available GPU).

## Model Choice
The first architecture used was a Convolutionary Neural Network of 3 hidden layers using a 3x3 filter size which
had subpar performance even with the usage of some regularization technique and optimization technique such as
Dropout, after having reached the best result we think we could with this model we moved to another approach.<br>
We approached Transfer Learning using fine-tuning, we choose 3 Model, VGG19, ResNet50V2 and InceptionRes-
NetV2. These models were chosen for the following reasons: VGG19 is the one we used during the exercitations,
ResNet50V2 and InceptionResNetV2 have both residual connections and Batch Normalizzation which should allow
to train faster deeper models and in the past we used them in other contexts and they yielded good performance.<br><br>
The best architecture, of the ones we tried, is InceptionResNetV2 model with 164 layers and its Inception modules.
As mentioned before, we used fine-tuning so we trained the whole network starting from the pre-learnt features of
InceptionResNetV2 from the training on the ImageNet dataset. Then we added at the top a Flattening layer and
some dense layers that perform the classification based on the features extracted form the CNN model.<br><br>
This top model is composed by 3 Dense layers of 100, 10, and 3 neurons respectively,and ReLU as activation
function. We chose to apply Batch-Normalization to these layers to improve the regularization, particularly the
robustness to co-variate shift. Moreover, on top of these layers we added Dropout with 0.5 and 0.2 rates to further
regularize the model and prevent overfitting. These values showed to perform better.

## Initialization
For the weight initialization with Xavier Initialization we used GlorotNormal to better initialize weights W and
letting backpropagation algorithm start in advantage position, given that final result of gradient descent is affected
by weights initialization.<br>
![equation](https://latex.codecogs.com/gif.latex?W%20%5Csim%20%5Cmathcal%7BN%7D%5Cleft%28%5Cmu%3D0%2C%5C%2C%20%5C%3B%5Csigma%5E%7B2%7D%3D%5Cfrac%7B2%7D%7BN_%7Bin%7D%20&plus;%20N_%7Bout%7D%7D%5Cright%29)

## Results
With the first architecture we were able to reach on Kaggle an accuracy on the test dataset of 84%, while with
the second architecture we've reached an accuracy on the test set of 92.66% using validation and losing between
validation and test accuracy only 0.31%, so it's a quite good model.<br>
![validation](/results/validation.png)
<br>
Then re-training on the whole dataset using the same model we've reached an accuaracy on the test set of 94.44%.<br>
![full](/results/full.png)
