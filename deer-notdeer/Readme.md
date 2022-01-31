# Deer / Not Deer Project

This folder will contain my work to create a dataset and model to detect deer.  This project will classify image as having a deer or being background.

This is not object detection, and it is similar to Dog-vs-Cat or Mask-NoMask classification. However

## Data Sets

This project uses two datasets:

* Deer Image Dataset
* Background Image Dataset

### Deer Dataset

[Kaggle](https://www.kaggle.com/faisalakhtar/images-of-deer-for-svm-classifier)

### Background Image Dataset

[Kaggle](https://www.kaggle.com/arnaud58/landscape-pictures)


## Keras Data Augmentation Layers

See [Keras issue](https://github.com/keras-team/keras/issues/15699).  It looks like there are issues using the preprocessing layers and saving a model.

Use `ImageDataGenerator` for the time being.


https://keras.io/api/layers/preprocessing_layers/

https://keras.io/api/layers/preprocessing_layers/image_augmentation/

