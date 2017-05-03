## Adversarial toolbox

Toolkit for working with adversarial images using Keras and Cleverhans

### WIP

### Generate images

offers multiple ways to generate adversarial examples for standard neural network architectures

Generate images with common techniques

* Fast Gradient Sign Method - FSGM

* Jacobian Based Saliency Map Approach - JSMA

* DeepFool --> coming soon

* Universal Adversarial Images --> coming soon


### Architectures supported

* Vanilla convnet

* VGG_16 with batchnorm

* Resnet (all)

* MLP --> coming soon

* Logistic Classifier --> coming soon


### Features

Simply, this toolbox offers am easy way to generate adversarial images on deep neural networks. 
Adversarial images can be a pain to generate, with most implementations being in disparate frameworks, using different parameters. 
Here we consolidate some of that into an easy to use and setup testing setup. 


Using one of the included models, or one that you've defined, its easy to train a classifier in Keras using the included trainer. 
Currently only training on Cifar10 (mnist coming soon) is supported. 

With a trained .h5 model, it can be used to generate the adversarial images via any of the above supported methods
