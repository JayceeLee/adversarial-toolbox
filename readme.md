## Adversarial toolbox

### WIP

Simply, this toolbox offers high level toolbox to generate and work with adversarial images on deep neural networks. 
Adversarial examples can be a pain to generate, with most implementations being in disparate frameworks, using different parameters. Here we consolidate some of that into an easy to use and setup testing setup. 


### Setting up the toolbox

This project uses the standard libraries needed for most ML projects. The main routines are built on `Keras` with
a `Tensorflow` backend. Theano implementations of the attacks are not explicitly included, but they would be easy to add. In addition, you also need the below libraries
* numpy
* scipy
* matplotlib
* pillow

Or install everything with ```python python -r requirements.txt```

This toolbox relies heavily on [Cleverhans](https://github.com/openai/cleverhans/) so the library is included at the current master version to make CI easier. 


### Train Base Keras CNN



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
