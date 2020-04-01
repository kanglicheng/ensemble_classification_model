# Ensemble Model to Do Out-of-Distribution Detection
Apply ensemble classification model to detect out-of-distribution samples.  
* [slides](https://github.com/liyin2015/ensemble_classification_model/blob/master/Uncertainty%20Estimation%20using%20Ensemble%20Model%20(1).pptx)
* [Source code](https://github.com/liyin2015/ensemble_classification_model/blob/master/Ensemble_model_classification-final.ipynb)

## Purpose
Follow [Simple and scalable predictive uncertainty estimation using deep ensembles](https://arxiv.org/abs/1612.01474), implemented the ensemble model on classification task with **PyTorch**. Experiments show the effect of out-of-distribution detection under different loss functions and optimizers. 

## Datasets
Three datasets are used to demonstration purpose, including
* [MNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#mnist)

![sample mnist images and labels](https://github.com/liyin2015/ensemble_classification_model/blob/master/figures/mnist.png)


* [FASHION-MNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#fashion-mnist)

![sample fashion-mnist images and labels](https://github.com/liyin2015/ensemble_classification_model/blob/master/figures/fashion_mnist.png)


* [NOT-MNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html)

![sample not-mnist images and labels](https://github.com/liyin2015/ensemble_classification_model/blob/master/figures/not_mnist.png)


## Optimizers
Two of the most conventional and popular optimizers are tried in the experiments: 
* Adam Optimizer
* SGD Optimizer

## Loss Functions
Two loss functions are explored:
* Brier Score
* Softmax Cross Entropy

Details can be found in the slide.

## Experimental Results
An ensemble model consists of 5 single NNs is trained on MNIST training dataset, and then tested on MNIST test dataset, FASHION-MNIST and NOT-MNIST to demonstrate the out-of-distribution detection effect. The model is trained a total of 20 epochs. The following figures show two metrics-- testing accuray and averaged probability score of predicted labels. Both the ensemble net and single net are evaluated with these two metrics.  
1. Brier Score and Adam Optimizer

![Brier Score and Adam Optimizer](https://github.com/liyin2015/ensemble_classification_model/blob/master/figures/brier_score_adam.png)

2. Brier Score and SGD Optimizer

![Brier Score and SGD Optimizer](https://github.com/liyin2015/ensemble_classification_model/blob/master/figures/brier_score_sgd.png)

3. Softmax Cross Entropy and Adam Optimizer

![Softmax Cross Entropy and Adam Optimizer](https://github.com/liyin2015/ensemble_classification_model/blob/master/figures/softmax_cross_entropy_adam.png)


## Takesways
1. Ensemble model is able to gain better accuracy, delay the overfitting, and shows higher uncertainty when it comes to out-of-distribution samples.
2. SGD optimizer is better than Adam optimizer to detect out-of-distribution examples.
3. To make a single model more resilient to out-of-distribution examples, avoid overfitting the model.


## Help Needed
Fork to add more cases such as regression task, adversarial learning, and/or more related experiments.

## References
[1] Lakshminarayanan, Balaji, Alexander Pritzel, and Charles Blundell. "Simple and scalable predictive uncertainty estimation using deep ensembles." Advances in neural information processing systems. 2017.
