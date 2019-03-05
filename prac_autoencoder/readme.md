# Autoencoder

The **autoencoder** neural network is commonly used for **feature selection** and **extraction**. An autoencoder neural network is an unsupervised learning algorithm that applies backpropagation, setting the target values to be equal to the inputs. i.e. it uses $y^{(i)} = x^{(i)}$

![](images/autoencoder.png)

The autoencoder tries to learn a function $h_{W,b}(x) = x$. In other words, it is trying to learn an approximation to the identity function, so as to output $\bar x$ that is similar to input $x$. The identity function seems a particularly trivial function to be trying to learn; but by placing constraints on the network, such as by *limiting the number of hidden units*, we can discover very interesting structure about the data (dimensionality reduction).

In short, the simple autoencoder often ends up learning a low-dimensional representation very similar to PCAs.


Following this [tutorial](http://kvfrans.com/variational-autoencoders-explained/), we will first start from a simple network and add parts step by step.

> A common way of describing a neural network is an approximation of some function we wish to model. Alternatively, they can also be considered as a data structure that holds information, **very interesting information**.

![](images/1.jpg)

Given the above flowchart, we have a network comprised of a few deconvolution layers. We set the input to always be a vector of ones. Then, we can train the network to reduce the *mean squared error* between itself and one target image. The "**data**" for that image is now contained within the network's parameters.

After scaling to multiple images, we use a **one-hot vector** for the input. [1, 0, 0, 0] for instance, could represent a cat image, while [0, 1, 0, 0] could represent a dog image. To let the network memorize different images, we use a vector of real numbers instead of a one-hot vector.

Choosing the latent variables randomly is obviously a bad idea. In an autoencoder, we add in another component that takes the original images and encodes them into vectors for us. The deconvolutional layers then "decode" the vectors back to the original images.

![](images/2.jpg)

We have finally reached a stage where our model has some hint of a practical use. We can train our network on as many as images we want. If we save the encoded vector of an image, we can reconstruct it later by passing it into the decoder portion. What we have here is the **standard/basic** *autoencoder*.


## Deep Autoencoder

The extension of the basic autoencoder is the **deep autoencoder**, which have more hidden layers.

![](images/deepAE.png)

The additional hidden layers enable the autoencoder to learn mathematically more complex underlying patterns in the data. The first layer of the deep autoencoder may learn first-order features in the raw input (such as edges in an image). The second layer may learn second-order feature corresponding to patterns in the appearance of the first-order features (e.g. in terms of what edges tend to occur together, to form contour ot corner detectors). Deeper layers of the deep autoencoder tends to learn even higher-order features.


## Variational Autoencoder

A **variational autoencoder** (**VAE**) resembles a classical autoencoder and is a neural network consisting of an encoder, a decoder, and a loss function.

The problem of the standard autoencoder is its incapability in **generalisation**. We can't generate anything yet, since we don't know how to create latent vectors other than encoding them from images.

There is [a simple solution](https://arxiv.org/pdf/1312.6114.pdf). We add a constraint on the encoding network, forcing it to generate latent vectors, which roughly follow a *unit Gaussian distribution*. It is the constraint that separates a variational autoencoder from a standard one.

In practice, there is a tradeoff between how accurate our network can be and how close its latent variables can match the unit gaussian distribution.

This constraint forces the encoder to be very efficient, creating information - rich latent variables. This improves generalization, so latent variables that we either randomly generated, or we got from encoding non-training images, will produce a nicer result when decoded.

### Results

![](images/prediction.png)

![](images/learned_distribution.png)


## Denoising Autoencoder

When there are **more** nodes in the hidden layer than the inputs, the autoencoder network is risking to learn the so-called "Identity Function", meaning that the output equals the input, marking the autoencoder *useless*.

Denoising autoencoders solve this problem by **corrupting the data on purpose by randomly turning some of the input values to zero**. In general, the percentage of input nodes which are being set to zero is about *50%*. Other sources suggest a lower count, such as 30%. It depends on the amount of data and input nodes you have.

![](images/DAE.png)

When calculating the *Loss* function, it is important to compare the output values with the **original input**, not with the corrupted input. The risk of learning the identity function instead of extracting features is therefore eliminated.



## Reference

1. https://github.com/llSourcell/autoencoder_explained/blob/master/variational_autoencoder.py
2. http://kvfrans.com/variational-autoencoders-explained/
3. https://towardsdatascience.com/denoising-autoencoders-explained-dbb82467fc2
4. 