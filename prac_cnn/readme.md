# Neural Networks coding practical
The purpose of this exercise is to practice writing and deploying neural network architectures on established benchmark datasets. You will have the chance to experiment with various types of network layers, regularisation and optimisation techniques -- all widely used nowadays in deep learning projects across industry and research.

You will be working with the [Keras](https://keras.io/) framework, which allows you to rapidly prototype a network architecture without worrying about boilerplate code, and evaluating your model on Fashion-MNIST -- a slightly more exciting and difficult 1:1 correspondence of the MNIST dataset :-).

## classification results
| activation | dropout | batch_norm | epoch | accuracy |
| --- | --- | --- | --- | --- |
| sigmoid | NA | NA | 30 | 84.74% |
| RELU | NA | NA | 30 | 86.75% |
| RELU | Y | NA | 30 | 87.68% |
| RELU | Y | NA | 40 | 84.73% |
| RELU | Y | Y | 40 | 83.46% |

## activation functions

Activation functions calculate a "weighted sum" of its input, add a bias and then decide whether or not it should be fired. It is just a node that is added to the output end of NNs. **Non-linear mapping** is another key aspect of activation functions.

Three popular types:
1. sigmoid

$f(x) = \frac{1}{1+e^{-\beta x}}$ where $\beta$ is usually zero [logistic function]

problems: vanishing / exploding gradients & output not zero-centered

2. tanh (Hyperbolic Tangent)

$f(x) = \frac{2}{1+e^{-2x}} -1$ 

problems: vanishing / exploding gradients

3. ReLu (Rectified Linear Unit)

$R(z) = max(0, z)$

problems: resulting in dead neurons
BUT, it helps NNs to learn fast and avoid vanishing gradients

addtional info:

| name      | equation                          | derivative                | notes     |
| --        | --                                | --                        | --        |
| sigmoid   | $f(x)=\frac{1}{1+e^{-\beta x}}$   | $f^{'}(x)=f(x)(1-f(x))$   | function is *monotonic* but derivative is not            |
| tanh      | $f(x)=\frac{2}{1+e^{-2x}}-1$      | $f^{'}(x)=1-f^{2}(x)$     | function is *monotonic* but derivative is not            |
| ReLu      | $R(z)=max(0, z)$                  | $f^{'}(x)=0/1$            | both function and derivative are *monotonic*      |