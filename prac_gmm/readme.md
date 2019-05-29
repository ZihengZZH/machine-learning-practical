# Gaussian Mixture Model

### What is a Gaussian?
A Distribution is a listing of outcomes of an experiment and the probability associated with each outcome

### What is a Gaussian Mixture Model?
It's a probability distribution that consists of multiple probability distributions.

### Problem
Given a set of data X drawn from an unknown distribution (probably a GMM), estimate the parameters $\theta$ of the GMM model that fits the data

### Solution
Maximize the likelihood $p(X|\theta)$ of the data with regard to the model parameters

$\theta* = \arg \max_{\theta}p(X|\theta) = \arg \max_{\theta}\prod_{i=1}^N p(x_i|\theta)$