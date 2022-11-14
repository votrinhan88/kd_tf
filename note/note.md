# Deep Learning
## 1. Layers
### Batch Normalization:
- [Paper](https://arxiv.org/abs/1502.03167)
- Has different training and validation behavior
- No need to use bias term in layers before/after BatchNorm (with no activation in-between).
    <details><summary>Explanation (Quoting original paper)</summary>

    Note that, since we normalize $Wu+b$, the bias $b$ can be ignored since its effect will be canceled by the subsequent mean subtraction (the role of the bias is subsumed by $β$ in Alg. 1). Thus, $z = g(Wu + b)$ is replaced with $z = g(BN(Wu))$.
    [StackOverflow thread](https://ai.stackexchange.com/questions/27716/when-should-you-not-use-the-bias-in-a-layer)
    </details>
### Dropout:
- [Paper](https://jmlr.org/papers/v15/srivastava14a.html)
- Has different training and validation behavior
- Effect of using Dropout right after a convolutional layer is debatable (traditionally used after fully-connected layers).
### Pooling
- Max Pooling vs. Average Pooling:
    + ![Max Pooling vs. Average Pooling](pics\maxpool_vs_avgpool.png "Text to show on mouseover")
    + Both have inferior effectiveness compared to transposed convolutional layer (it's like trainable pooling).

## 2. Models
### Generative Adversarial Network (GAN)
- Normalizing to the [-1, 1] range (instead of normalizing to mean=0, std=1) is much more appropriate with the generator's output tanh layer.
- Train step can of discriminator can include:
    + Two steps, for real batch and synthetic batch separately. 
    Worked on traditional and deep convolutional networks.
    + One step for a combined batch. Worked on fully-connected models.
- Read list:
    + https://towardsdatascience.com/gan-objective-functions-gans-and-their-variations-ad77340bce3c

## 3. Library
### Tensorflow
#### Subclassed models
- Summary with full output shape: [StackOverflow discussion](https://stackoverflow.com/questions/55235212/model-summary-cant-print-output-shape-while-using-subclass-model)