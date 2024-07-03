## [Lec3 youtube](https://www.youtube.com/watch?v=P6sfmUTpUmc&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=6)


## [Jupyter Notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part3_bn.ipynb)

- we want the logits to be roughly 0 at the beggining
- at the begininning we want all the logits to have roughly a uniform distribution
- problem with saturated tanh, when it's near 1 or -1 it kills the gradient. So changing the input  is not going to impact the output of the tanh to much because it's in the flat region of the tanh. **vanishing gradient**
- We don't want the gaussian to have a bigger standard deviation so how do we scale w to remain with the same distribution. We need to divide by the square root of the first dim
- kaiming initialization: most common way to init NN . It uis not to much important today thanks to residual connection, batch normalization, optimizers like RMSProp and Adam

### [kaiming init paper](https://arxiv.org/pdf/1502.01852)


## Batch normalization

### [Batch normalization paper](https://arxiv.org/pdf/1502.03167)

### [problems that can occur with BN](https://arxiv.org/pdf/2105.07576)

- We have the hidden state roughly gaussian at leat at initialization. So BN noramlizes the hidden state to gaussian
- mean over all the elements in the batch
- we aare coupling between the examples in the batch. We can think that as a regularizer
- it stabilizes training
- We want to put it after a multiplitcation layer or a conv layer
- try to avoid this layer and use other forms of normalization