- we want the logits to be roughly 0 at the beggining
- at the begininning we want all the logits to have roughly a uniform distribution
- problem with saturated tanh, when it's near 1 or -1 it kills the gradient. So changing the input  is not going to impact the output of the tanh to much because it's in the flat region of the tanh. **vanishing gradient**
- We don't want the gaussian to have a bigger standard deviation so how do we scale w to remain with the same distribution. We need to divide by the square root of the first dim
- kaiming initialization: most common way to init NN . It uis not to much important today thanks to residual connection, batch normalization, optimizers like RMSProp and Adam


## Batch normalization

- We have the hidden state roughly gaussian at leat at initialization. So BN noramlizes the hidden state to gaussian
- mean over all the elements in the batch
- we aare coupling between the examples in the batch