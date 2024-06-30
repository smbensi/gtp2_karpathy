## [youtube video](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=4)

## [git](https://github.com/karpathy/makemore)

## [paper bengio](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

- through the embedding space you can transfer knowledge


## tips python

- in PyTorch you can index with  matrices of integers
- `unbind` function in Torch returns a tuple of all slices along a given dimension,alreay without it
- the `view` operation in PyTorch is **extremely efficient**. In each tensor, there is something called the `storage()` it's all the number always in 1-dim vector and this is how the vector is represented in computational memory. With `view()` no memory is created 
- in **PyTorch cross-entropy** the adding of offset does not change the probability so it take the maximum value in the logits and substract it to all the members