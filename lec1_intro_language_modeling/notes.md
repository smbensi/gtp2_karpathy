# [link to the lecture](https://www.youtube.com/watch?v=PaCmpygFfXo)

# [git](https://github.com/karpathy/makemore)


- makemore is a character level langugage model
- Bigram working with only 2 chars at a time, looking only at the previous char to predict the next one  
- **likelihood** is the product of all the probabilities it's a measure of the quality of the model
- our **goal** is to maximuze the likelihood of the data w.r.t model parameters (statistical modeling)
equivalent to maximizing the log likelihood (because log is monotonic)
equivalent to miniminzing the negative log likelihood
equivalent to miniminzing the average negative log likelihood


## Notes on python

- `zip()` takes 2 iterators and paiirs them up and creates an iterator over the tuples of the consecutive entries and his length is like the shortest of the 2 original iterators 
- `torch.multinomial` returns a sample according to a  propability distribution
- `torch.sum` provides `dim` and `keepdim` . The `dim` will sum up over this dimension if dim=0 the 0 dimension will disappear (`torch.squeeze()`) or will be 1 if keepdim=True
- `Broadcasting semantics` in torch 
- In-place operation is more efficient because it does not have to create new memory