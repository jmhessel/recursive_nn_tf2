# Tensorflow2 Recursive NNs

Recursive neural networks are similar to recurrent neural networks in the sense that they involve repeated application of the same network weights to update internal representations. However, instead of operating in a linear fashion, the topology of the network is dynamically determined for each input example according to a specified tree structure (e.g., a parse tree, etc.). Fig 1 from [Tai, Socher, and Manning (2015)](https://arxiv.org/pdf/1503.00075.pdf) summarizes nicely:

<p align="center">
  <img width="300" src="https://github.com/jmhessel/recursive_nn_tf2/raw/master/treelstm.png">
</p>

## What features are supported?

- Make your own Tree recursive nets! A base class that can be extended by overriding the `_combine_inner` function. See the in-line comments in `recursive_nn.py` for info about inputs/outputs.
- Support for (dynamically) passing features to each node of the tree.
- An implementation of Child-Sum Tree-LSTMs from Tai, Socher, and Manning (2015). (`tree_lstm.py`)
- See the minimal `main` in both files for API examples.

## What is required?

- I built this on tensorflow 2.2, but it probably will work with tensorflow 2.1 or above.

## A few notes

- I have not throughly tested this implementation: bugfixes are welcomed!
- There may be more efficient and clever ways of implementing this functionality; I mostly wanted to play with `dynamic=True` functionality in tf.keras
