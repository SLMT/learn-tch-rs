# My [tch-rs](https://github.com/LaurentMazare/tch-rs) Learning Playground

Since I am recently working on a big project that needs to use reinforcement learning on Rust, I created this repository for me to learn how to use [tch-rs](https://github.com/LaurentMazare/tch-rs), which is a wrapper written in Rust that provides a interface to C++ PyTorch (libtorch).

## Setup Environment

Just follow the guideline in the README of [tch-rs](https://github.com/LaurentMazare/tch-rs) and run `cargo run --example basics` on `tch-rs`'s repository to test if everything works.

Note: the official PyTorch package has already had built-in CUDA runtime, so we don't need to install CUDA toolkit.

## OpenAI Gym

In order to test reinforcement learning methods, we need to use [OpenAI Gym](https://github.com/openai/gym) to setup environments, so we require OpenAI Gym to be installed on the host machine.

First, the host machine must have Python runtime, and install OpenAI Gym via the following command:

```
> pip install gym
```

To test if we can run OpenAI Gym properly, run the following command:

```
> cargo run --bin gym_test
```

## Setup Data Sets

### MNIST

1. Download the data set from [the website](http://yann.lecun.com/exdb/mnist/)
2. Unpack all the data set
3. Ensure that all the data files are exactly named as follows:
    - `train-images-idx3-ubyte`
    - `train-labels-idx1-ubyte`
    - `t10k-images-idx3-ubyte`
    - `t10k-labels-idx1-ubyte`

## Available Binaries

Each of the following binaries can be run with `cargo run --binary [NAME]`.

- `mnist_dense`: an example code for training a dense net with MNIST data set
- `mnist_cnn`: an example code for training a CNN with MNIST data set
- `gym_test`: a test for checking if it can run [OpenAI Gym](https://github.com/openai/gym) properly.
