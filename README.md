# PocketNN
The Linux version of [PocketNN](https://github.com/jaewoosong/pocketnn). 
## Datasets
Two sample datasets are copied from their original website and are stored in `data/`
- MNIST dataset: MNIST dataset is from [the MNIST website](http://yann.lecun.com/exdb/mnist/)
- Fashion-MNIST dataset: Fashion-MNIST dataset is from [its github repository](https://github.com/zalandoresearch/fashion-mnist).

## Repo structure
```
├── data              
├── images      # hold the images in `README.md`
├── libs        # hold the libraries needed
├── src         # hold the source code for training and inferencing
├── tests           # hold the unit tests
└── weights         # hold the trained weights and biases
 ```
## How to run
- `cmake -S . -B build`
- `cmake --build build`
- Run the compiled binary, for example `./build/MyPocketNN`

The result for running a simple 2-layer fully connected network on dummy data is like the following picture

![](./images/fc_int_dfa_simple.png)

The result for training a 3-layer fully connected network on MNIST is in the following picture

![](./images/fc_int_dfa_mnist.png)

The result for inference only with pretrained weights of the 3-layer fully connected network on MNIST is in the following picture

![](./images/fc_int_dfa_mnist_inference.png)