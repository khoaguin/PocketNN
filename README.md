# PocketNN
The Linux version of [PocketNN](https://github.com/jaewoosong/pocketnn). 

## How to run
- `cmake -S . -B build`
- `cmake --build build`
- Run the compiled binary, for example `./build/MyPocketNN`
## Datasets
Two sample datasets are copied from their original website and are stored in `data/`
- MNIST dataset: MNIST dataset is from [the MNIST website](http://yann.lecun.com/exdb/mnist/). The site says "Please refrain from accessing these files from automated scripts with high frequency. Make copies!" So I made the copies and put them in this repository.
- Fashion-MNIST dataset: Fashion-MNIST dataset is from [its github repository](https://github.com/zalandoresearch/fashion-mnist). It follows the MIT License which allows copy and distribution.
