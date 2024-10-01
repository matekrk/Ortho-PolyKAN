# Ortho-PolyKAN
## Orthogonal polynomials for Kolmogorov Arnold Networks

## Note
The code is based on:
- Efficient KAN https://github.com/Blealtan/efficient-kan
- ChebyKAN https://github.com/SynodicMonth/ChebyKAN
- TorchKAN https://github.com/1ssb/torchkan

## Set up
- conda with python
- numpy, matplotlib, torch, torchvision, wandb?

## Examples
- ``` python src/main.py baseline_lenet mnist --verbose --save_model --cuda --objective_str ce --optimizer_str adam --lr 0.001 --n_epochs 2 ```
- ``` python src/main.py baseline_fc fashion --save_model --cuda --seed 8 --objective_str mse --optimizer_str adam --lr 0.0005 --n_epochs 3 --widths 64 16 --activation elu ```
- ``` python src/main.py baseline_conv mnist --verbose --save_model --cuda --objective_str ce --optimizer_str sgd --lr 0.001 --n_epochs 2 --widths 16 16 --bn --activation leaky_relu --pooling max ```
- ``` python src/main.py kan_vanilla mnist --cuda --objective_str ce --optimizer_str adam --lr 0.001 --n_epochs 2 --widths 784 64 10 --activation silu ```
- ``` python src/main.py kan_relu mnist --cuda --objective_str ce --optimizer_str adam --lr 0.001 --n_epochs 2 --widths 64 10 --activation relu --relu_grid_size 5 --relu_k 3 --relu_train_boundary --init_feature_extractor --verbose ```
- ``` python src/main.py kan_chebyshev mnist --seed 42 --cuda --objective_str mse --optimizer_str adam --lr 0.001 --n_epochs 2 --widths 64 64 --activation relu --init_feature_extractor --polynomial_order 4 --layer_norm```
- ``` python src/main.py kan_legendre svhn --seed 42 --cuda --objective_str mse --optimizer_str adam --lr 0.001 --n_epochs 2 --widths 64 64 --activation gelu --init_feature_extractor --polynomial_order 4 --layer_norm```

## TODOs:
- [x] models,data,train,evaluation,plotting code structure
- [x] support mnist and fashion
- [x] support cifar and svhn
- [ ] support other datasets
- [x] make splines, KANlinears etc. universal accross KAN types
- [ ] add conv layer @mmiezianko
- [x] fix relu kan equal_size_conv
- [ ] save the optimal models / hyperparams
- [x] legendre|chebyshev polynomials
- [ ] wandb logger
- [ ] notebook guides
- [ ] consistency at comments / documentation

### Meet us
We are Group of Machine Learning Research at the Jagiellonian University in Krakow, Poland. Find out more https://gmum.net/

Directly meet me: https://mateusz-pyla.u.matinf.uj.edu.pl 
