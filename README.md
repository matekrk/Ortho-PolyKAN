# Ortho-PolyKAN
## Orthogonal polynomials for Kolmogorov Arnold Networks

## Note
The code is heavily based on:
- Efficient KAN https://github.com/Blealtan/efficient-kan
- ChebyKAN https://github.com/SynodicMonth/ChebyKAN

## Set up
- conda with python
- numpy, matplotlib, torch, torchvision, wandb?

## Examples
- ``` python src/main.py baseline_lenet mnist --verbose --save_model --cuda --objective_str ce --optimizer_str adam --lr 0.001 --n_epochs 2 ```
- ``` python src/main.py baseline_fc mnist --save_model --cuda --seed 8 --objective_str mse --optimizer_str adam --lr 0.0005 --n_epochs 3 --widths 64 16 --activation elu ```
- ``` python src/main.py baseline_conv mnist --verbose --save_model --cuda --objective_str ce --optimizer_str sgd --lr 0.001 --n_epochs 2 --widths 16 16 --bn --activation leaky_relu --pooling max ```
- ``` python src/main.py kan_vanilla mnist --cuda --objective_str ce --optimizer_str adam --lr 0.001 --n_epochs 2 --layers_hidden 784 64 10 ```
- ``` python src/main.py kan_cheby mnist --seed 42 --cuda --objective_str mse --optimizer_str adam --lr 0.001 --n_epochs 2 ```

## TODOs:
- [x] models,data,train,evaluation,plotting code structure
- [x] support mnist
- [ ] support other datasets
- [ ] make splines, KANlinears etc. universal accross KAN types
- [ ] save the optimal models / hyperparams
- [ ] legendre polynomials
- [ ] wandb logger
- [ ] notebook guides
- [ ] consistency at comments / documentation

### Meet us
We are Group of Machine Learning Research at the Jagiellonian University in Krakow, Poland. Find out more https://gmum.net/
