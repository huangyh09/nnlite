# nnwrapper
A light toolbox with some utilities and wrappers for Neural Network Models


## Install

```
pip install -U nnwrapper

pip install -U git+https://github.com/huangyh09/nnwrapper
```

## Quick Usage

```python
from functools import partial
from nnwrapper import NNWrapper

torch.manual_seed(0)
dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'

## VAE model (one hidden layer, dim=64), loss, and optimizer
model = nnwrapper.models.VAE_base(1838, 32, hidden_dims=[64], device=dev)
criterion = partial(nnwrapper.models.Loss_VAE_Gaussian, beta=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.95)

## NNWrapper for model training
my_wrapper = NNWrapper(model, criterion, optimizer, device=dev)
my_wrapper.fit(train_loader, epoch=3000, validation_loader=None, verbose=False)

plt.plot(my_wrapper.train_losses)
```


## Examples
See the [examples](./examples) folder, including
* CNN-1D: []
* VAE for 3K PBMC: []
* and more


