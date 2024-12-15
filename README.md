# nnlite
A light toolbox with utilities and wrappers for Neural Network Models


## Install

```
# for published version
pip install -U nnlite

# or developing version
pip install -U git+https://github.com/huangyh09/nnlite
```

## Quick Usage

```python
import nnlite
from functools import partial

torch.manual_seed(0)
dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'

## VAE model (one hidden layer, dim=64), loss, and optimizer
model = nnlite.models.VAE_base(1838, 32, hidden_dims=[64], device=dev)
criterion = partial(nnlite.models.Loss_VAE_Gaussian, beta=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.95)

## NNWrapper for model training
my_wrapper = nnlite.NNWrapper(model, criterion, optimizer, device=dev)
my_wrapper.fit(train_loader, epoch=3000, validation_loader=None, verbose=False)
my_wrapper.predict(test_loader)

plt.plot(my_wrapper.train_losses)
```


## Examples
See the [examples](./examples) folder, including
* CNN-1D: [CamoTSS-CNN-demo.ipynb](./examples/CamoTSS-CNN-demo.ipynb)
* VAE for 3K PBMC: [PBMC3K_VAE.ipynb](./examples/PBMC3K_VAE.ipynb)
* and more.
