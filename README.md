# A unified optimization framework of ANN-SNN Conversion: towards optimal mapping from activation values to firing rates

## Accepted at ICML2023, check [ICML2023 openview](https://openreview.net/forum?id=6nFolgGq9E).


## Usage

Please first change the variable `DIR` at File `.Preprocess\getdataloader.py` (line 12) to your own dataset directory. 


Train model with SlipReLU-Layer 

```
python main.py --action='train' --model={vgg16, resnet18, resnet20, resnet34} --dataset={cifar10, cifar100} --a={0.1, 0.2, ..., 0.9} --l=QUASI_LATENCY

```
Test accuracy in ANN mode or SNN mode

```
python main.py --action='train' -model={vgg16, resnet18, resnet20, resnet34} --dataset={cifar10, cifar100} --mode={ann, snn} --t=SIMULATION_TIME
```
