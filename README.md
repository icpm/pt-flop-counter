# Flops counter for :fire:torch CV models

## How to install

`git submodule add git@github.com:icpm/pt-flop-counter.git`

## How to use

```python
from torchvision.models import resnet50
from pt-flop-counter import profile
model = resnet50()
flops, params = profile(model, input_size=(1, 3, 224,224))
```

## Results on Recent Models

| Model         | Params(M) | FLOPs(G) |
| ------------- | --------- | -------- |
| alexnet       | 61.10     | 0.71     |
| vgg11         | 132.86    | 7.75     |
| vgg11_bn      | 132.87    | 7.76     |
| vgg13         | 133.05    | 11.46    |
| vgg13_bn      | 133.05    | 11.48    |
| vgg16         | 138.36    | 15.62    |
| vgg16_bn      | 138.37    | 15.65    |
| vgg19         | 143.67    | 19.79    |
| vgg19_bn      | 143.68    | 19.82    |
| densenet121   | 7.98      | 2.79     |
| densenet161   | 28.68     | 7.69     |
| densenet169   | 14.15     | 3.33     |
| densenet201   | 20.01     | 4.28     |
| resnet18      | 11.69     | 1.58     |
| resnet34      | 21.80     | 3.44     |
| resnet50      | 25.56     | 3.53     |
| resnet101     | 44.55     | 7.26     |
| resnet152     | 60.19     | 10.99    |
| squeezenet1_0 | 1.25      | 0.70     |
| squeezenet1_1 | 1.24      | 0.34     |
