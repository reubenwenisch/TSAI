# TSAI

## The Dataset
Here MyDataset consists of using <br>
#### Inputs and Data Representation <br>
* Images from the MNIST dataset
* A random number between 0 and 9 (represented as one hot encoded data) <br>
#### Targets and Data Representation<br>
* the "number" that was represented by the MNIST image
* The "sum" of the MNIST number with the random number (not one hot encoded)
#### Datset Methods used 
##### Data generation strategy
The random number is generated using torch.randint and limited to 10 numbers.
Inheriting from the torchvision.datasets.MNIST to use existing methods to load, download MNIST data to the system.

### Combining the inputs
The MNIST model has conv and classification fc blocks. The output of this is combined with the one hot encoded random number
using concatenate method. Post this there is a small fc network to generate the sum output.

## The Deep Learning model
* Here for the MNIST we have a conv block and a classifier block
* Next the output of this MNIST is taken along with the random number into the mix block to generate the sum output.

```
MyModule(
  (conv): Sequential(
    (0): Conv2d(1, 64, kernel_size=(5, 5), stride=(1, 1))
    (1): ReLU()
    (2): Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1))
    (3): ReLU()
    (4): Conv2d(128, 10, kernel_size=(1, 1), stride=(1, 1))
    (5): ReLU()
    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Linear(in_features=1000, out_features=512, bias=True)
    (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Linear(in_features=512, out_features=256, bias=True)
    (4): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
    (6): Linear(in_features=256, out_features=10, bias=True)
  )
  (mix): Sequential(
    (0): Linear(in_features=20, out_features=50, bias=True)
    (1): ReLU()
    (2): Linear(in_features=50, out_features=20, bias=True)
  )
)
```

## Loss Functions:
2 Loss functions used seperately because these are 2 seperate outputs.
Loss function used is basically CrossEntropyLoss which is basically a combination of (Softmax+ NLL).

# Results
Train epoch: 4 train loss: 0.1481 mnist_accuracy: 0.99 rand_acc: 0.99
Test Accuracy of the MNIST: 0.99 %
Test Accuracy of the random number: 0.98 %

## Training Logs
```
epoch: 0 train loss: 5.3513 mnist_accuracy: 0.11 rand_acc: 0.08
epoch: 0 train loss: 2.9840 mnist_accuracy: 0.91 rand_acc: 0.11
epoch: 0 train loss: 2.7382 mnist_accuracy: 0.94 rand_acc: 0.12
epoch: 0 train loss: 2.6110 mnist_accuracy: 0.95 rand_acc: 0.14
epoch: 0 train loss: 2.5201 mnist_accuracy: 0.96 rand_acc: 0.16
Loss: 0.019246196446971345
epoch: 1 train loss: 2.0617 mnist_accuracy: 0.98 rand_acc: 0.30
epoch: 1 train loss: 1.9656 mnist_accuracy: 0.99 rand_acc: 0.40
epoch: 1 train loss: 1.8766 mnist_accuracy: 0.99 rand_acc: 0.44
epoch: 1 train loss: 1.7779 mnist_accuracy: 0.99 rand_acc: 0.50
epoch: 1 train loss: 1.6756 mnist_accuracy: 0.99 rand_acc: 0.55
Loss: 0.012565091489274831
epoch: 2 train loss: 1.1643 mnist_accuracy: 0.98 rand_acc: 0.79
epoch: 2 train loss: 1.0400 mnist_accuracy: 0.99 rand_acc: 0.85
epoch: 2 train loss: 0.9504 mnist_accuracy: 0.99 rand_acc: 0.88
epoch: 2 train loss: 0.8686 mnist_accuracy: 0.99 rand_acc: 0.90
epoch: 2 train loss: 0.7950 mnist_accuracy: 0.99 rand_acc: 0.92
Loss: 0.005845460194496748
epoch: 3 train loss: 0.4283 mnist_accuracy: 0.99 rand_acc: 0.98
epoch: 3 train loss: 0.3994 mnist_accuracy: 0.99 rand_acc: 0.98
epoch: 3 train loss: 0.3580 mnist_accuracy: 0.99 rand_acc: 0.99
epoch: 3 train loss: 0.3296 mnist_accuracy: 0.99 rand_acc: 0.99
epoch: 3 train loss: 0.3040 mnist_accuracy: 0.99 rand_acc: 0.99
Loss: 0.0022759300541481935
epoch: 4 train loss: 0.1712 mnist_accuracy: 1.00 rand_acc: 0.99
epoch: 4 train loss: 0.1731 mnist_accuracy: 0.99 rand_acc: 0.99
epoch: 4 train loss: 0.1652 mnist_accuracy: 0.99 rand_acc: 0.99
epoch: 4 train loss: 0.1558 mnist_accuracy: 0.99 rand_acc: 0.99
epoch: 4 train loss: 0.1481 mnist_accuracy: 0.99 rand_acc: 0.99
Loss: 0.001122798312648471.
```