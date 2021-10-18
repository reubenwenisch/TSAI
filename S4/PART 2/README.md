# TSAI - S4

Here we are attempting to train a neural network with < 20k params and achieve > 99% accuracy

### Accuracy

```The accuracy reached = 
The accuracy reached = 98.48% on epoch 16 
Total params: 18,240
```

### Model

The model consists of a conv block followed by a GAP layer then a FC layer with log softmax as the activation function.

```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 7),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(0.2),
            nn.Conv2d(32, 50, 3),
            nn.ReLU(),
            nn.BatchNorm2d(50),
            nn.Conv2d(50, 32, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.fc1 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.adaptive_avg_pool2d(x, (1, 1)))
        x = x.view(-1, 32)
        x = self.fc1(x)
        return F.log_softmax(x)
```

### Data Augmentations used

Affine transforms was used this essentially bends the image across multiple axis.

```
transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9, 1.1))
```

## Loss and optimization algorithm used

Adam optimizer was used along with NLL as the loss function.

```
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss = F.nll_loss(output, target)
```

