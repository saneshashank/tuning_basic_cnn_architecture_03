# Tuning Basic CNN Architecture for MNIST

A comprehensive exploration into optimizing CNN architectures for MNIST to strike the best balance between **accuracy** and **parameter count**.

## 🎯 Project Overview

This project demonstrates systematic deep learning model optimization—balancing compactness, speed, and accuracy using best practices in PyTorch. Through three progressive iterations, we explore how to achieve >99.4% accuracy on MNIST with minimal parameters.

## 📁 Project Structure

```
tuning_basic_cnn_architecture_03/
├── tuning_CNN_MNIST_01.ipynb    # Baseline large model (~15k parameters)
├── tuning_CNN_MNIST_02.ipynb    # Small model (<5k parameters)
├── tuning_CNN_MNIST_03.ipynb    # Final optimized model (<8k parameters)
├── model_1.py                   # Model definition for Notebook 01
├── model_2.py                   # Model definition for Notebook 02
├── model_3.py                   # Model definition for Notebook 03
└── README.md                    # This file
```

## 📊 Model Architecture Progression

| Notebook | Parameters | Best Test Acc | Architecture Notes |
|----------|------------|---------------|--------------------|
| 01 | 15,076 | 99.41% | Big, GAP, no dropout |
| 02 | 3,272 | 98.83% | Very small, underfits, some augmentation |
| 03 | 7,760 | 99.40% | Optimal, GAP + conv post-pool, aug, tuned LR, dropout |

## 🔬 Detailed Notebook Analysis

### Notebook 01: Large First Model (~15k params)

**Purpose:**
- Develop a robust CNN with more than 10k but less than 20k parameters
- Use **Global Average Pooling (GAP)** instead of traditional fully connected layers
- Explore effects of max-pooling position and batch size

**Key Results:**
- Params: **15,076**
- Best Train Acc: **99.47%**
- Best Test Acc: **99.41%**

**Key Insights:**
- GAP can effectively replace FC layers without loss in accuracy
- Lower gap between train and test accuracy (no overfitting)
- No image augmentation needed for strong result
- Suggests possible parameter reduction without much loss
- Max-pooling layer placement is crucial for optimal feature extraction

### Notebook 02: Very Small Model (<5k params)

**Purpose:**
- Test how far the parameter count can be reduced
- <5,000 param network with simple augmentations

**Key Results:**
- Params: **3,272**
- Best Train Acc: **98.39%**
- Best Test Acc: **98.83%**

**Key Insights:**
- Could not reach 99% test accuracy in 15 epochs—model underfits
- Output channels are likely too low; ~10k may be a sweet spot
- Image augmentation (rotation) applied
- Dropout avoided since underfitting was observed

### Notebook 03: Final Model (<8k params, with Augmentation)

**Purpose:**
- Find an optimal minimal model that achieves **>99.4% test accuracy** in 15 epochs
- Apply augmentation, LR tuning, and judicious dropout

**Key Results:**
- Params: **7,760**
- Best Train Acc: **98.86%**
- Best Test Acc: **99.40%**

**Key Insights:**
- Initial output channel size of 10 (matching MNIST classes) helps compactness
- After several iterations, a model with ~7.8k params hit >99.4% accuracy
- Adding a convolution after GAP further improved performance
- Data augmentation (rotation + color jitter) proved beneficial
- Best results with learning rate 0.045, batch size 128, LR scheduler, 5% dropout

## 🛠️ Installation & Setup

### Prerequisites

```bash
pip install torch torchvision matplotlib tqdm numpy
```

### Quick Start

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from model_3 import Net  # Import the optimized model

# Load the optimized model
model = Net()
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# Load MNIST data
transform = transforms.Compose([
    transforms.RandomRotation(7),
    transforms.ColorJitter(brightness=0.4),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
```

## 📈 Model Architecture (Final Optimized - Model 3)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block 28
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # nn.BatchNorm2d(8),
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.BatchNorm2d(16),

        ) # output_size = 24
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.BatchNorm2d(16),

        ) # output_size = 22 

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            # nn.BatchNorm2d(8),

        ) # output_size = 11

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.BatchNorm2d(16),

        ) # output_size = 9 
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            # nn.BatchNorm2d(16),

        ) # output_size = 9 
        

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3,3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            # nn.BatchNorm2d(10),

        ) # output_size = 7 
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7) 
        ) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            # nn.BatchNorm2d(10),

        )

        self.dropout = nn.Dropout(0.05) # Reduced dropout rate

    def forward(self, x):
        x = self.convblock1(x)
        x = self.dropout(x)
        x = self.convblock2(x)
        x = self.dropout(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.dropout(x)
        x = self.convblock5(x)
        x = self.dropout(x)
        x = self.convblock6(x)
        x = self.dropout(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
```

## 🚀 Usage Instructions

### 1. Training from Scratch

```python
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Initialize model, optimizer, and scheduler
model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.045, momentum=0.9)
scheduler = StepLR(optimizer, step_size=8, gamma=0.1)

# Training loop
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]'
                  f' Loss: {loss.item():.6f}')

# Run training for 15 epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(1, 16):
    train(model, device, train_loader, optimizer, epoch)
    scheduler.step()
```

### 2. Quick Evaluation

```python
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

# Evaluate the model
test_accuracy = test(model, device, test_loader)
```

### 3. Running Individual Notebooks

```bash
# Start with the final optimized model
jupyter notebook tuning_CNN_MNIST_03.ipynb

# Or explore the progression
jupyter notebook tuning_CNN_MNIST_01.ipynb  # Baseline
jupyter notebook tuning_CNN_MNIST_02.ipynb  # Small model
jupyter notebook tuning_CNN_MNIST_03.ipynb  # Final optimized
```

## 📊 Performance Visualization

The notebooks include comprehensive visualizations:

- **Training/Validation Loss Curves**: Track model convergence
- **Accuracy Progression**: Monitor improvement across epochs
- **Parameter Count Comparison**: Visualize efficiency gains
- **Confusion Matrices**: Analyze classification performance
- **Sample Predictions**: Verify model behavior on test images

## 🎯 Reproducing Results

### For Notebook 03 (Recommended):

1. **Environment Setup**:
   ```bash
   pip install torch torchvision matplotlib tqdm
   ```

2. **Data Preparation**:
   - MNIST dataset will be automatically downloaded
   - Data augmentation: RandomRotation(7°) + ColorJitter(brightness=0.4)

3. **Training Configuration**:
   - Learning Rate: 0.045
   - Batch Size: 128
   - Optimizer: SGD with momentum=0.9
   - Scheduler: StepLR (step_size=8, gamma=0.1)
   - Dropout: 5%
   - Epochs: 15

4. **Expected Results**:
   - Parameters: ~7,760
   - Test Accuracy: >99.4%
   - Training Time: ~15 minutes on Collab GPU

### Hyperparameter Sensitivity:

- **Learning Rate**: 0.045 works best; 0.01 too slow, 0.1 too aggressive
- **Batch Size**: 128 optimal; 64 slightly slower convergence
- **Dropout**: 5% prevents overfitting without hampering learning
- **Augmentation**: Rotation + ColorJitter essential for final accuracy boost

## 🔑 Key Learnings

1. **Parameter Efficiency**: Reduction to 7.8k parameters (from 15k) with minimal accuracy loss
2. **Global Average Pooling**: Strong alternative to dense FC layers in image classification
3. **Strategic Augmentation**: Rotation + color jitter crucial for pushing final accuracy
4. **Architecture Design**: Initial channel count matching output classes aids compactness
5. **Training Strategy**: LR scheduling and minimal dropout essential for optimization
6. **Sweet Spot**: <8k params achieves >99.4% accuracy with strong generalization

## 🏆 Best Practices Demonstrated

- **Progressive Model Development**: Start large, then optimize systematically
- **GAP over FC**: Reduces parameters while maintaining performance
- **Appropriate Regularization**: Balance between underfitting and overfitting
- **Hyperparameter Tuning**: Systematic approach to LR, batch size, and augmentation
- **Efficient Architecture**: Thoughtful channel progression and pooling placement

## 📚 Further Reading

- [Global Average Pooling Paper](https://arxiv.org/abs/1312.4400)
- [PyTorch CNN Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Data Augmentation Techniques](https://pytorch.org/vision/stable/transforms.html)

---

**Note**: This project serves as an excellent template for efficient CNN design and demonstrates that significant parameter reduction is possible without sacrificing accuracy through careful architectural choices and training strategies.
