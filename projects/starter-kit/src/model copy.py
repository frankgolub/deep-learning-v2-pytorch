import torch
import torch.nn as nn
import torch.nn.functional as F


# define the CNN architecture
# The architecture of the following network comports with recommendations from the CS231 and the 
# Pytorch classification example. Batch-norm could be used as a regularizer.
# The pattern follows: INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC, with the recommendation
# that K < 3, M > 0, and pooling in force. A filter size of 5 is sufficient to avoid zero-padding and fitting to 
# a substantial number of zero's in hidden layers that are not representative of the image.

# The cifar10_cnn_exercise.ipynb was specifically relied upon.

class MyModel(nn.Module):
    def __init__(self, num_classes: int = 50, dropout: float = 0.2) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))

        # convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, 
                               out_channels=96,
                               kernel_size=11,
                               stride=4,
                               padding=3)
        self.max1 = nn.MaxPool2d(kernel_size=(3, 3),
                                 stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, 
                               out_channels=256,
                               kernel_size=5,
                               padding=2)
        self.max2 = nn.MaxPool2d(kernel_size=(3, 3),
                                 stride=2)        
        self.conv3 = nn.Conv2d(in_channels=256, 
                               out_channels=384,
                               kernel_size=3,
                               padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, 
                               out_channels=384,
                               kernel_size=3,
                               padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, 
                               out_channels=256,
                               kernel_size=3,
                               padding=1)                 
        self.max3 = nn.MaxPool2d(kernel_size=(3, 3),
                                 stride=2) 
        self.fc1 = nn.Linear(in_features=9216,
                             out_features=4096)
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(in_features=4096,
                             out_features=4096)
        self.dropout2 = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(in_features=4096,
                             out_features=num_classes) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)

        x = self.max1(F.relu(self.conv1(x)))
        x = self.max2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.max3(F.relu(self.conv5(x)))

        x = torch.flatten(x, 1)

        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter) # dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
