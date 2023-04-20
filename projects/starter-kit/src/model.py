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
        self.conv1 = nn.Conv2d(3, 8, 5)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5)

        self.dropout = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(44944, 256*4) # 53*53*16, where 53 is close to 224/4. used error msg for first input dim
        self.batch_norm_1 = nn.BatchNorm1d(256*4)
        self.fc2 = nn.Linear(256*4, 128*4)
        self.batch_norm_2 = nn.BatchNorm1d(128*4)
        self.fc3 = nn.Linear(128*4, num_classes)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)

        x = self.dropout(self.batch_norm_1(F.relu(self.fc1(x))))
        x = self.dropout(self.batch_norm_2(F.relu(self.fc2(x))))
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
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
