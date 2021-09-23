import torch
import torch.nn as nn
import torch.nn.functional as F

def Classifier(input_shape, num_classes, hparams):
    if input_shape[0] == 1:
        return MNISTNet(input_shape, num_classes)
    else:
        assert False


class MNISTNet(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)    #TODO(AR): might need to remove softmax for KL div in TRADES