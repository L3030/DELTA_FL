import torch

class CNNNet(torch.nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(in_features=400, out_features=120, bias=True),
            torch.nn.ReLU(),
#             torch.nn.MaxPool2d(2)
            torch.nn.Linear(in_features=120, out_features=84, bias=True),     #120
            torch.nn.ReLU(),
            torch.nn.Linear(84, 10, bias = True)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        res = conv2_out.view(conv2_out.size(0), -1)
#         fc1_out = self.fc1(res)
        # res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return out
    
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64 * 3 * 3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return out

class Logistic(torch.nn.Module):
    def __init__(self, dim_in=28*28, num_classes=10):
        super(Logistic, self).__init__()
        self.layer_input = torch.nn.Linear(dim_in, num_classes)
        # self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, x):
        # print(x.shape)
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        # print(x.shape)
        x = self.layer_input(x)
        return x

class MLP(torch.nn.Module):
    def __init__(self, dim_in=28*28, dim_hidden=64, dim_out=10):
        super(MLP, self).__init__()
        self.layer_input = torch.nn.Linear(dim_in, dim_hidden)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout()
        self.layer_hidden = torch.nn.Linear(dim_hidden, dim_out)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)