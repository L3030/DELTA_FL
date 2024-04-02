import torch
from PIL import Image

class Net1(torch.nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
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
            torch.nn.Linear(84, 2, bias = True)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        res = conv2_out.view(conv2_out.size(0), -1)
#         fc1_out = self.fc1(res)
        # res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return out

    # def _load_image(self, img_dir):
    #     batch_x = []
    #    for img_pth in img_dir:
    #        img = Image.open(os.path.join(raw_dir, img_pth))
    #        img = torch.tensor(np.array(img)).float() / 255
    #    return img.permute(2, 0, 1).unsqueeze(0)
class Net_celeba(torch.nn.Module):
    def __init__(self):
        super(Net_celeba, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 84, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(84, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64 * 4 * 4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        # print(conv3_out.shape)
        res = conv3_out.view(conv3_out.size(0), -1)
        # print(res.shape)
        out = self.dense(res)
        return out
class CNN_feature(torch.nn.Module):
    def __init__(self):
        super(CNN_feature, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.fc1 = torch.nn.Linear(256 * 6 * 6, 512)
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.conv4(x))
        x = x.view(-1, 256 * 6 * 6)
        x = torch.nn.functional.relu(self.fc1(x))
        
        return x  
    
class CNN_chatgpt(torch.nn.Module):
    def __init__(self):
        super(CNN_chatgpt, self).__init__()
        self.feature = CNN_feature()
        self.classifier = torch.nn.Linear(512, 2)
        
    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x    
    
class Net_feature(torch.nn.Module):
    def __init__(self):
        super(Net_feature, self).__init__()
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
    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        return res
    
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.feature = Net_feature()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(64 * 3 * 3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 62)
        )

    def forward(self, x):
        res = self.feature(x)
        out = self.classifier(res)
        return out

class CNNWithoutBatchNorm(torch.nn.Module):
    def __init__(self):
        super(CNNWithoutBatchNorm, self).__init__()
        self.block1 = torch.nn.ModuleDict(
            {
                "conv": torch.nn.Conv2d(1, 64, 5, 1, 2),
                # "bn": torch.nn.BatchNorm2d(64),
                "relu": torch.nn.ReLU(True),
                "pool": torch.nn.MaxPool2d(2),
            }
        )
        self.block2 = torch.nn.ModuleDict(
            {
                "conv": torch.nn.Conv2d(64, 64, 5, 1, 2),
                # "bn": torch.nn.BatchNorm2d(64),
                "relu": torch.nn.ReLU(True),
                "pool": torch.nn.MaxPool2d(2),
            }
        )
        self.block3 = torch.nn.ModuleDict(
            {"conv": torch.nn.Conv2d(64, 128, 5, 1, 2), 
            #  "bn": torch.nn.BatchNorm2d(128),
             "relu": torch.nn.ReLU(True),}
        )

        self.block4 = torch.nn.ModuleDict(
            {"fc": torch.nn.Linear(128*7*7, 2048), "relu": torch.nn.ReLU(True)}
        )
        self.block5 = torch.nn.ModuleDict({"fc": torch.nn.Linear(2048, 512), "relu": torch.nn.ReLU(True)})
        self.block6 = torch.nn.ModuleDict({"fc": torch.nn.Linear(512, 10)})

    def forward(self, x):
        x = self.block1["conv"](x)
        x = self.block1["relu"](x)
        x = self.block1["pool"](x)

        x = self.block2["conv"](x)
        x = self.block2["relu"](x)
        x = self.block2["pool"](x)

        x = self.block3["conv"](x)
        x = self.block3["relu"](x)

        x = x.view(x.shape[0], -1)

        x = self.block4["fc"](x)
        x = self.block4["relu"](x)

        x = self.block5["fc"](x)
        x = self.block5["relu"](x)

        x = self.block6["fc"](x)
        return x


    
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
        # self.dropout = torch.nn.Dropout()
        self.layer_hidden = torch.nn.Linear(dim_hidden, dim_out)
        # self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        # x = self.dropout(x) 
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x
    
class MLP_LEAF(torch.nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.fc1(x))
        y = self.fc2(x)
        return y
    
class MLP_CIFAR(torch.nn.Module):
    def __init__(self, dim_in=28*28*3, dim_hidden=64, dim_out=10):
        super(MLP_CIFAR, self).__init__()
        self.layer_input = torch.nn.Linear(dim_in, dim_hidden)
        self.relu = torch.nn.ReLU()
        # self.dropout = torch.nn.Dropout()
        self.layer_hidden = torch.nn.Linear(dim_hidden, dim_out)
        # self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        # x = self.dropout(x) 
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x