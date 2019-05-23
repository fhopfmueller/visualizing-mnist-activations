import torch
import mnist_classifier
from torchvision import datasets, transforms
import time

class Timer:
    def __init__(self, msg):
        self.msg=msg
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.end = time.time()
        print(self.msg, self.end-self.start)

#device = torch.device("cuda") not enough mem...
device = torch.device("cpu")

# define and load model
model = mnist_classifier.Net().to(device)
model.load_state_dict(torch.load("data/mnist_cnn.pt"))
model.eval()

# load data
with Timer("callings datasets.MNIST") as t:
    train_data = datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    test_data = datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

with Timer("constructing torch arrays:") as t:
    x_train = [train_data[i][0] for i in range(len(train_data))] # list of torch arrays of [1, 28, 28]
    y_train = [train_data[i][1] for i in range(len(train_data))] # list of ints
    x_train = torch.stack(x_train).to(device)
    y_train = torch.LongTensor(y_train).to(device)

with Timer("running model:") as t:
    with torch.no_grad():
        hidden_activations = model.hidden_activations(x_train)

torch.save(hidden_activations, "data/hidden_activations.pt")
torch.save([x_train, y_train], "data/x_y_train.pt")
