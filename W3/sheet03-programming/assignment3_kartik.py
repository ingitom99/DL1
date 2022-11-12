import numpy as np
import torch

from torch.utils.data import DataLoader
from torch import optim

from torchvision.datasets import MNIST
from torchvision import transforms as T

import utils
#import solution I am not able to find this file.

data_root = './data'
train_dataset = MNIST(data_root, train = True,download = True, transform = T.ToTensor())
test_dataset = MNIST(data_root, train = False,download = True, transform = T.ToTensor())

utils.show_samples(train_dataset)

def train(model,train_dataset, test_dataset, epochs = 10,batch_size = 10,lr = 1e-2,momentum = 0.0):
    train_loader = DataLoader(train_dataset,batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset,batch_size = batch_size)
    optimizer = optim.SGD(model.parameters(), lr = lr)
    criterion = torch.nn.CrossEntropyLoss()
    arr_epoch_loss = []
    arr_epoch_train_accuracy = [];
    for i in range(0,epochs):

        Loss = 0
        for j,data in enumerate(train_loader,0):
            inputs,label = data

            optimizer.zero_grad()
            output = model(inputs)

            loss = criterion(output,label)
            loss.backward()
            optimizer.step()
            Loss += loss.item()

        arr_epoch_loss.append(Loss)
        train_acc = utils.accuracy(model, train_loader)
        arr_epoch_train_accuracy.append(train_acc)
        test_acc = utils.accuracy(model, test_loader)

    return train_dataset,test_acc, arr_epoch_loss, arr_epoch_train_accuracy






if __name__ == "__main__":
    torch.manual_seed(1)
    model = utils.Lenet5()
    train(model, train_dataset, test_dataset,epochs = 1, momentum=1e-1) 

    model2 = utils.Lenet5()
    train(model2, train_dataset, test_dataset, epochs = 1, momentum=2e-1)


