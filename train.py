# Import libraries
import pandas as pd
import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import json
from collections import OrderedDict

import argparse

# variables for default values incl. hyperparameters
DEFAULT_ARCH = 'densenet121'
ALLOWED_ARCHS_LIST = ['densenet121', 'densenet161', 'densenet169']
LEARNING_RATE = 0.0003
HIDDEN_UNITS = 512
EPOCHS = 3
CHECKPOINT = 'checkpoint.pth'


# setup of argparse (command-line parsing module)
parser = argparse.ArgumentParser(
    description='Program to train image classifier using neural network',
)

# arguments
parser.add_argument('data_path', action="store")
parser.add_argument("-a", "--arch", type=str, help="architecture options: 'densenet121'(default), 'densenet161', 'densenet169' ")
parser.add_argument("--save_dir", type=str, help="save directory, default is 'checkpoint.pth' ")
parser.add_argument("-lr", "--learning_rate", type=float, help="small positive number related to step size in minimizing loss, default: 0.0003 ")
parser.add_argument("-hu", "--hidden_units", type=int, help="number of units in hidden layer in neural network ")
parser.add_argument("-e", "--epochs", type=int, help="number of epochs during training of the network, default: 1 ")
parser.add_argument('--gpu', default=False, action='store_true', help= "if --gpu argument is NOT included, cpu will be used even if gpu is available")

results = parser.parse_args()

# confirmation of provided data path
print('\nData Path = {!r}'.format(results.data_path))

# confirmation of architecture
if not results.arch:
    # if no such input, use default value
    architecture = DEFAULT_ARCH
elif results.arch in ALLOWED_ARCHS_LIST:
    # if the provided input is allowed, use it
    architecture = results.arch
else:
    # if provided input is not allowed, throw ValueError
    raise ValueError('The input value for architecture is not allowed. Please see --help for more details')

print('Architecture = {!r}'.format(architecture))

# confirmation of save directory
if not results.save_dir:
    save_dir = CHECKPOINT
else:
    save_dir = results.save_dir
    
print('Save directory = {!r}'.format(save_dir))


# confirmation of learning rate
if not results.learning_rate:
    learning_rate = LEARNING_RATE
else:
    learning_rate = results.learning_rate
    
print('Learning rate = {!r}'.format(learning_rate))

# confirmation of hidden units
if not results.hidden_units:
    hidden_units = HIDDEN_UNITS
else:
    hidden_units = results.hidden_units
    
print('Hidden units = {!r}'.format(hidden_units))



# confirmation of epochs
if not results.epochs:
    epochs = EPOCHS
else:
    epochs = results.epochs
    
print('Epochs = {!r}\n'.format(epochs))



# directories based on expected data folder structure
data_dir = results.data_path
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define transforms for the training, validation, and testing sets

batch_size = 32

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

valid_test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform = valid_test_transforms)
test_data = datasets.ImageFolder(test_dir, transform = valid_test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Create dict 'cat_to_name' of integer category label to actual name of flowers
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# extracting images and labels from the iterable 'dataloaders'
images, labels = next(iter(trainloader))

# define model
if architecture == 'densenet121':
    model = models.densenet121(pretrained=True)
    
elif architecture == 'densenet161':
    model = models.densenet161(pretrained=True)
    
elif architecture == 'densenet169':
    model = models.densenet169(pretrained=True)
    
else:
    # if provided input is not allowed, throw ValueError
    raise ValueError('The input value for architecture is not allowed. Please see --help for more details')

# keep variable for expected in_features
in_features = model.classifier.in_features
print('Input features = {!r}'.format(in_features))
    
# turn off gradient tracking
for param in model.parameters():
    param.requires_grad = False

# define classifier and add to model
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(in_features, 500)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(500, 200)),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(200, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier

# build network

# Use GPU if it's available (if requested in input)
if results.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    # note: if --gpu argument is not included, cpu will be used even if gpu is available
    device = torch.device("cpu")


# turn off gradients for our model
for param in model.parameters():
    param.requires_grad = False
    
# define classifer
model.classifier = nn.Sequential(nn.Linear(in_features, hidden_units),
                          nn.ReLU(),
                          nn.Dropout(p=0.2),
                          nn.Linear(hidden_units, 102),
                          nn.LogSoftmax(dim=1))

# define loss
criterion = nn.NLLLoss()

# define optimizer
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# move model to available device whether it is gpu or cpu
model.to(device);

# train network
print("Training starting... Please wait")


steps = 0
running_loss = 0
print_every = 25
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # forward pass
        logps = model.forward(inputs)
        
        # loss
        loss = criterion(logps, labels)
        
        # backpropagation
        loss.backward()
        
        optimizer.step()
        
        # add to running loss
        running_loss += loss.item()
        
        # do validation cycle every 1 out of 'print_every' times
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            
            # model to evaluation mode
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # forward pass
                    logps = model.forward(inputs)
                    
                    # loss
                    batch_loss = criterion(logps, labels)
                    
                    # add to accumulated loss
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy in percent
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                  f"Valid accuracy: {accuracy/len(validloader):.3f}")
            
            # reset running loss per epoch
            running_loss = 0
            
            # revert to training mode
            model.train()
            
       
print("Training complete!")

# Save checkpoint 

# assign 'class_to_idx' as attribute to model
model.class_to_idx = trainloader.dataset.class_to_idx



# define checkpoint with parameters to be saved

checkpoint = {'classifier': model.classifier,
              'learning_rate': LEARNING_RATE,
              'class_to_idx': model.class_to_idx,
              'state_dict': model.classifier.state_dict(),
              'optimizer_state_dict': optimizer.state_dict()}


# save checkpoint
torch.save(checkpoint, save_dir)