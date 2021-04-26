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

# variables for default values
TOP_K = 1


# setup of argparse (command-line parsing module)
parser = argparse.ArgumentParser(
    description='Program to train image classifier using neural network',
)

# arguments
parser.add_argument('image_path', action="store", help="for instance: 'flowers/test/1/image_06743.jpg' ")
parser.add_argument("checkpoint", action="store", help="for instance: 'checkpoint.pth' ")
parser.add_argument("-cn", "--category_names", type=str, help="add directory of json file mappings, for instance 'cat_to_name.json' ")
parser.add_argument("-tk", "--top_k", type=int, help="top k probable categories to output")
parser.add_argument('--gpu', default=False, action='store_true', help= "if --gpu argument is NOT included, cpu will be used even if gpu is available")

results = parser.parse_args()

image_path = results.image_path
checkpoint = results.checkpoint

# confirmation of provided data path
print('\nImage Path = {!r}'.format(image_path))
print('Checkpoint = {!r}\n'.format(checkpoint))

# setting top_k
if not results.top_k:
    top_k = TOP_K
else:
    top_k = results.top_k

#raise ValueError('The input value for architecture is not allowed. Please see --help for more details')

if results.category_names:
    with open(results.category_names, 'r') as f:
        cat_to_name = json.load(f)

# Use GPU if it's available (if requested in input)
if results.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    # note: if --gpu argument is not included, cpu will be used even if gpu is available
    device = torch.device("cpu")

# Function that loads a checkpoint and rebuilds the model

def build_model_from_checkpoint(checkpoint_path):
    '''
    This function loads a checkpoint, builds and returns the model.
    '''

    checkpoint = torch.load(checkpoint_path)
    
    model = models.densenet121(pretrained=True)
    
    
    for param in model.parameters():
        param.requires_grad = False  
        
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier.load_state_dict(checkpoint['state_dict'])
    
    learning_rate = checkpoint['learning_rate']
    
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])                    
    return optimizer, model

# build model from checkpoint
optimizer, model = build_model_from_checkpoint(checkpoint)


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array
    '''    
    # TODO: Process a PIL image for use in a PyTorch model 
    im = Image.open(image)
    width, height = im.size
    if width > height:
        ratio=width/height
        im.thumbnail((ratio*256,256))
    elif height > width:
        im.thumbnail((256,height/width*256))
    
    # take the dimensions of resized image
    new_width, new_height = im.size 
   
    left = (new_width - 224)/2
    top = (new_height - 224)/2
    right = (new_width + 224)/2
    bottom = (new_height + 224)/2
    
    # crop
    im = im.crop((left, top, right, bottom))
    
    
    np_image = np.array(im)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    new_np_image = (np_image - mean)/std
    transposed = new_np_image.transpose((2,0,1))
    toFloatTensor = torch.FloatTensor([transposed])
    return toFloatTensor

def predict(image_path, model, topk=5):
    ''' Predict the class of an image using our trained deep learning model
    '''
    # use gpu if available
    model.to(device)
    
    # open image
    img = Image.open(image_path)
    
    # evaluation mode
    model.eval();
    
    # get image and move to device
    img = process_image(image_path)

    inputs = img.to(device)
    
    # fwd pass
    output = model.forward(inputs)
    
    # get probabilities from output
    probabilities = torch.exp(output).data  
    
    # get top k probs and index
    probs = torch.topk(probabilities, topk)[0].tolist()[0]
    index = torch.topk(probabilities, topk)[1].tolist()[0]
    
    # make list of labels using class_to_idx dict
    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])
        
    label = []
    for i in range(topk):
        label.append(ind[index[i]])
    
    return probs, label

# use function to get probabilities and labels and verify the expected output
probs, classes = predict(image_path, model, topk=top_k)


# using a pandas dataframe for the output of top k classes
df = pd.Series(probs, classes).reset_index()
df.columns = ['class', 'probability']

if not results.category_names:
    df['name'] = df['class']
else:
    df['name'] = df['class'].apply(lambda x: cat_to_name[x])

df.drop(columns=['class'], inplace=True)

df = df.set_index('name')

print(df)