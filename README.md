# AI Application - Image Classifier

Project code for Udacity's "AI Programming with Python" Nanodegree program.

Part 1/2: Develop code for an image classifier built with PyTorch (Jupyter Notebook)

Part 2/2: Convert the code into a command line application.

# Data

The data is not included in this repo, but can be downloaded here:
https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz

The dataset is split into three parts, training, validation, and testing.

# Running the Application

1. Train

Train a new network on a data set with train.py

Basic usage: python train.py data_directory

Prints out training loss, validation loss, and validation accuracy as the network trains

Options:
- Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
- Choose architecture: python train.py data_dir --arch "densenet161"
- Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 3
- Use GPU for training: python train.py data_dir --gpu

For details please run: python train.py --help

2. Predict

Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint

Options:
- Return top KK most likely classes: python predict.py input checkpoint --top_k 5
- Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
- Use GPU for inference: python predict.py input checkpoint --gpu

For details please run: python predict.py --help
