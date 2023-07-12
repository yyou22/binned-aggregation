from __future__ import print_function
from cifar10_models.vgg import vgg16_bn, vgg19_bn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision import transforms as T
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pickle
from zoo_attack import attack_main

parser = argparse.ArgumentParser(description='FGSM Attack on CIFAR-10 with VGG Models')
parser.add_argument('--natural', action='store_true', help='natural prediction on the unperturbed dataset')
parser.add_argument('--epsilon', default=0.3, type=float, help='epsilon, the maximum amount of perturbation that can be applied')
parser.add_argument('--model', default='vgg16', help='[vgg16|vgg19|resnet|trades], model that is being attacked')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()

start_idx = 0

mean = [0.4914, 0.4822, 0.4465]
std = [0.2471, 0.2435, 0.2616]

inv_mean = [-0.4914/0.2471, -0.4822/0.2435, -0.4465/0.2616]
inv_std = [1/0.2471, 1/0.2435, 1/0.2616]

epsilon = args.epsilon

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

def main():

	if args.model == "vgg16":
		model = vgg16_bn(pretrained=True).to(device)
	elif args.model == "vgg19":
		model = vgg19_bn(pretrained=True).to(device)
	if args.model == "resnet":
		model = resnet34(pretrained=True).to(device)
	elif args.model == "TRADES":
		model = ResNet34().to(device)
		model.load_state_dict(torch.load("./resnet/model-advres-epoch200.pt", map_location=torch.device('cpu')))

	X_data = np.load("/content/data/ZOO/X.npy")
	Y_data = np.load("/content/data/ZOO/Y.npy")

	if args.natural:
		natural(model, X_data, Y_data, epsilon)
	else:
		attack_main(model, X_data, Y_data, epsilon)

if __name__ == "__main__":
	main()