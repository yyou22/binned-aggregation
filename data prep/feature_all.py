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

parser = argparse.ArgumentParser(description='VGG Extracted Features on adversarial CIFAR10 dataset')
parser.add_argument('--natural', action='store_true', help='natural prediction on the unperturbed dataset')
parser.add_argument('--epsilon', default=0.03, type=float, help='epsilon, the maximum amount of perturbation that can be applied')
parser.add_argument('--model', default='vgg16', help='[vgg16|vgg19], model that is being attacked')

args = parser.parse_args()

mean = [0.4914, 0.4822, 0.4465]
std = [0.2471, 0.2435, 0.2616]

inv_mean = [-0.4914/0.2471, -0.4822/0.2435, -0.4465/0.2616]
inv_std = [1/0.2471, 1/0.2435, 1/0.2616]

epsilon = args.epsilon

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])
testset = torchvision.datasets.CIFAR10(root='/content/data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def feature_(model, test_loader):

	model.eval()

	normalize = T.Normalize(mean, std)

	for data, target in test_loader:

		data, target = data.to(device), target.to(device)

		data = normalize(data)

		X, y = Variable(data, requires_grad = True), Variable(target)

		# output of model
		out = model(X)

		features.append(out.cpu().detach().numpy())

	if args.natural:
		path = '../data/' + args.model + '/000'
	else:
		path = '../data/' + args.model + '/' + ''.join(str(args.epsilon).split('.'))

	if not os.path.exists(path):
		os.makedirs(path)

	features = np.array(features)

	np.save(path + '/features.npy', features)

def main():

	if args.model == "vgg16":
		model = vgg16_bn(pretrained=True).features.to(device)
	elif args.model == "vgg19":
		model = vgg19_bn(pretrained=True).features.to(device)
	elif args.model == "resnet":
		model = resnet34(pretrained=True).to(device)
		model = FeatureExtractor(model)
	elif args.model == "TRADES":
		model = ResNet34().to(device)
		model.load_state_dict(torch.load("./resnet/model-advres-epoch200.pt", map_location=torch.device('cpu')))
		model = FeatureExtractor(model)

	feature_(model, test_loader)

if __name__ == "__main__":
	main()

