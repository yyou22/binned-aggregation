from __future__ import print_function
from cifar10_models.vgg import vgg16_bn, vgg19_bn
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision import transforms as T
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms,datasets
from numba import jit
import math
import time
import scipy.misc
import argparse
import os
import sys
import pickle
from PIL import Image
#from setup_mnist_model import MNIST
#from setup_cifar10_model import CIFAR10

from models.resnet import *
from cifar10_models.vgg import vgg16_bn, vgg19_bn
from cifar10_models.resnet import resnet18, resnet34

parser = argparse.ArgumentParser(description='FGSM Attack on CIFAR-10 with VGG Models')
parser.add_argument('--natural', action='store_true', help='natural prediction on the unperturbed dataset')
parser.add_argument('--epsilon', default=0.3, type=float, help='epsilon, the maximum amount of perturbation that can be applied')
parser.add_argument('--model', default='vgg19', help='[vgg16|vgg19|resnet|trades], model that is being attacked')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()

mean = [0.4914, 0.4822, 0.4465]
std = [0.2471, 0.2435, 0.2616]

tensor_ = transforms.ToTensor()

normalize = transforms.Normalize(mean, std)

inv_mean = [-0.4914/0.2471, -0.4822/0.2435, -0.4465/0.2616]
inv_std = [1/0.2471, 1/0.2435, 1/0.2616]

inv_normalize = transforms.Normalize(inv_mean, inv_std)

transform_ = transforms.Compose(
			[
				transforms.ToTensor(),
				transforms.Normalize(mean, std),
			]
		)

epsilon = 0.3
samples = 100

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

"""##L2 Black Box Attack"""

@jit(nopython=True)
def coordinate_ADAM(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, adam_epoch, up, down, step_size,beta1, beta2, proj):
	for i in range(batch_size):
		grad[i] = (losses[i*2+1] - losses[i*2+2]) / 0.0002 
	# ADAM update
	mt = mt_arr[indice]
	mt = beta1 * mt + (1 - beta1) * grad
	mt_arr[indice] = mt
	vt = vt_arr[indice]
	vt = beta2 * vt + (1 - beta2) * (grad * grad)
	vt_arr[indice] = vt
	epoch = adam_epoch[indice]
	corr = (np.sqrt(1 - np.power(beta2,epoch))) / (1 - np.power(beta1, epoch))
	m = real_modifier.reshape(-1)
	old_val = m[indice] 
	old_val -= step_size * corr * mt / (np.sqrt(vt) + 1e-8)
	# set it back to [-0.5, +0.5] region
	if proj:
		old_val = np.maximum(np.minimum(old_val, up[indice]), down[indice])
	m[indice] = old_val
	adam_epoch[indice] = epoch + 1

@jit(nopython=True)
def coordinate_Newton(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, adam_epoch, up, down, step_size, beta1, beta2, proj):
	cur_loss = losses[0]
	for i in range(batch_size):
		grad[i] = (losses[i*2+1] - losses[i*2+2]) / 0.0002 
		hess[i] = (losses[i*2+1] - 2 * cur_loss + losses[i*2+2]) / (0.0001 * 0.0001)
	hess[hess < 0] = 1.0
	hess[hess < 0.1] = 0.1
	m = real_modifier.reshape(-1)
	old_val = m[indice] 
	old_val -= step_size * grad / hess
	# set it back to [-0.5, +0.5] region
	if proj:
		old_val = np.maximum(np.minimum(old_val, up[indice]), down[indice])
	m[indice] = old_val

def loss_run(input,target,model,modifier,use_tanh,use_log,targeted,confidence,const):
	if use_tanh:
		pert_out = torch.tanh(input +modifier)/2
	else:
		pert_out = input + modifier

	pert_out_nor = normalize(pert_out + 0.5)

	output = model(pert_out_nor)
	
	if use_log:
		output = F.softmax(output,-1)
	
	if use_tanh:
		loss1 = torch.sum(torch.square(pert_out-torch.tanh(input)/2),dim=(1,2,3))
	else:
		loss1 = torch.sum(torch.square(pert_out-input),dim=(1,2,3))

	real = torch.sum(target*output,-1)
	other = torch.max((1-target)*output-(target*10000),-1)[0]
 
	if use_log:
		real=torch.log(real+1e-30)
		other=torch.log(other+1e-30)
	
	confidence = torch.tensor(confidence).type(torch.float64).cuda()
	
	if targeted:
		loss2 = torch.max(other-real,confidence)
	else:
		loss2 = torch.max(real-other,confidence)
	
	loss2 = const*loss2
	l2 = loss1
	loss = loss1 + loss2
	
	return loss.detach().cpu().numpy(), l2.detach().cpu().numpy(), loss2.detach().cpu().numpy(), output.detach().cpu().numpy(), pert_out.detach().cpu().numpy()

def l2_attack(cur_class, input, target, model, targeted, use_log, use_tanh, solver, reset_adam_after_found=True,abort_early=True,
							batch_size=128,max_iter=2000,const=0.01,confidence=0.0,early_stop_iters=200, binary_search_steps=10,
							step_size=0.005,adam_beta1=0.9,adam_beta2=0.999): #FIXME
	
	early_stop_iters = early_stop_iters if early_stop_iters != 0 else max_iter // 10

	input = torch.from_numpy(input).cuda()
	input = input.contiguous()
	target = torch.from_numpy(target).cuda()
	target = target
	
	var_len = input.view(-1).size()[0]
	modifier_up = np.zeros(var_len, dtype=np.float32)
	modifier_down = np.zeros(var_len, dtype=np.float32)
	real_modifier = torch.zeros(input.size(),dtype=torch.float32).cuda()
	mt = np.zeros(var_len, dtype=np.float32)
	vt = np.zeros(var_len, dtype=np.float32)
	adam_epoch = np.ones(var_len, dtype=np.int32)
	grad=np.zeros(batch_size,dtype=np.float32)
	hess=np.zeros(batch_size,dtype=np.float32)

	upper_bound=1e10
	lower_bound=0.0
	out_best_attack=input.clone().detach().cpu().numpy()
	out_best_const=const  
	out_bestl2=1e10
	out_bestscore=-1
	
	recent_attack=input.clone().detach().cpu().numpy()
	recent_score=-1

	if use_tanh:
		input = torch.atanh(input*1.99999)

	if not use_tanh:
		modifier_up = 0.5-input.clone().detach().view(-1).cpu().numpy()
		modifier_down = -0.5-input.clone().detach().view(-1).cpu().numpy()
	
	def compare(x,y):
		if not isinstance(x, (float, int, np.int64)):
			if targeted:
				x[y] -= confidence
			else:
				x[y] += confidence
			x = np.argmax(x)
		if targeted:
			return x == y
		else:
			return x != y

	for step in range(binary_search_steps):
		bestl2 = 1e10
		prev=1e6
		bestscore=-1
		last_loss2=1.0
		# reset ADAM status
		mt.fill(0)
		vt.fill(0)
		adam_epoch.fill(1)
		stage=0
		
		for iter in range(max_iter):
			if (iter+1)%100 == 0:
				loss, l2, loss2, _ , __ = loss_run(input,target,model,real_modifier,use_tanh,use_log,targeted,confidence,const)
				print("[STATS][L2] iter = {}, loss = {:.5f}, loss1 = {:.5f}, loss2 = {:.5f}".format(iter+1, loss[0], l2[0], loss2[0]))
				sys.stdout.flush()

			var_list = np.array(range(0, var_len), dtype = np.int32)
			indice = var_list[np.random.choice(var_list.size, batch_size, replace=False)]
			var = np.repeat(real_modifier.detach().cpu().numpy(), batch_size * 2 + 1, axis=0)
			for i in range(batch_size):
				var[i*2+1].reshape(-1)[indice[i]]+=0.0001
				var[i*2+2].reshape(-1)[indice[i]]-=0.0001
			var = torch.from_numpy(var)
			var = var.view((-1,)+input.size()[1:]).cuda()
			losses, l2s, losses2, scores, pert_images = loss_run(input,target,model,var,use_tanh,use_log,targeted,confidence,const) 
			real_modifier_numpy = real_modifier.clone().detach().cpu().numpy()
			if solver=="adam":
				coordinate_ADAM(losses,indice,grad,hess,batch_size,mt,vt,real_modifier_numpy,adam_epoch,modifier_up,modifier_down,step_size,adam_beta1,adam_beta2,proj=not use_tanh)
			if solver=="newton":
				coordinate_Newton(losses,indice,grad,hess,batch_size,mt,vt,real_modifier_numpy,adam_epoch,modifier_up,modifier_down,step_size,adam_beta1,adam_beta2,proj=not use_tanh)
			
			real_modifier = torch.from_numpy(real_modifier_numpy).cuda()

			#real_modifier = torch.clamp(real_modifier, -epsilon, epsilon)
			l2_norm = torch.sum(torch.square(real_modifier),dim=(1,2,3))
			#print(l2_norm)
			if l2_norm > epsilon:
				real_modifier = epsilon * real_modifier / l2_norm

			#print(real_modifier.shape)
			#clipping perturbation based on upper bound
			#m_norm = np.linalg.norm(real_modifier_numpy)
			#print('l2 norm: ' + str(m_norm))
			#if m_norm > max_perturbation:
				#real_modifier_numpy = (real_modifier_numpy / m_norm) * max_perturbation
				# Convert the adjusted numpy array back to torch tensor
				#real_modifier = torch.from_numpy(real_modifier_numpy).cuda()
				#print(real_modifier.shape) #1,3,32,32
				#print(real_modifier_numpy.shape) #1,3,32,32
			
			if losses2[0]==0.0 and last_loss2!=0.0 and stage==0:
				if reset_adam_after_found:
					mt.fill(0)
					vt.fill(0)
					adam_epoch.fill(1)
				stage=1 
			last_loss2=losses2[0]

			if abort_early and (iter+1) % early_stop_iters == 0:
				if losses[0] > prev*.9999:
						print("Early stopping because there is no improvement")
						break
				prev = losses[0]
		 
			if l2s[0] < bestl2 and compare(scores[0], np.argmax(target.cpu().numpy(),-1)):
				bestl2 = l2s[0]
				bestscore = np.argmax(scores[0])

			if l2s[0] < out_bestl2 and compare(scores[0],np.argmax(target.cpu().numpy(),-1)) and compare(scores[0],cur_class[0]):
				#print(np.argmax(scores[0]))
				#print(cur_class[0])
				if out_bestl2 == 1e10:
					print("[STATS][L3](First valid attack found!) iter = {}, loss = {:.5f}, loss1 = {:.5f}, loss2 = {:.5f}".format(iter+1, losses[0], l2s[0], losses2[0]))
					sys.stdout.flush()
				out_bestl2 = l2s[0]
				out_bestscore = np.argmax(scores[0])
				out_best_attack = pert_images[0]
				out_best_const = const

			recent_attack = pert_images[0]
			recent_score = np.argmax(scores[0])
	
		if compare(bestscore,  np.argmax(target.cpu().numpy(),-1)) and bestscore != -1:
			print('old constant: ', const)
			upper_bound = min(upper_bound,const)
			if upper_bound < 1e9:
					const = (lower_bound + upper_bound)/2
			print('new constant: ', const)
		else:
			print('old constant: ', const)
			lower_bound = max(lower_bound,const)
			if upper_bound < 1e9:
					const = (lower_bound + upper_bound)/2
			else:
					const *= 10
			print('new constant: ', const)
	
	if out_bestl2 == 1e10:
		return recent_attack, recent_score
	else:
		return out_best_attack, out_bestscore

def generate_data(test_loader,targeted,samples,start):
	inputs=[]
	labels=[]
	targets = []
	num_label=10
	cnt=0
	for i, data in enumerate(test_loader):
		if cnt<samples:
			if i>start:
				data, label = data[0],data[1]
				if targeted:
					seq = range(num_label)
					for j in seq:
						if j==label.item():
							continue
						inputs.append(data[0].numpy())
						targets.append(np.eye(num_label)[j])
						labels.append(label)
				else:
					inputs.append(data[0].numpy())
					targets.append(np.eye(num_label)[label.item()])
					labels.append(label)
				cnt+=1
			else:
				continue
		else:
			break

	inputs=np.array(inputs)
	targets=np.array(targets)
	labels = np.array(labels)

	print(inputs.shape)
	print(targets.shape)

	return inputs,targets,labels

def attack(inputs, targets, model, targeted, use_log, use_tanh, solver, device):
	adv_examples = []
	noise = []

	r = []

	print(len(inputs))
	print('go up to',len(inputs))
	# run 1 image at a time, minibatches used for gradient evaluation
	for i in range(len(inputs)):
		print('tick',i+1)

		input_ = np.array(np.expand_dims(inputs[i], axis=0), dtype=np.float32)
		cur_class = np.argmax(F.softmax(model(normalize(torch.from_numpy(input_).cuda() +0.5)),-1).detach().cpu().numpy(),-1)
		print(cur_class)

		attack,score=l2_attack(cur_class, np.expand_dims(inputs[i],0), np.expand_dims(targets[i],0), model, targeted, use_log, use_tanh, solver, device)
		attack = attack.reshape(3, 32, 32)
		print('attack shape')
		print(attack.shape)
		r.append(attack)
		print('r length')
		print(len(r))
		#append adv_examples
		adv_examples.append((attack+0.5).transpose(1 , 2 , 0))
		noise.append((attack-inputs[i]).transpose(1 , 2 , 0))
	r_ = np.array(r)
	print('r_ shape')
	print(r_.shape)
	return r_, adv_examples, noise

def attack_main(model, X_data, Y_data, epsilon_):
	np.random.seed(42)
	torch.manual_seed(42)

	epsilon = epsilon_

	#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
	# test_set = datasets.MNIST(root = './data', train=False, transform = transform, download=True)
	#test_set = datasets.CIFAR10(root = './data', train=False, transform = transform, download=True)
	#test_loader = torch.utils.data.DataLoader(test_set,batch_size=1,shuffle=True)

	use_cuda=True
	device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

	# model = MNIST().to(device)
	#model = CIFAR10().to(device) 

	#model = vgg16_bn(pretrained=True).to(device)

	# model.load_state_dict(torch.load('./models/mnist_model.pt'))
	# model.load_state_dict(torch.load('./models/cifar10_model.pt'))
	model.eval()

	use_log=True
	use_tanh=False #FIXME
	targeted=False
	#solver="newton"
	solver="adam"
	#start is a offset to start taking sample from test set
	#samples is the how many samples to take in total : for targeted, 1 means all 9 class target -> 9 total samples whereas for untargeted the original data 
	#sample is taken i.e. 1 sample only 
	#inputs, targets, labels = generate_data(test_loader,targeted,samples=samples,start=0)

	X_data = X_data.transpose(0, 3, 1, 2)
	X_data = (X_data - 0.5)

	inputs = X_data[:samples]
	labels = Y_data[:samples]
	targets = np.eye(10)[Y_data.flatten().astype(int)][:samples]

	print(inputs.shape)
	print(targets.shape)
	print(labels.shape)

	timestart = time.time()
	adv, adv_examples, noise = attack(inputs, targets, model, targeted, use_log, use_tanh, solver, device)
	print("adv shape")
	print(adv.shape)
	#adv = adv.reshape(samples, 3, 32, 32)
	#adv = inputs

	timeend = time.time()
	print("Took",(timeend-timestart)/60.0,"mins to run",len(inputs),"samples.")

	#for i in range(len(inputs)):
		#input_ = (inputs[i]+0.5).transpose(1, 2, 0)
		#data = transform_(input_).to(device)
		#data = data.unsqueeze(0)  # Add batch dimension
		#print(np.argmax(model(data).detach().cpu().numpy(), -1))

	if use_log:
		confid_level = F.softmax(model(normalize(torch.from_numpy(adv).cuda() +0.5)),-1).detach().cpu().numpy()
		valid_class = np.argmax(F.softmax(model(normalize(torch.from_numpy(inputs).cuda() +0.5)),-1).detach().cpu().numpy(),-1)
		adv_class = np.argmax(F.softmax(model(normalize(torch.from_numpy(adv).cuda() +0.5)),-1).detach().cpu().numpy(),-1)

	else:
		confid_level = F.softmax(model(normalize(torch.from_numpy(adv).cuda() +0.5)),-1).detach().cpu().numpy()
		valid_class = np.argmax(model(normalize(torch.from_numpy(inputs).cuda() +0.5)).detach().cpu().numpy(),-1)
		adv_class = np.argmax(model(normalize(torch.from_numpy(adv).cuda() +0.5)).detach().cpu().numpy(),-1)

	#print("confid level")
	#print(confid_level)
		
	acc = ((valid_class==adv_class).sum())/len(inputs)
	wrong = (valid_class==adv_class).sum()
	print("Targets: ", labels)
	print("Valid Classification: ", valid_class)
	print("Adversarial Classification: ", adv_class)
	print("Success Rate: ", (1.0-acc)*100.0)
	print("Total distortion: ", np.sum((adv-inputs)**2)**.5)
	
	#distortions = np.max(np.abs(adv - inputs), axis=(1,2,3))
	#print("Maximum distortions per image: ", distortions)

	distortions = np.sqrt(np.sum((adv - inputs)**2, axis=(1, 2, 3)))
	average_l2_norm = np.mean(distortions)
	print("Average distortion: ", average_l2_norm)


	path = '../data/ZOO/' + args.model + '/' + ''.join(str(epsilon).split('.'))

	if not os.path.exists(path):
		os.makedirs(path)

	adv_examples = np.array(adv_examples)
	confid_level = np.array(confid_level)
	noise = np.array(noise)

	np.save(path + '/adv_X.npy', adv_examples)
	np.save(path + '/Y_hat.npy', adv_class)
	np.save(path + '/confid_level.npy', confid_level)
	np.save(path + '/noise.npy', noise)

	print('adv_examples')
	print(adv_examples.shape)
	print('adv_class')
	print(adv_class.shape)
	print('noise')
	print(noise.shape)

	f = open(path + '/error.pckl', 'wb')
	pickle.dump(wrong, f)
	f.close()

	return

	#visualization of created cifar10 adv examples 
	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	cnt=0
	num_images = len(inputs) + len(adv) # total number of images
	num_rows = math.ceil(num_images / 5) # calculate number of rows based on 5 images per row

	plt.figure(figsize=(20, 4 * num_rows)) # adjust the figure size

	#display natural images
	for i in range(len(inputs)):
		cnt+=1
		plt.subplot(num_rows, 5, cnt) # replace 10,10 with num_rows,5
		plt.xticks([], [])
		plt.yticks([], [])
		plt.title("{}->{}".format(classes[valid_class[i]],classes[adv_class[i]]))
		plt.imshow(((inputs[i]+0.5)).transpose(1,2,0))

	#display adversarial images
	for i in range(len(adv)):
		cnt+=1
		plt.subplot(num_rows, 5, cnt) # replace 10,10 with num_rows,5
		plt.xticks([], [])
		plt.yticks([], [])
		plt.title("{}->{}".format(classes[valid_class[i]],classes[adv_class[i]]))
		plt.imshow(((adv[i]+0.5)).transpose(1,2,0))
		
	plt.tight_layout()
	plt.show()
	if targeted:
		if solver=="newton":
			plt.savefig('newton_targeted_cifar10.png')
		else:
			plt.savefig('adam_targeted_cifar10.png') 
	else:
		if solver=="newton":
			plt.savefig('newton_untargeted_cifar10.png')
		else:
			plt.savefig('adam_untargeted_cifar10.png') 

def main():

	if args.model == "vgg16":
		model = vgg16_bn(pretrained=True).to(device)
	elif args.model == "vgg19":
		print("vgg19")
		model = vgg19_bn(pretrained=True).to(device)
	if args.model == "resnet":
		model = resnet34(pretrained=True).to(device)
	elif args.model == "TRADES":
		model = ResNet34().to(device)
		model.load_state_dict(torch.load("./resnet/model-advres-epoch200.pt", map_location=torch.device('cpu')))

	X_data = np.load("/content/data/ZOO/X.npy").astype(np.float32)
	Y_data = np.load("/content/data/ZOO/Y.npy").astype(np.float32)

	attack_main(model, X_data, Y_data, epsilon)

if __name__ == "__main__":
	main()

	