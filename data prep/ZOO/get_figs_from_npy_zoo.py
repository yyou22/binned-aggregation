from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import shutil

label_str = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def normalize_noise(parent1="./data/vgg16/01/", parent3="./data/vgg16/03/", parent5="./data/vgg16/05/"):

	noise1 = np.load(os.path.join(parent1, 'noise.npy'))
	noise3 = np.load(os.path.join(parent3, 'noise.npy'))
	noise5 = np.load(os.path.join(parent5, 'noise.npy'))

	all_data = np.concatenate((noise1, noise3, noise5))

	return all_data

def generate_data_and_noise_png(data_file_name="X.npy", parent="../data/ZOO/", noise=True, noise_array=None):

	data = np.load(os.path.join(parent, data_file_name))
	data[data > 1] = 1
	data[data < 0] = 0

	if noise:
		global noise_data
		noise_data = np.load(os.path.join(parent, "noise.npy"))#.transpose(0,2,3,1)
		#min_val = np.min(noise_data)
		#max_val = np.max(noise_data)

		min_val = np.min(noise_array)
		max_val = np.max(noise_array)

		# Subtract min_val from all elements and divide by the range of values
		noise_data = (noise_data - min_val) / (max_val - min_val)
		noise_data[noise_data > 1] = 1
		noise_data[noise_data < 0] = 0

	parent = os.path.join(parent, "img_data/")
	shutil.rmtree(parent, ignore_errors=True)
	os.mkdir(parent)

	for i in range(data.shape[0]):
		cur_dir = os.path.join(parent, str(i))
		shutil.rmtree(cur_dir, ignore_errors=True)
		os.mkdir(cur_dir)
		plt.imsave(os.path.join(cur_dir, "img.png"), data[i], format="png")
		if noise:
			plt.imsave(os.path.join(cur_dir, "noise.png"), noise_data[i], format="png")

def generate_label_txt(label_file_name="Y.npy", parent="../data/ZOO/"):
	labels = np.load(os.path.join(parent, label_file_name))
	parent = os.path.join(parent, "img_data/")
	os.makedirs(parent, exist_ok=True)

	for i in range(labels.size):
		cur_dir = os.path.join(parent, str(i))
		os.makedirs(cur_dir, exist_ok=True)
		f = open(os.path.join(cur_dir, "label.txt"), "w")
		f.write(label_str[labels[i]])
		f.close()

adv_data = "adv_X.npy"
adv_label = "Y_hat.npy"

generate_data_and_noise_png(noise = False, parent = "./data/")
generate_label_txt(parent = "./data/")

noise_vgg16 = normalize_noise(parent1="./data/vgg16/01/", parent3="./data/vgg16/03/", parent5="./data/vgg16/05/")
noise_vgg19 = normalize_noise(parent1="./data/vgg19/01/", parent3="./data/vgg19/03/", parent5="./data/vgg19/05/")
noise_resnet = normalize_noise(parent1="./data/resnet/01/", parent3="./data/resnet/03/", parent5="./data/resnet/05/")
noise_TRADES = normalize_noise(parent1="./data/TRADES/01/", parent3="./data/TRADES/03/", parent5="./data/TRADES/05/")

all_noise = np.concatenate((noise_vgg16, noise_vgg19, noise_resnet, noise_TRADES))

generate_label_txt(label_file_name = adv_label, parent = "./data/vgg16/00/")
generate_data_and_noise_png(data_file_name = adv_data, parent = "./data/vgg16/01/", noise_array=all_noise)
generate_label_txt(label_file_name = adv_label, parent = "./data/vgg16/01/")
generate_data_and_noise_png(data_file_name = adv_data, parent = "./data/vgg16/03/", noise_array=all_noise)
generate_label_txt(label_file_name = adv_label, parent = "./data/vgg16/03/")
generate_data_and_noise_png(data_file_name = adv_data, parent = "./data/vgg16/05/", noise_array=all_noise)
generate_label_txt(label_file_name = adv_label, parent = "./data/vgg16/05/")

generate_label_txt(label_file_name = adv_label, parent = "./data/vgg19/00/")
generate_data_and_noise_png(data_file_name = adv_data, parent = "./data/vgg19/01/", noise_array=all_noise)
generate_label_txt(label_file_name = adv_label, parent = "./data/vgg19/01/")
generate_data_and_noise_png(data_file_name = adv_data, parent = "./data/vgg19/03/", noise_array=all_noise)
generate_label_txt(label_file_name = adv_label, parent = "./data/vgg19/03/")
generate_data_and_noise_png(data_file_name = adv_data, parent = "./data/vgg19/05/", noise_array=all_noise)
generate_label_txt(label_file_name = adv_label, parent = "./data/vgg19/05/")

generate_label_txt(label_file_name = adv_label, parent = "./data/resnet/00/")
generate_data_and_noise_png(data_file_name = adv_data, parent = "./data/resnet/01/", noise_array=all_noise)
generate_label_txt(label_file_name = adv_label, parent = "./data/resnet/01/")
generate_data_and_noise_png(data_file_name = adv_data, parent = "./data/resnet/03/", noise_array=all_noise)
generate_label_txt(label_file_name = adv_label, parent = "./data/resnet/03/")
generate_data_and_noise_png(data_file_name = adv_data, parent = "./data/resnet/05/", noise_array=all_noise)
generate_label_txt(label_file_name = adv_label, parent = "./data/resnet/05/")

generate_label_txt(label_file_name = adv_label, parent = "./data/TRADES/00/")
generate_data_and_noise_png(data_file_name = adv_data, parent = "./data/TRADES/01/", noise_array=all_noise)
generate_label_txt(label_file_name = adv_label, parent = "./data/TRADES/01/")
generate_data_and_noise_png(data_file_name = adv_data, parent = "./data/TRADES/03/", noise_array=all_noise)
generate_label_txt(label_file_name = adv_label, parent = "./data/TRADES/03/")
generate_data_and_noise_png(data_file_name = adv_data, parent = "./data/TRADES/05/", noise_array=all_noise)
generate_label_txt(label_file_name = adv_label, parent = "./data/TRADES/05/")

