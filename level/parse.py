from __future__ import print_function
import numpy as np
import argparse
import math

parser = argparse.ArgumentParser(description='binned aggregation')
parser.add_argument('--level', default=1, type=int)

args = parser.parse_args()

def lvl4(datas):

	idx_ = np.load('./idx.npy')
	idx__ = np.load('./idx2.npy')
	idx___ = np.load('./idx3.npy')

	bin_ = np.empty([80, 40])
	bin_.fill(-1)

	lvl4 = []
	in_lvl2 = []
	idx = []
	i = 0

	for data in datas:

		bin_y = math.floor(data[1] * 80);
		bin_x = math.floor(data[0] * 40);

		if (bin_y == 80):
			bin_y = bin_y - 1
		if (bin_x == 40):
			bin_x = bin_x = 1

		if (bin_[bin_y][bin_x] == -1):
			bin_[bin_y][bin_x] = i;
			lvl4.append(data)
			idx.append(i)

			if i in idx_:
				in_lvl2.append(1)
			elif i in idx__:
				in_lvl2.append(2)
			elif i in idx___:
				in_lvl2.append(3)
			else:
				in_lvl2.append(4)

		i = i + 1

	lvl4 = np.array(lvl4)
	idx = np.array(idx)
	np.save('./idx4.npy', idx)
	idx = idx.reshape((idx.shape[0], 1))

	in_lvl2 = np.array(in_lvl2)
	in_lvl2 = in_lvl2.reshape((in_lvl2.shape[0], 1))

	type_ = ['%d'] * 2 + ['%.5f'] * 14 + ['%d'] * 2

	result = np.concatenate((idx, in_lvl2, lvl4), axis=1)
	np.savetxt("./lvl4.csv", result, header="ogi,vis,xpost,ypost,xposp,yposp,0,1,2,3,4,5,6,7,8,9,pred,target", comments='', delimiter=',', fmt=type_)

def lvl3(datas):

	idx_ = np.load('./idx.npy')
	idx__ = np.load('./idx2.npy')

	bin_ = np.empty([40, 40])
	bin_.fill(-1)

	lvl3 = []
	in_lvl2 = []
	idx = []
	i = 0

	for data in datas:

		bin_y = math.floor(data[1] * 40);
		bin_x = math.floor(data[0] * 40);

		if (bin_y == 40):
			bin_y = bin_y - 1
		if (bin_x == 40):
			bin_x = bin_x = 1

		if (bin_[bin_y][bin_x] == -1):
			bin_[bin_y][bin_x] = i;
			lvl3.append(data)
			idx.append(i)

			if i in idx_:
				in_lvl2.append(1)
			elif i in idx__:
				in_lvl2.append(2)
			else:
				in_lvl2.append(3)

		i = i + 1

	lvl3 = np.array(lvl3)
	idx = np.array(idx)
	np.save('./idx3.npy', idx)
	idx = idx.reshape((idx.shape[0], 1))

	in_lvl2 = np.array(in_lvl2)
	in_lvl2 = in_lvl2.reshape((in_lvl2.shape[0], 1))

	type_ = ['%d'] * 2 + ['%.5f'] * 14 + ['%d'] * 2

	result = np.concatenate((idx, in_lvl2, lvl3), axis=1)
	np.savetxt("./lvl3.csv", result, header="ogi,vis,xpost,ypost,xposp,yposp,0,1,2,3,4,5,6,7,8,9,pred,target", comments='', delimiter=',', fmt=type_)

def lvl2(datas):

	idx_ = np.load('./idx.npy')

	bin_ = np.empty([20, 20])
	bin_.fill(-1)

	lvl2 = []
	in_lvl1 = []
	idx = []
	i = 0

	for data in datas:

		bin_y = math.floor(data[1] * 20);
		bin_x = math.floor(data[0] * 20);

		if (bin_y == 20):
			bin_y = bin_y - 1
		if (bin_x == 20):
			bin_x = bin_x = 1

		if (bin_[bin_y][bin_x] == -1):
			bin_[bin_y][bin_x] = i;
			lvl2.append(data)
			idx.append(i)

			if i in idx_:
				in_lvl1.append(1)
			else:
				in_lvl1.append(2)

		i = i + 1

	lvl2 = np.array(lvl2)
	idx = np.array(idx)
	np.save('./idx2.npy', idx)
	idx = idx.reshape((idx.shape[0], 1))

	in_lvl1 = np.array(in_lvl1)
	in_lvl1 = in_lvl1.reshape((in_lvl1.shape[0], 1))

	type_ = ['%d'] * 2 + ['%.5f'] * 14 + ['%d'] * 2

	result = np.concatenate((idx, in_lvl1, lvl2), axis=1)
	np.savetxt("./lvl2.csv", result, header="ogi,vis,xpost,ypost,xposp,yposp,0,1,2,3,4,5,6,7,8,9,pred,target", comments='', delimiter=',', fmt=type_)

def lvl1(datas):

	bin_ = np.empty([10, 10])
	bin_.fill(-1)

	lvl1 = []
	idx = []
	i = 0

	for data in datas:

		bin_y = math.floor(data[1] * 10);
		bin_x = math.floor(data[0] * 10);

		if (bin_y == 10):
			bin_y = bin_y - 1
		if (bin_x == 10):
			bin_x = bin_x - 1

		if (bin_[bin_y][bin_x] == -1):
			bin_[bin_y][bin_x] = i;
			lvl1.append(data)
			idx.append(i)

		i = i + 1

	lvl1 = np.array(lvl1)
	idx = np.array(idx)
	np.save('./idx.npy', idx)
	idx = idx.reshape((idx.shape[0], 1))

	type_ = ['%d'] + ['%.5f'] * 14 + ['%d'] * 2

	result = np.concatenate((idx, lvl1), axis=1)
	np.savetxt("./lvl1.csv", result, header="ogi,xpost,ypost,xposp,yposp,0,1,2,3,4,5,6,7,8,9,pred,target", comments='', delimiter=',', fmt=type_)

def main():

	datas = np.genfromtxt('./data_all_vgg16_0.csv', dtype=float, delimiter=',', skip_header=1)

	if args.level == 1:
		lvl1(datas)
	elif args.level == 2:
		lvl2(datas)
	elif args.level == 3:
		lvl3(datas)
	else:
		lvl4(datas)

if __name__ == "__main__":
	main()