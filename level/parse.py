from __future__ import print_function
import numpy as np
import argparse
import math

parser = argparse.ArgumentParser(description='binned aggregation')
parser.add_argument('--level', default=1, type=int)

args = parser.parse_args()

def lvl2(datas):

	bin_ = np.empty([40, 40])
	bin_.fill(-1)

	lvl2 = []
	idx = []
	i = 0

	for data in datas:

		bin_y = math.floor(data[1] * 40);
		bin_x = math.floor(data[0] * 40);

		if (bin_y == 40):
			bin_y = 39
		if (bin_x == 40):
			bin_x = 39

		if (bin_[bin_y][bin_x] == -1):
			bin_[bin_y][bin_x] = i;
			lvl2.append(data)
			idx.append(i)

		i = i + 1

	lvl2 = np.array(lvl2)
	idx = np.array(idx)
	idx = idx.reshape((idx.shape[0], 1))

	type_ = ['%d'] + ['%.5f'] * 14 + ['%d'] * 2

	result = np.concatenate((idx, lvl2), axis=1)
	np.savetxt("./lvl2.csv", result, header="ogi,xpost,ypost,xposp,yposp,0,1,2,3,4,5,6,7,8,9,pred,target", comments='', delimiter=',', fmt=type_)

def lvl1(datas):

	bin_ = np.empty([20, 20])
	bin_.fill(-1)

	lvl1 = []
	idx = []
	i = 0

	for data in datas:

		bin_y = math.floor(data[1] * 20);
		bin_x = math.floor(data[0] * 20);

		if (bin_y == 20):
			bin_y = 19
		if (bin_x == 20):
			bin_x = 19

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
	else:
		lvl2(datas)

if __name__ == "__main__":
	main()