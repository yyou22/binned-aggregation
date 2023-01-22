# FGSM Attack on CIFAR10 Dataset with VGG Models

For our implementation, we use pretrained PyTorch models by [huyvnphan](https://github.com/huyvnphan/PyTorch_CIFAR10).

## Download Pretrained models
```python
python train.py --download_weights 1
```

## Sample a subset of CIFAR10 dataset
```python
python select_data.py
```
This will download the raw CIFAR10 testing data (10,000 entries) via PyTorch within the `cifar10` directory, and then randomly sample 10 images from each class (100 images in total) and save them as numpy arrays in `../data/`. 

`X.npy` includes all 100 images as normalized matrices, while `Y.npy` includes the ground truth labels as a numpy array.

## Conduct Attack on Data
```python
python fgsm.py --model [vgg16|vgg19] --epsilon [0.01|0.02|0.03]
```
This will create sub-folders within `../data/` and store the generated outputs. An example path would be `../data/vgg16/001/[generated files]`, where `vgg16` stands for the model chosen and `001` stands for perturbation size 0.01.

`adv_X.npy` stores the perturbed images as matrices, `confid_level.npy` stores the confidence levels across all classes in each prediction, `error.pckl` stores the robust error, `Y_hat.npy` stores the predicted labels, and `noise.npy` stores the applied perturbation normalized as images.

## Predict on Unperturbed Dataset
```python
python fgsm.py --model [vgg16|vgg19] --natural
```
If the flag `--natural` is present, it will ignore the argument passed to `--epsilon`.

`confid_level.npy` stores the confidence levels across all classes in each prediction, `error.pckl` stores the natural error. The outputs will be stored in `../data/[model name]/000/`

## Extract features of the VGG models
```python
python vgg_feature.py --model [vgg16|vgg19] --epsilon [0.01|0.02|0.03] [--natural]
```

This will save the features as `features.npy` in the corresponding folder.

## Apply t-SNE and Output CSV Files
```python
python dimen_reduc.py --model [vgg16|vgg19]
```

This will apply dimensionality reduction on the images and output it as a csv file `data.csv` in the corresponding path combined with confidence levels, prediction labels, and ground truth labels.