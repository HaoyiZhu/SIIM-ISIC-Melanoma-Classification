# SIIM-ISIC Melanoma Classification: Identify melanoma in lesion images
This is an competition on kaggle and is the big home work of the course: Practice of AI programming. The website of the competition is [here](https://www.kaggle.com/c/siim-isic-melanoma-classification)

## Requirements
* python 3.7
* pytorch 1.5.1
* torchvision 0.6.1
* pandas 1.0.5
* efficientnet_pytorch
* Windows
```Bash
# You can install efficientnet_pytorch by:
pip install efficientnet_pytorch
#Or:
git clone https://github.com/lukemelas/EfficientNet-PyTorch
cd EfficientNet-Pytorch
pip install -e
```

## Quick Start
* Get data: 

Download the data from [here](https://www.kaggle.com/c/siim-isic-melanoma-classification/data) and put them into [./data](./data)

* Training:

You can change the hyperparametrics such as lr, batch_size, begin_epoch, end_epoch, snapshop, etc. as you like in line 24 to line 30 in [train.py](train.py). 

Then simply run:
```bash
python train.py
```

* Testing:
If you want to use the model trained by yourself, just change the path in demo_inference.py.

Then simply run:
```bash
python demo_inference.py
```

## Results Saving
* Training:

After every snapshot, a model will be saved in [./exp](./exp)

* Testing:

The result will be saved in [./result](./result) in the format of csv.