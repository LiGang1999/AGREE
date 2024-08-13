# Federated Domain Adaptation via Pseudo-label Refinement (Accepted at ICME 2023)

Here is the official implementation of the method in our paper [Federated Domain Adaptation via Pseudo-label Refinement](https://doi.org/10.1109/ICME55011.2023.00314).

## Setup

### Install Package Dependencies (Python 3.9.12)

```
pip install -r requirements.txt
```

### Install Datasets

We need users to declare a `base path` to store the dataset as well as the log of training procedure. The directory structure should be
```
base_path
│       
└───dataset
│   │   DigitFive
│       │   mnist_data.mat
│       │   mnistm_with_label.mat
|       |   svhn_train_32x32.mat  
│       │   ...
│   │   DomainNet
│       │   ...
│   │   OfficeCaltech10
│       │   ...
└───trained_model_1
│   │	parmater
│   │	runs
└───trained_model_2
│   │	parmater
│   │	runs
...
└───trained_model_n
│   │	parmater
│   │	runs    
```
Our framework now support three multi-source domain adaptation datasets: ```DigitFive, DomainNet and OfficeCaltech10```.

## Unsupervised Federated Domain Adaptation

The configuration files can be found under the folder  `./config`, and we provide three config files with the format `.yaml`. To perform the training on the specific dataset (e.g., DomainNet), please use the following command:

```python
CUDA_VISIBLE_DEVICES=0 python main.py --config DomainNet.yaml --target-domain clipart -dp > log_clipart.txt
```

## Reference

If you find this useful in your work, please consider citing our paper.

```
@inproceedings{li2023federated,
  title={Federated Domain Adaptation via Pseudo-label Refinement},
  author={Li, Gang and Zhang, Qifei and Wang, Peizheng and Zhang, Jie and Wu, Chao},
  booktitle={2023 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={1829--1834},
  year={2023},
  organization={IEEE}
}
```

## Acknowledgements

The implementation code is adapted from [KD3A](https://github.com/FengHZ/KD3A).
