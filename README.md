# AMSTE
PyTorch implementation of "Associative Memory with Spatio-Temporal Enhancement for Video Anomaly Detection"  by Yuanhong Zhong, Yongting Hu, Panliang Tang and Heng Wang.


## Dependencies
* Python 3.6
* PyTorch 1.1.0
* Numpy
* Sklearn

## Datasets
* USCD Ped2 [[dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)]
* CUHK Avenue [[dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)]
* ShanghaiTech [[dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)]

These datasets are from an official github of "Future Frame Prediction for Anomaly Detection - A New Baseline (CVPR 2018)".
Download the datasets into your_dataset_directory.

## Training
python train.py # for training
You can freely define parameters with your own settings like

## Evaluation
Test your own model
Check your dataset_type (ped2, Avenue or shanghai)
python evaluate.py   # for Evaluation

We also provide the pre-trained models and the labels of UCSD Ped2, Avenue and ShanghaiTech datasets at https://pan.baidu.com/s/1zcDMUZYfdo4jPI_i2dudIg?pwd=fmsi. To test these models, you need download and put them in weight folder.

