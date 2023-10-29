# SEResNet
# A modified ResNet with squeeze and excitation attention mechanism
Code accompanying the paper " Squeeze-and-excitation attention residual learning of propulsion fault features for diagnosing autonomous underwater vehicles" by Wenliao Du, Xinlong Yu, Zhen Guo, Hongchao Wang, Ziqiang Pu, Chuan Li (Ready to be submitted for publication).

Tensorflow 2.0 implementation
 Inspired by Wenliao Du et al.  [Efficient channel attention residual learning for the time-series fault diagnosis of wind turbine gearboxes] (https-443/77726476706e69737468656265737421f9f8518f24396d5e7d0dc7a597456d3a7479/article/10.1088/1361-6501/acf9c0/pdf), - The Squeeze and Excitation attention mechanism (SE) gains more effect on feature extraction compared to traditional CNN.
SE is combined with one-dimensional ResNet for the fault diagnosis of autonomous underwater vehicle.
This repository contains reprodce of several experiments mentioned in the paper.
# Requirements
python 3.11
Tensorflow == 2.6.2
Numpy == 1.19.2
Keras == 2.6.0
Note: All experiment were excecuted with NVIDIA GeForce GTX 1050Ti

# File discription
main-cnn: One-dimensional CNN for fault classification.
main-eca: One-dimensional CNN with ECA for fault classification.
main-se: One-dimensional CNN with SE for fault classification.
main-res-se: One-dimensional ResNet with SE for fault classification. (our proposed SEResNet).
main-resnet: One-dimensional ResNet for fault classification.
# Implementation details
Hyperparameter settings: Adam optimizer is used with learning rate of in both the generator and the discriminator;The batch size is , total iteration is 10,000. LABDA (Weight of cycle consistency loss) is . Random projection in SWD is .1e-4 32 10 32
# Usage
Note: Due to the copyright, no any data set is uploaded. For more detail pelase contact Authors.
# Ackonwledgements
This work was supported in part by the National Nature Science Foundation of China (52275138, 52175112), the Key R&D Projects in Henan Province (231111221100, 221111240200), the Science and Technology Major Project of Henan Province of China (221100220200), and the Key R&D Plan (Industrial) Project in Yancheng City (BE2023024).
