# Towards Efficient ECG-based Atrial Fibrillation Detection via Parameterised Hypercomplex Neural Networks

This is a Python and PyTorch code for the PH-CNN framework in our paper:

>L. Basso, Z. Ren and W. Nejdl, 
"Towards Efficient ECG-based Atrial Fibrillation Detection via Parameterised Hypercomplex Neural Networks", 2022.

### Abstract:

Atrial fibrillation (AF) is the most common cardiac arrhythmia and associated with 
a higher risk for serious conditions like stroke. Long-term recording of the 
electrocardiogram (ECG) with wearable devices embedded with an automatic and 
timely evaluation of AF helps to avoid life-threatening situations. 
However, the use of a deep neural network for auto-analysis of ECG on wearable devices 
is limited by its complexity. In this work, we propose lightweight convolutional 
neural networks (CNNs) for AF detection inspired by the recently proposed 
parameterised hypercomplex (PH) neural networks. Specifically, the convolutional 
and fully-connected layers of a real-valued CNN are replaced by PH convolutions 
and multiplications, respectively. PH layers are flexible to operate in any 
channel dimension n and able to capture inter-channel relations. 
We evaluate PH-CNNs on publicly available databases of dynamic and in-hospital 
ECG recordings and show comparable performance to corresponding real-valued 
CNNs while using approx. 1/n model parameters.

----------------

### Experiments on CPSC 2018 and 2021 data

CPSC 2018:
`python train_cpsc2018.py`

CPSC 2021:
`python train_cpsc2021.py`

Attributes for `train_config` and `model_config` can be adapted in these files or in `cfg.py` files.

| Parameter                | Attribute Name | Options                                        |
|--------------------------|----------------|------------------------------------------------|
| Real-valued or PHC model | model_name     | "cnn", "cnn_phc"                               |
| CNN module               | cnn_name       | "multi_scopic", "resnetNS", "densenet_vanilla" |
| Attention module         | attn_name      | "se", "none"                                   |
| Dimensionality           | n_leads        | CPSC 2021: 2, 4 <br/> CPSC 2018: 12            |

Proposed PH-CNN architecture:

![DCNN Architecture](/images/dcnn.png?raw=true "DCNN Architecture")

It includes three modules: (1) a CNN, (2) a squeeze-and excitation (SE) attention, and (3) a
multilayer perceptron (MLP) classifier. Compared to real-valued
DNNs, parameterised hypercomplex (PH) convolution and multiplication
replace real-valued convolutional and fully-connected (FC)
layers, respectively. We construct separate models for two tasks: (a)
AF detection, where every sampling point of the input ECG signal
gets classified as AF/non-AF, and (b) global abnormality classification,
where the output is a vector of class probabilities.

Implemented and tested PH-CNN backbones:
* [Multi-Scopic CNN](https://ieeexplore.ieee.org/abstract/document/9099511)
* [ResNet](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) with separable convolutions
* [DenseNet](https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html)

---
Code adapted from benchmark test from torch_ecg repository at https://github.com/DeepPSP/torch_ecg/tree/master/benchmarks

H. Wen and J. Kang, “torch ecg: An ECG deep learning framework implemented using PyTorch,” 2022. [Online].
Available: https://github.com/DeepPSP/torch_ecg

H. Wen and J. Kang, “A novel deep learning package for electrocardiography research,” Physiological Measurement, pp. 1–29, 2022.
