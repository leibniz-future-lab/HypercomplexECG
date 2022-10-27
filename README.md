# Towards Efficient ECG-based Atrial Fibrillation Detection via Parameterised Hypercomplex Neural Networks

This is a Python and PyTorch code for the PH-CNN framework in our paper:

L. Basso, Z. Ren and W. Nejdl, 
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

### Tests on CPSC 2018 and 2021 data

CPSC 2018:
`python train_cpsc2018.py`

CPSC 2021:
`python train_cpsc2021.py`

Parameters can be adapted in `cfg.py` files.

---
Code adapted from benchmark test from torch_ecg repository at https://github.com/DeepPSP/torch_ecg/tree/master/benchmarks

H. Wen and J. Kang, “torch ecg: An ECG deep learning framework implemented using PyTorch,” 2022. [Online].
Available: https://github.com/DeepPSP/torch_ecg

H. Wen and J. Kang, “A novel deep learning package for electrocardiography research,” Physiological Measurement, pp. 1–29, 2022.
