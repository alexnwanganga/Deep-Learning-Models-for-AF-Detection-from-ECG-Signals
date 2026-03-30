# Overview

This repository presents my undergraduate research conducted during the Quantum Algorithms & Optimization REU at the University of Tennessee, Knoxville (May–August 2024).
I collaborated with undergraduate and graduate researchers under faculty supervision to develop classical and quantum machine learning models for detecting atrial fibrillation (AF) from ECG signals. The research was conducted at the University of Tennessee, Knoxville, as part of the Quantum Algorithms & Optimization REU, a 10-week program that took place from May 28 to August 2, 2024. 

This project combines:

- Deep learning (1D CNNs)
- Quantum machine learning (QCNNs)
- Signal processing for biomedical data 

# Problem Statement

<strong> Atrial fibrillation </strong> (AF) is the most common <strong> cardiac arrhythmia</strong>, a heart condition characterized
by uncoordinated electrical pulses in the heart. These pulses can be read through <strong>ECG</strong>
readings, and a medical professional can diagnose AF by analyzing the ECG data. However,
the intermittent nature of atrial fibrillation makes it challenging to detect AF in short ECG
readings.

Key challenges:

- Short ECG segments may not clearly exhibit AF
- ECG signals are often noisy and high-dimensional
- Class imbalance (majority normal vs minority AF)

Goal:
Develop robust models capable of accurately detecting AF from short, noisy ECG recordings.

# Research & Methods
### 1. Data Engineering Pipeline
- Processed 8,528 ECG recordings (.mat + .hea files)
- Built a full preprocessing pipeline:
  - Automated file parsing using wfdb
  - Label extraction from reference CSV
  - Class-based dataset restructuring
- Constructed a unified dataset:
  - Binary classification (AF vs Normal)
  - 88% Normal / 12% AF

### 2. Classical Deep Learning Model (1D CNN)

Designed and optimized a 1D Convolutional Neural Network for time-series ECG classification:

<strong> Key features: </strong>

- Multi-layer Conv1D architecture with pooling
- Adaptive global average pooling
- Dropout regularization
- Bayesian hyperparameter optimization (skopt)
- GPU acceleration (CUDA / Apple MPS)

<strong> Why 1D CNNs? </strong>

- Automatically extracts temporal features (e.g., R–R intervals)
- Robust to shifts in signal alignment
- Well-suited for raw waveform data


### 3. Quantum Machine Learning (QCNN)

Implemented a flexible Quantum Convolutional Neural Network framework using PennyLane:

<strong> Key contributions: </strong>

- Modular QCNN architecture:
  - Multiple quantum convolution layer designs (11 variants)
  - Multiple pooling strategies
- Explored encoding strategies:
  - Angle embedding
  - Amplitude embedding
  - FRQI encoding
- Developed an automated configuration search pipeline:
  - Iterates across preprocessing + encoding + circuit designs
- Integrated classical preprocessing:
  - PCA, autoencoders, spectral embedding, compressive sensing

<strong> Why Explore Quantum Neural Networks? </strong>

<strong>Quantum Neural Networks</strong> are known to produce effective predictive models with excellent
generalization performance even when provided with only a small amount of training data.



# Results
### Classical
Here are statistical results for the classical model: <br>

- <strong>Accuracy: </strong> 0.9955 ± 0.0019 <br>
- <strong>Precision: </strong> 0.9971 ± 0.0016 <br>
- <strong>Recall: </strong> 0.9988 ± 0.0023 <br>
- <strong>F1 Score:</strong> 0.9969 ± 0.0011 <br>
- <strong>AUROC: </strong> 0.9998 ± 0.0001 <br>
- <strong>AUPRC: </strong> 1.000 ± 0.0000 <br>


Achieved near-perfect classification on ECG data.


### Quantum
Here are statistical results for the quantum model: <br>

- <strong>Accuracy: </strong> 0.8437 <br>
- <strong>Precision: </strong> 0.8781 <br>
- <strong>Recall: </strong> 0.9536 <br>
- <strong>F1 Score: </strong>	0.9143 <br>
- <strong>AUROC: </strong>	0.5179 <br>
- <strong>AUPRC: </strong> 0.8779 <br>

<br>

Demonstrated feasibility of QCNNs, though limited by:

- Qubit constraints
- Encoding bottlenecks
- Noise and simulation overhead


# Technical Skills
- <strong>Languages:</strong> Python
- <strong>Deep Learning:</strong> PyTorch, TensorFlow
- <strong>Quantum ML:</strong> PennyLane
- <strong>Data Processing:</strong> NumPy, Pandas, SciPy, WFDB
- <strong>Optimization:</strong> Scikit-Optimize (Bayesian optimization)
- <strong>Visualization:</strong> Matplotlib

# Key Contributions

- Built an end-to-end ECG classification pipeline from raw waveform data
- Designed and tuned a high-performance 1D CNN model
- Developed a configurable QCNN experimentation framework
- Explored hybrid classical–quantum workflows
- Automated large-scale model experimentation across architectures


