# Summary

Welcome to my Git Repo which presents the undergraduate research I contributed to this summer. 
I worked alongside another undertgraduate researcher and two graduate researchers under the supervision of Dr. Yao. 
The research was conducted at the University of Tennessee, Knoxville as part of the Quantum Algorithms & Optimization REU. 
This was a 10 week program that took place from May 28 till August 2, 2024. 
I was tasked with developing quantum convolutional networks for AF (Atrial Fibrillation) detection from ECG signals. 
This allowed me to gain experience in quantum computing and deep learning concepts as well as experience in research.
I learned a lot at this program as well as how to work both individually on tasks I was assigned and in teams with other undergraduate and graduate researchers.
Throughout this program, I gained experience with PyTorch, PennyLane, and TensorFlow libraries. 
I also gained experience with Machine Learning concepts (both Classical and Quantum) and Quantum Computing. 

# Deep Learning Models for AF Detection from ECG Signals

<strong> Atrial fibrillation </strong> (AF) is the most common <strong> cardiac arrhythmia</strong>, a heart condition characterized
by uncoordinated electrical pulses in the heart. These pulses can be read through <strong>ECG</strong>
readings and a medical professional can diagnose AF by analyzing the ECG data. However,
the intermittent nature of atrial fibrillation makes it challenging to detect AF in short ECG
readings.

• Our goal is to develop a machine learning model that can accurately diagnose AF despite its
intermittent and random symptoms

• Our model also must overcome the challenge of noisy ECG signals

# Research & Methods
At the beginning of this REU program, I attended several tutorials and seminars that introduced me to the concepts of quantum computing. 
This helped give me background knowledge of the field as well as insight into the things I needed to learn for my specific project. 
During the time of the tutorials for the first 4 weeks, I focused on getting familiar with the research topic and on preprocessing the data to feed into a CNN.
After the tutorials for the first 4 weeks, I was tasked with creating a 1-dimensional Convolutional Neural Network for Binary Classification of AF from ECG signals.

### Data
To train and test the data we used a sample library of 8,528 “.mat” files containing ECG data
88% of the data is from patients with a normal heart rhythm
12% of patients diagnosed with AF 

### Pre-Processing
We first wrote a program to sort the entirety of the the .hea and .mat files into their respective classification folders according to a Reference chart (a .csv file) containing three classifications Normal, AF, and Other. 
The next step in our program is to structure the sorted data files into a database that we can feed into our CNN. 
Once the data is extracted into a readable database with the respective classifications we split the data into training and testing sets. We then further conduct a 80/20 split for training and validation respectively.
Finally, we optimized the hyper-parameters and fed the data into our CNN. 


### Why use 1D CNNs and Quantum Neural Networks

<strong>1D Convolutional Neural Networks</strong> are good at learning and extracting import features from the data set.
It can scan over the ECG signal data multiple times and automatically extract data.
Effective for identifying patterns such as Normal vs AF R to R intervals with different starting points in the data see

<strong>Quantum Neural Networks</strong> are known to produce effective predictive models with excellent
generalization performance even when provided with only a small amount of training data. Our
goal is to apply the best quantum convolutional layers and pooling layers to yield the best test
results. Below are different common <strong>Quantum Encoding</strong> methods we can use to transform our
classical data to quantum bits.

# Results
### Classical
With some guidance from my graduate student mentors my research partner and I were able to develop a 1D Convolutional Neural Network that yielded excellent results for the binary classification of the ECG data into AF and Normal classes. 
Here are our statistical results for the classical model: <br>

<strong>Accuracy: </strong> 0.9955 ± 0.0019 <br>
<strong>Precision: </strong> 0.9971 ± 0.0016 <br>
<strong>Recall: </strong> 0.9988 ± 0.0023 <br>
<strong>F1 Score:</strong> 0.9969 ± 0.0011 <br>
<strong>AUROC: </strong> 0.9998 ± 0.0001 <br>
<strong>AUPRC: </strong> 1.000 ± 0.0000 <br>
<br>
### Quantum
After several weeks of experimenting with the quantum neural network, I made decent progress. I was building on top of a former REU researcher and wrote a program to cycle through all the Quantum Neural Network configurations (adjusting preprocessing techniques, encoding methods, convolutional layer gates, and pooling gates), by doing this with the predefined gates and methods we could easily see which configurations yielded the best results.  

<strong>Accuracy: </strong> 0.8437 <br>
<strong>Precision: </strong> 0.8781 <br>
<strong>Recall: </strong> 0.9536 <br>
<strong>F1 Score: </strong>	0.9143 <br>
<strong>AUROC: </strong>	0.5179 <br>
<strong>AUPRC: </strong> 0.8779 <br>





