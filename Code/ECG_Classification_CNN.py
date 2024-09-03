#--------------------------Waveform Database Libraries----------------------------------------------------------

import pandas as pd
import scipy.io
import os
import wfdb
import random

#---------------------------CNN Libraries---------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, TensorDataset, random_split

#---------------------------Hyper-parameter Tuniung Libraries---------------------------------------------------------

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

#--------------------------Read and Combine Funtions----------------------------------------------------------

def read_mat_files(folder_path, label):
    data_list = [] 
    label_list = []

    #Loads all the mat files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".mat"):
            file_path = os.path.join(folder_path, filename)
            mat_data = scipy.io.loadmat(file_path)
            
            data_list.append(mat_data)
            label_list.append(label)

    return data_list, label_list

def combine_data(folder_paths):
    all_data = []
    all_labels = []
    signals_array = []

    #
    for label, folder_path in enumerate(folder_paths):
        data_list, labels = read_mat_files(folder_path, label)
        all_data.extend(data_list)
        all_labels.extend(labels)

        # Process each .mat file in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".mat"):
                file_path = os.path.join(folder_path, filename)
                record = wfdb.rdrecord(file_path.replace(".mat", ""))
                
                for idx, signal in enumerate(record.p_signal.T):
                    signals_array.append(signal)

    # Convert to DataFrame
    dataFrame = pd.DataFrame(signals_array)
    labelSeries = pd.Series(all_labels, name='label')

    return dataFrame, labelSeries
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#---------------------------------Defining Database Structure---------------------------------------------------

# Determines path to each class folder
folder_paths = [
    '/Users/mateo/Repos/Deep-Learning-of-ECG-signals/Data/Class_A',
    '/Users/mateo/Repos/Deep-Learning-of-ECG-signals/Data/Class_N'  #Removed Folder O Since we want binary classification
]

# Combine data
dataFrame, labelSeries = combine_data(folder_paths)
 
# Verify that DataFrame and Series are created properly
print(f"DataFrame shape: {dataFrame.shape}")
print(f"Label Series shape: {labelSeries.shape}")

# Combine DataFrame and Series
dataFrame = pd.concat([labelSeries, dataFrame], axis=1)

# Randomize the rows in the DataFrame
dataFrame = dataFrame.sample(frac=1, random_state=43).reset_index(drop=True)

# Verify the combined DataFrame
print(f"Combined and randomized DataFrame shape: {dataFrame.shape}")

# Save to CSV
try:
    dataFrame.to_csv('combined_ECG_Data.csv', index=False)
    print("CSV file created successfully.")
except Exception as e:
    print(f"Error saving CSV file: {e}")


#---------------------------------Designating Training and Testing Sets---------------------------------------------------

# --- Train and Test split manually (test with patient 233 and 234 ECG windows) ---
# --- Dataset: Previously sectioned ECG recordings into 2-second (360 Hz) windows ---

train = dataFrame.iloc[0:4000] 
test = dataFrame.iloc[4001:]

sub_timewindow = 960

# Print the shape of train to understand its dimensions
print("Shape of train DataFrame:", train.shape)

X_train = train.iloc[:, 0:sub_timewindow].values  # voltages, train
X_test = test.iloc[:, 0:sub_timewindow].values    # voltages, test
Y_train = train['label'].values      # results, train
Y_test = test['label'].values        # results, test

print('Train Shape - voltages, label:')
print(X_train.shape, Y_train.shape)
print('Test Shape - voltages, label:')
print(X_test.shape, Y_test.shape)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(2)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(2)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

# --- Create training and validation sets ---

# Set a seed value
seed = 43
set_seed(seed)

# Split the training set into training and validation
val_split = 0.2
train_size = int((1 - val_split) * len(X_train_tensor))
val_size = len(X_train_tensor) - train_size
train_dataset, val_dataset = random_split(TensorDataset(X_train_tensor, Y_train_tensor), [train_size, val_size])

# DataLoaders for training, validation, and test setsr
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_tensor, Y_test_tensor), batch_size=batch_size, shuffle=False)

#---------------------------------Defining Model---------------------------------------------------

# Define the model architecture in PyTorch
class ECGModel(nn.Module):
    def __init__(self, num_filters, kernel_size, dropout_rate): # Allows for inout of hyperparameter changes
        super(ECGModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters*2, kernel_size=kernel_size)
        self.pool = nn.MaxPool1d(3)
        self.conv3 = nn.Conv1d(in_channels=num_filters*2, out_channels=num_filters*4, kernel_size=kernel_size)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Equivalent to GlobalAveragePooling1D
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(num_filters*4, 1)
    
    def forward(self, x):
        x = x.view(x.size(0), 1, -1)  # Reshape to (batch_size, channels, sequence_length)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.global_avg_pool(x).squeeze(-1)
        x = torch.sigmoid(self.fc(x))
        return x
    
#---------------------------------Define Space---------------------------------------------------

space = [
    Integer(8, 128, name='num_filters'),
    Integer(3, 7, name='kernel_size'),
    Real(1e-5, 1e-2, "log-uniform", name='learning_rate'),
    Real(0.1, 0.5, name='dropout_rate'),
    Integer(8, 128, name='batch_size')
]

#---------------------------------Determining Hyperparameters---------------------------------------------------

# Define the objective function
@use_named_args(space)
def objective(**params):
    num_filters = params['num_filters']
    kernel_size = params['kernel_size']
    learning_rate = params['learning_rate']
    dropout_rate = params['dropout_rate']
    batch_size = params['batch_size']

    # Print parameters for debugging
    print(f'Params: {params}')

    # Check if batch_size is valid
    if not isinstance(batch_size, int) or batch_size <= 0:
        return -1e10

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = ECGModel(num_filters, kernel_size, dropout_rate).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / len(train_loader)

    # Validation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.squeeze().cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)

    # Check for large values
    if np.isnan(auroc) or np.isnan(auprc) or np.isinf(auroc) or np.isinf(auprc):
        return -1e10  # Assign a large negative value if nan or inf is encountered
    print('AUROC: ', auroc)
    print('AUPRC: ', auprc)
    # Return the negative value since we want to maximize these scores
    return -(auroc + auprc)

# Device configuration

# Check for MPS availability 
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Perform Bayesian Optimization
res = gp_minimize(objective, space, n_calls=50, random_state=42)

#---------------------------------Training and Validation Loop---------------------------------------------------
# Best parameters
best_params = res.x
print(f'Best parameters: {best_params}')

# Use the best parameters to retrain the model on the entire training set
num_filters = best_params[0]
kernel_size = best_params[1]
learning_rate = best_params[2]
dropout_rate = best_params[3]
batch_size = int(best_params[4])

# Create data loaders with the best batch size
train_loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_tensor, Y_test_tensor), batch_size=batch_size, shuffle=False)


# Model instance
model = ECGModel(num_filters, kernel_size, dropout_rate).to(device)
print(model)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Epochs was 100
epochs = 100
patience = 30 # Number of ecpoch program will wait for before stopping
best_val_loss = float('inf')
epochs_of_no_improve = 0
early_stop = False 

# Check for MPS availability 
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

train_losses, val_losses = [], []

for epoch in range(epochs):
    # Training Phase
    model.train()
    running_train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
    
    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation Phase
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            running_val_loss += loss.item()
    
    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    print(f'Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

    # Determine whether or not to implement an Early Stop
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
    
    if epochs_no_improve >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        early_stop = True
        break

 # Message to show how many epochs were completed before stopping   
if not early_stop:
    print(f'Training completed after {epochs} epochs')


# ---------------------------------- Plot Training and Validation Loss ----------------------------------

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(train_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs',fontsize = 14)
plt.ylabel('Loss', fontsize = 14)
#plt.axvline(x=40, color='r', linestyle='--', alpha=0.5)

"""
plt.annotate('Underfitting',
             fontsize=16,
             color='red',
             xy=(5, 0.3),        # Point to annotate
             xytext=(10, 0.4),   # Text location
             arrowprops=dict(facecolor='red', shrink=0.05))

plt.annotate('Overfitting',
             fontsize=16,
             color='red',
             xy=(85, 0.15),      
             xytext=(75, 0.3), 
             arrowprops=dict(facecolor='red', shrink=0.05))
             
             """

plt.title('Training and Validation Loss', fontsize = 20)
plt.legend(fontsize = 14)
plt.show()


# ------------------------------------- Evaluation on Test Set -------------------------------------

model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        y_pred.extend(outputs.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

y_pred = np.array(y_pred).flatten()
y_pred_binary = (y_pred > 0.5).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred_binary)
precision = precision_score(y_true, y_pred_binary)
recall = recall_score(y_true, y_pred_binary)
f1 = f1_score(y_true, y_pred_binary)
auroc = roc_auc_score(y_true, y_pred)
auprc = average_precision_score(y_true, y_pred)

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_binary))

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'AUROC: {auroc:.4f}')
print(f'AUPRC: {auprc:.4f}')