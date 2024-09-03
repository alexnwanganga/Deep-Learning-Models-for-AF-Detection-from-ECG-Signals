#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 19:27:45 2024

@author: mateo
"""
#Imports nessecary libraries for WaveForm Analysis, File redirecting, and reading CSV files
import wfdb
import csv
import os
import shutil

#/Users/mateo/Repos/Deep-Learning-of-ECG-signals

# Determines path to orgin as well as each destination  
folder_A = '/Users/mateo/Repos/Deep-Learning-of-ECG-signals/Data/Class_A'
folder_N = '/Users/mateo/Repos/Deep-Learning-of-ECG-signals/Data/Class_N'
folder_O = '/Users/mateo/Repos/Deep-Learning-of-ECG-signals/Data/Class_O'
source_dir = '/Users/mateo/Repos/Deep-Learning-of-ECG-signals/Data'
ref_dir = '/Users/mateo/Repos/Deep-Learning-of-ECG-signals/References'

# Total number of files is 8528
for i in range(0, 10):
    #Declares file name in each cycle 
    file_name = f'A{i+1:05d}' 
    
    #Path to Header and Data files
    data_path = os.path.join(source_dir, file_name)
    
    try:
        
        
        #Reads header file 
        header = wfdb.rdheader(data_path)
        print(f"Header of file: {file_name}:")
        print(header)
        
        #Reads data file 
        record = wfdb.rdrecord(data_path)
        print(f"Signal Data of file: {file_name}:")
        print(record)
        
        #Extracts the signal data
        for idx, signal in enumerate(record.p_signal.T):
            print(f"\nSignal data for {file_name}:")
            print(signal)
        
        #Plots the signal data on graph 
        wfdb.plot_wfdb(record=record, title = f'Signal Data for {file_name}')
            
        
        #Scans through csv file data 
        
        #Finds path to reference file and scans thru new line in reference file 
        with open (os.path.join(ref_dir, 'REFERENCE.csv'), newline = '') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                file_id, category = row
                
                #Checks if the current file read corresponds to the classifaication being read
                if file_id == file_name:
                
                    #Determines destination based on category
                    if category == 'N':
                        target_dir = folder_N
                    elif category == 'A':
                        target_dir = folder_A
                    else:
                        target_dir = folder_O
                    
                    # Source and destination paths
                    source_path_mat = os.path.join(source_dir, f'{file_name}.mat')
                    dest_path_mat = os.path.join(target_dir, f'{file_name}.mat')
                    source_path_hea = os.path.join(source_dir, f'{file_name}.hea')
                    dest_path_hea = os.path.join(target_dir, f'{file_name}.hea')
                    
                    # Moves the .mat and .hea file to the target directory
                    if os.path.exists(source_path_mat):
                        shutil.move(source_path_mat, dest_path_mat)
                        shutil.move(source_path_hea, dest_path_hea)
                        print(f"Moved from {source_path_mat} to {dest_path_mat}")
                    else:
                        print(f"File in {source_path_mat} not found. Skipping.")

        
    # Allows program to continue if the expected file is not in the original specified location   
    except FileNotFoundError:
        print(f"File {file_name} not found. Skipping.")
        continue
        
    
    