import os
import sys
import math
import random
import pickle
import datetime
from typing import Dict, Union, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import scipy
from scipy import interpolate
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm.auto import tqdm as tqdm_auto
from einops import rearrange

import torch
import torch.nn.functional as F
import torch.nn.utils as clip_grad
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import models, transforms
from torchinfo import summary

sys.path.insert(0, "/home/labs/rudich/meiray/RadarGaugeNet/utils")
from metrics import *
from plotting import *


UNetModel_folder_checkpoints = "/home/labs/rudich/meiray/UNetModel_folder_checkpoints/"


def find_latest_model(directory, file_pattern='checkpoint_UNetModel_'):
    latest_model = ''
    latest_date = datetime.date.min
    max_epoch = -1

    for file in os.listdir(directory):
        if file.startswith(file_pattern):
            # Extract the date and epoch from the filename
            parts = file.split('_')
            date_str = parts[-3]
            epoch_str = parts[-1].replace('.pth', '')
            
            try:
                file_date = datetime.datetime.strptime(date_str, '%d-%m-%Y').date()
                epoch = int(epoch_str)

                if file_date > latest_date or (file_date == latest_date and epoch > max_epoch):
                    latest_date = file_date
                    max_epoch = epoch
                    latest_model = file
            except ValueError:
                # If the date or epoch number is not valid, skip this file
                continue

    if latest_model:
        return os.path.join(directory, latest_model)
    else:
        return None



def train_and_evaluate(train_dataloader,test_dataloader,re_train, device,
                        optimizer, UNetModel, criterion, batch_size,border_israel_128,full_border_israel_128,
                        epochs,h,w,threshold,station_names_radar_indexs,lon_lat_stns,loss_weights,south_station_mask):
    


  
    if re_train == False:
        checkpoint_path = find_latest_model(UNetModel_folder_checkpoints)
        checkpoint = torch.load(checkpoint_path)
        UNetModel.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f'continue training from epoch {start_epoch}')


    ##### start of training loop ########
    for epoch in tqdm_auto(range(epochs), colour='#2bceee'):

        UNetModel.train()
        total_train_loss = 0
        weighted_total_loss = 0
        Train_stn_PRC = []
        Train_stn_RCL = []
        all_recon_batch = []
        all_stn_mask_batch = []
        all_radar_batch= []
        Train_stn_ACC = []
        
        for radar_img ,mask,recon_timestamps in  tqdm_auto(train_dataloader):
            radar_img_batch = radar_img.to(device)
            stn_mask_batch = mask.to(device)
            optimizer.zero_grad()
            recon_batch = UNetModel(radar_img_batch)

            valid_index = (stn_mask_batch != -999)        
            crit_loss = criterion(recon_batch, stn_mask_batch) 
            crit_loss = crit_loss[valid_index]
            crit_loss = torch.log(crit_loss + 1.)
            recon_loss_stn = crit_loss.mean()
                
            
            
            all_recon_batch.append(recon_batch)
            all_stn_mask_batch.append(stn_mask_batch)
            all_radar_batch.append(radar_img_batch)


            

            # Bias Correction for South
            south_mask = south_station_mask.unsqueeze(0).unsqueeze(1).to(device)  # Shape: [1, 1, 128, 128]
            # Apply the mask to extract only southern station values
            south_mask = south_mask.expand_as(recon_batch)  # Expand to match batch size
            south_mask = (south_mask > 0).bool()  # Convert to boolean mask

            if south_mask.sum() > 0:  # Ensure there are southern stations
            # Compute bias only for valid pixels in the south
                south_bias = torch.mean((recon_batch - stn_mask_batch)[south_mask])  # Compute mean bias
                south_bias_loss = torch.abs(south_bias)  # Penalize absolute bias
            else:
                south_bias_loss = torch.tensor(0.0, device=device)

            # Combine Losses
            loss = (
                recon_loss_stn * loss_weights[0] +
                south_bias_loss * loss_weights[1]  
            )

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()



            
            
            
            
        all_recon_batch = torch.cat(all_recon_batch, dim=0)  
        all_stn_mask_batch = torch.cat(all_stn_mask_batch, dim=0)
        all_radar_batch = torch.cat(all_radar_batch, dim=0)
        
        
       
        #binary metrics for stns:
        binary_metrics_stns = calculating_stns_metrics_with_nan(all_recon_batch,all_stn_mask_batch,threshold,station_names_radar_indexs)
        Train_stn_ACC.append(binary_metrics_stns.iloc[:, 0])
        Train_stn_PRC.append(binary_metrics_stns.iloc[:, 1])
        Train_stn_RCL.append(binary_metrics_stns.iloc[:, 2])


        
            
            
        stn_mask_batch_plot = stn_mask_batch.clone() 
        stn_mask_batch_plot[stn_mask_batch_plot== -999] = 0
        print('training imgs:')
        plot_original_vs_reconstructed(recon_batch[:2], radar_img_batch[:2], recon_timestamps[:2],            stn_mask_batch[:2],border_israel_128,station_names_radar_indexs,full_border_israel_128)   


     
        # Calculate and display the average training loss
        avg_train_loss = total_train_loss / len(train_dataloader)

        stn_accuracy = np.nanmean(Train_stn_ACC)
        stn_precision = np.nanmean(Train_stn_PRC)
        stn_recall = np.nanmean(Train_stn_RCL)

        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.6f}")
        
   
    
        # Evaluation step on the test dataset
        UNetModel.eval()
        total_loss_test = 0
        Test_stn_PRC = []
        Test_stn_ACC = []
        Test_stn_RCL = []
        test_all_recon_batch = []
        test_all_stn_mask_batch = []
        test_all_radar_batch = []

        
        with torch.no_grad():
            for radar_img ,mask ,recon_timestamps  in test_dataloader:
                radar_img_batch = radar_img.to(device)
                stn_mask_batch = mask.to(device)
                recon_batch = UNetModel(radar_img_batch)
                
                # test loss 
                valid_index = (stn_mask_batch != -999)
                crit_loss_test = criterion(recon_batch, stn_mask_batch) 
                crit_loss_test = crit_loss_test[valid_index]
                crit_loss_test = torch.log(crit_loss_test + 1.)
                total_stn_loss_test = crit_loss_test.mean()
                total_loss_test += total_stn_loss_test.item()
              
                test_all_recon_batch.append(recon_batch)
                test_all_stn_mask_batch.append(stn_mask_batch)
                test_all_radar_batch.append(radar_img_batch)
                


                 #binary metrics for stns:
            test_all_recon_batch = torch.cat(test_all_recon_batch, dim=0)  # Combine along batch dimension
            test_all_stn_mask_batch = torch.cat(test_all_stn_mask_batch, dim=0)
            test_all_radar_batch = torch.cat(test_all_radar_batch, dim=0)
            
            printing_training_metrics(test_all_radar_batch,test_all_recon_batch,test_all_stn_mask_batch,station_names_radar_indexs,lon_lat_stns)

            test_binary_metrics_stns = calculating_stns_metrics_with_nan(test_all_recon_batch,test_all_stn_mask_batch,threshold,station_names_radar_indexs)
            Test_stn_ACC.append(test_binary_metrics_stns.iloc[:, 0])
            Test_stn_PRC.append(test_binary_metrics_stns.iloc[:, 1])
            Test_stn_RCL.append(test_binary_metrics_stns.iloc[:, 2])


                
        stn_mask_batch_plot = stn_mask_batch.clone()
        stn_mask_batch_plot[stn_mask_batch_plot== -999] = 0
        
        print('test imgs:')
        plot_original_vs_reconstructed(recon_batch[:2], radar_img_batch[:2], recon_timestamps[:2], stn_mask_batch[:2],border_israel_128,station_names_radar_indexs,full_border_israel_128)   


        # Calculate and display the average test loss
        avg_test_loss = total_loss_test / len(test_dataloader)

        
        stn_accuracy_test = np.nanmean(Test_stn_ACC)        
        stn_precision_test = np.nanmean(Test_stn_PRC)
        stn_recall_test = np.nanmean(Test_stn_RCL)


        print(f"Epoch {epoch+1}/{epochs}, Test Loss: {avg_test_loss:.6f}")
        
    
          # Collect metrics 
        metrics_train = pd.DataFrame({'Epoch':[epoch],
                              'Train_mse':avg_train_loss,
                              'Train_ACC': stn_accuracy,
                              'Train_stn_PRC':stn_precision,
                              'Train_stn_RCL':stn_recall
                                   })
        metrics_test = pd.DataFrame({'Epoch':[epoch], 
                              'Test_mse':avg_test_loss,
                              'Test_ACC': stn_accuracy_test,
                              'Test_stn_PRC':stn_precision_test,
                              'Test_stn_RCL':stn_recall_test
                             })
        
        epoch_prints = "Epoch:{}; Train_mse:{};Train_ACC:{};Train_stn_PRC:{}; Train_stn_RCL:{};"
        
        m_all = metrics_train.values.flatten().tolist()
        
        m_all = np.round(m_all,3)
        print('\nMetrics'+" ", epoch_prints.format(*m_all))
        epoch_prints_test = "Epoch:{}; Test_mse:{};Test_ACC:{};Test_stn_PRC:{}; Test_stn_RCL:{};"

        m_all_test= metrics_test.values.flatten().tolist()

        m_all_test = np.round(m_all_test,3)
        print('\nMetrics'+" ", epoch_prints_test.format(*m_all_test))

        dfs_train.append(metrics_train)
        dfs_test.append(metrics_test)
      
       # Save the trained model
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': UNetModel.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),

        }

         


        


        torch.save(checkpoint, UNetModel_folder_checkpoints + f'checkpoint_UNetModel_{current_date}_epoch_{epoch+1}.pth')  
    
    return UNetModel,dfs_train,dfs_test,UNetModel.state_dict() 



