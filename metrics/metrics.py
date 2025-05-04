import numpy as np
import pandas as pd
import sys
from sklearn.metrics import confusion_matrix
sys.path.insert(0, "/home/labs/rudich/meiray/RadarGaugeNet/utils")
from data import *


def get_metrics_binar(o,p,labels=[0,1],smallest_event_level=None):
    if type(o)!=np.ndarray:
        o = o.detach().cpu().numpy()
        p =  p.detach().cpu().numpy()
    if smallest_event_level is not None:
        o[o>=smallest_event_level]=2
        p[p>=smallest_event_level]=2
    tn, fp, fn, tp = confusion_matrix(o,p,labels=labels).ravel()
    acc = (tp+tn)/(tp+fp+tn+fn)
    if tp+fp==0:
        prc = np.nan
    else:
        prc = tp/(tp+fp)
    if tp+fn==0:
        rcl = np.nan
    else:
        rcl = tp/(tp+fn)
    if rcl==np.nan or prc==np.nan or rcl==0:
        prcrcl_ratio = np.nan
        prcrcl_avg = np.nan
    else:
        prcrcl_ratio = prc/rcl
        prcrcl_avg = (prc+rcl)/2
    if tp+fp+fn==0:
        csi = np.nan
    else:
        csi = tp/(tp+fp+fn)
    if np.isnan(prc) or np.isnan(rcl) or (prc + rcl) == 0:
        f1 = np.nan
    else:
        f1 = 2 * (prc * rcl) / (prc + rcl)
    
    return [acc, prc, rcl, prcrcl_ratio, prcrcl_avg, csi, f1]

def calculating_stns_metrics_with_nan(recon,mask_batch,threshold,station_names_radar_indexs):
    metrics_list = []
    for (x,y), stations in station_names_radar_indexs.items():
        ground_truth = mask_batch[:,0,x,y]
        valid = ground_truth!= -999
        valid_ground_truth = ground_truth[valid]
        prediction = recon[:,0,x,y]
        valid_prediction = prediction[valid]
        gt_binary = (valid_ground_truth > threshold)*1
        pred_binary = (valid_prediction > threshold)*1
        accuracy,precision,recall,_,_,_,_ = get_metrics_binar(gt_binary,pred_binary)

         # Append to list
        metrics_list.append({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "station": stations,
            "x": x,
            "y": y
        })

    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df

def printing_training_metrics(radar_input,recon_outputs,stations_output,station_names_radar_indexs,lon_lat_stns,print_=True):  
    dict_north, dict_center, dict_south = create_station_dicts(station_names_radar_indexs,lon_lat_stns)
    regions_dicts = [dict_north,dict_center,dict_south]
    thresholds = [0,0.1,0.5,1]

    for threshold in thresholds:
        print(f"Threshold: {threshold}")
        # Iterate through regions and compute metrics
        for i,region_dict in enumerate(regions_dicts):
            region_name = ['North', 'Center', 'South'][i]
            print(f"Region: {region_name}")
            recon_metrics = calculating_stns_metrics_with_nan(recon_outputs,stations_output,threshold,region_dict)
            radar_metrics = calculating_stns_metrics_with_nan(radar_input,stations_output,threshold,region_dict)    


            # Extract test metrics for the region
            our_metrics = [
                recon_metrics.iloc[:, 0].median(),
                recon_metrics.iloc[:, 1].median(),
                recon_metrics.iloc[:, 2].median()
            ]

            our_metrics_series = pd.Series(our_metrics, index=['Accuracy', 'Precision', 'Recall'])
            # Extract test metrics for the region
            radar_metrics_final = [
                radar_metrics.iloc[:, 0].median(),
                radar_metrics.iloc[:, 1].median(),
                radar_metrics.iloc[:, 2].median()
            ]
            radar_metrics_series = pd.Series(radar_metrics_final, index=['Accuracy', 'Precision', 'Recall'])
            # Create a metrics table
            metrics_table = pd.DataFrame([radar_metrics_series, our_metrics_series],
                                         index=['Radar', 'Test'],
                                         columns=['Accuracy', 'Precision', 'Recall'])
            if print_:
                # Print the metrics
                print(metrics_table[['Accuracy', 'Precision', 'Recall']])
                print("\n")
            else:
                return metrics_table
