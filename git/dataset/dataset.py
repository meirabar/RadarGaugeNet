import torch
from torch.utils.data import Dataset,Subset
class RadarDataset(Dataset):
    """
        Args:
            timestamps (List of DatetimeIndex): List of timestamps corresponding to radar data.
            radar_data (torch.Tensor): Tensor of radar images, shape (N, H, W).
            station_data (torch.Tensor): Tensor of station values, shape (N, H, W).
        """

    def __init__(self, timestamps, radar, stn_values):
        self.radar_data = radar
        self.stn_values = stn_values
        self.timestamps = timestamps 
        
    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):
        timestamp = str(self.timestamps[idx])
        radar_img = self.radar_data[idx] 
        stn_values = self.stn_values[idx]
        return radar_img ,stn_values ,timestamp
        
