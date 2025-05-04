import time
import numpy as np
import pandas
from pyproj import Transformer
from pyproj import CRS
import matplotlib.pyplot as plt
import torch 
import torch.nn.functional as F

def calc_radar_lonlat(NX,NY,res,lon_lat=False):

	wgs84 = "epsg:4326" # define wgs84 coords
	ITM = "epsg:2039" # define ITM coords
 
	# IMS RADAR location in ITM coordinates 
	lat0 = 32.007
	lon0 = 34.81456004

	transformerITM = Transformer.from_crs(wgs84, ITM)
	Xcenter,Ycenter = transformerITM.transform(lat0,lon0) # transform between coords

	nx = (NX-1)/(2/(1./res))
	ny = (NY-1)/(2/(1./res))

	x = np.linspace(-nx, nx, NX)
	y = np.linspace(ny, -ny, NY)
	
	xv, yv = np.meshgrid(x, y)
	xv = xv*1000.
	yv = yv*1000.
	RADX = Xcenter+xv
	RADY = Ycenter+yv
    
        
	# Convert from LAT, LON to ITM (Israel Transverse Mercator) X, Y
	transformerITM_back = Transformer.from_crs(ITM,wgs84)
	RADlat,RADlon = transformerITM_back.transform(RADX,RADY)
	if lon_lat:
		return RADlat,RADlon
	return RADX,RADY



def resize_radar_data(radar, size=(128, 128)):
    """
    Resizes the radar data to the specified size using nearest interpolation.
    
    Parameters:
        radar (torch.Tensor): Input tensor of shape (T, H, W) where T is time steps.
        size (tuple): Target size (H, W).
    
    Returns:
        torch.Tensor: Resized radar data of shape (T, size[0], size[1]).
    """
    resized_radar = torch.zeros((radar.shape[0], *size))
    for i in range(radar.shape[0]):
        resized_radar[i] = F.interpolate(
            radar[i].unsqueeze(0).unsqueeze(0), size=size, mode='nearest'
        ).squeeze()
    return resized_radar

def find_closest_indices(ims_all_stns, resized_radar_x, resized_radar_y):
    """
    Finds the closest indices for each station based on Euclidean distance.
    
    Parameters:
        ims_all_stns (DataFrame): DataFrame containing 'isr_grid_X' and 'isr_grid_Y' columns.
        resized_radar_x (np.array): X-coordinates of the radar grid.
        resized_radar_y (np.array): Y-coordinates of the radar grid.
    
    Returns:
        dict: Dictionary mapping closest (x, y) indices to station coordinates.
        np.array: Array of closest indices for all stations.
    """

    stn_dic = {}
    closest_indices = []
    
    for station in ims_all_stns:
        distances = np.sqrt((resized_radar_x - station[0])**2 + (resized_radar_y - station[1])**2)
        min_distance_indices = np.unravel_index(np.argmin(distances), distances.shape)
        
        if min_distance_indices not in stn_dic:
            stn_dic[min_distance_indices] = station
        else:
            stn_dic[min_distance_indices] = [stn_dic[min_distance_indices], station]
        
        closest_indices.append(min_distance_indices)
    
    return stn_dic, np.array(closest_indices)



def resize_radar_coords(target_size,lon_lat=False):
    """

    Parameters:
         target_size (h,w): Define the target dimensions.


    Returns:
        resized_radar_x (np.array): X-coordinates of the radar grid.
        resized_radar_y (np.array): Y-coordinates of the radar grid.
    """
    RADX,RADY = calc_radar_lonlat(561,561,1,lon_lat)
    # Convert NumPy arrays to PyTorch tensors
    radar_x_coords_tensor = torch.tensor(RADX)
    radar_y_coords_tensor = torch.tensor(RADY)

    # Perform resizing using PyTorch's interpolate function
    resized_radar_x = F.interpolate(radar_x_coords_tensor.unsqueeze(0).unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze()
    resized_radar_y = F.interpolate(radar_y_coords_tensor.unsqueeze(0).unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze()

    return resized_radar_x,resized_radar_y


def station_pixels_in_radar(ims_all_stns):

    # Define the target dimensions
    target_size = (128, 128)
    resized_radar_x,resized_radar_y = resize_radar_coords(target_size)
    stn_x_y = ims_all_stns[['isr_grid_X', 'isr_grid_Y']].values

    stn_dic, closest_indices = find_closest_indices(stn_x_y, resized_radar_x, resized_radar_y)

    
    # Initialize a dictionary to store the mapping
    station_names_radar_indexs = {}

    # Iterate through the station coordinates dictionary and find corresponding station names
    for coords, values in stn_dic.items():

        # Check if the values are lists
        if isinstance(values, list):
            for i, value in enumerate(values):
                x, y = value
                # Find the station name in the DataFrame using coordinates
                matching_row = ims_all_stns[(ims_all_stns['isr_grid_X'] == x) & (ims_all_stns['isr_grid_Y'] == y)]

                if not matching_row.empty:
                    station_name = matching_row.index[0]  # Get the index (station name)
                    if coords not in station_names_radar_indexs:
                        station_names_radar_indexs[coords] = station_name
                    else:
                        # If the key is already present, update the value
                           station_names_radar_indexs[coords] = [station_names_radar_indexs[coords], station_name]


        else:
            x, y = values
            matching_row = ims_all_stns[(ims_all_stns['isr_grid_X'] == x) & (ims_all_stns['isr_grid_Y'] == y)]

            if not matching_row.empty:
                station_name = matching_row.index[0]  # Get the index (station name)
                station_names_radar_indexs[coords] = station_name
    #returns dict with key as the tensor coords in radar and station name as value
    return station_names_radar_indexs

def get_rainy_shared_timestamps(radar_all, radar_all_timestamps, ims_all, ims_all_timestamps, datetime_index):
    """
    Finds timestamps where both radar and IMS data indicate rain.
    
    Parameters:
        radar_all (torch.Tensor): Radar data.
        radar_all_timestamps (np.array): Timestamps for radar data.
        ims_all (pd.DataFrame): IMS data indexed by timestamps.
        ims_all_timestamps (np.array): Timestamps for IMS data.
        datetime_index (pd.DatetimeIndex): Index of all timestamps.
    
    Returns:
        torch.Tensor: Filtered radar data.
        pd.DataFrame: Filtered IMS data.
        pd.DatetimeIndex: Shared timestamps.
    """
    shared_timestamps_idxs, shared_timestamps = sync_timestamps([radar_all_timestamps, ims_all_timestamps], verbose=1)
    
    ims_rainday = shared_timestamps[ims_all.loc[shared_timestamps].sum(1) > 0]
    rainday_ims_idx = np.where(np.isin(shared_timestamps, ims_rainday))[0]
    
    radar_sum = radar_all.sum([1, 2])
    bool_indices = radar_sum > 0
    int_indices = np.where(bool_indices)[0]
    
    radar_rain_idx = np.where(np.isin(shared_timestamps_idxs[0], int_indices))[0]
    
    radar_rainday = shared_timestamps[radar_rain_idx]
    rainday_radar_idx = np.where(np.isin(shared_timestamps, radar_rainday))[0]
    
    rainday_inxs = np.intersect1d(rainday_ims_idx, rainday_radar_idx)
    rainday_timestamps = np.intersect1d(ims_rainday, radar_rainday)
    rainday_timestamps = pd.DatetimeIndex(rainday_timestamps)
    
    radar = radar_all[rainday_inxs]
    ims = ims_all.loc[datetime_index]
    
    return radar, ims, rainday_timestamps

def save_rainy_shared_data(radar, ims, rainday_timestamps):
    """
    Saves the rainy shared timestamps and corresponding data.
    
    Parameters:
        radar (torch.Tensor): Filtered radar data.
        ims (pd.DataFrame): Filtered IMS data.
        rainday_timestamps (pd.DatetimeIndex): Shared timestamps.
    """
    torch.save(radar, 'tensor_RR_level85_1hour_all_1_hour_accumulated_rainy_shared_times_ims.pkl')
    torch.save(ims, 'ims_1_hour_accumulated_rainy_shared_times_radar.pkl')
    torch.save(rainday_timestamps, 'tensor_RR_level85_1hour_all_1_hour_accumulated_rainy_shared_timestamps.pkl')




def create_null_mask_64(station_names_radar_indexs, ims, radar_size_64_timestamps):
    null_mask_ims_64 = torch.ones(hourly_radar_size_128.shape[0], 128,128)
    filtered_ims_all = ims[ims.index.isin(radar_size_64_timestamps)]
    for coord, names in station_names_radar_indexs.items():
        x, y = coord
        if isinstance(names, list):
            ims_with_nan_64_1 = torch.tensor(filtered_ims_all[names[0]].values, dtype=torch.float)
            ims_with_nan_64_2 = torch.tensor(filtered_ims_all[names[1]].values, dtype=torch.float)
            check_null1= torch.zeros(hourly_radar_size_128.shape[0])
            check_null2 = torch.zeros(hourly_radar_size_128.shape[0])
            for i in range(ims_with_nan_64_1.size(0)):
                    if ims_with_nan_64_1[i].isnan():
                        check_null1[i] = 1

            for i in range(ims_with_nan_64_2.size(0)):
                    if ims_with_nan_64_2[i].isnan():
                        check_null2[i] = 1

            for i in range(ims_with_nan_64.size(0)):
                if check_null1[i]==1 and check_null2[i]==1:
                    null_mask_ims_64[i, x, y] = 0

        else:
            ims_with_nan_64 = torch.tensor(filtered_ims_all[names].values, dtype=torch.float)
            for i in range(ims_with_nan_64.size(0)):
                if ims_with_nan_64[i].isnan() :
                    null_mask_ims_64[i, x, y] = 0



    return null_mask_ims_64


def create_mask_128(station_names_radar_indexs, ims):
    """
    Creates a mask tensor of shape (12699, 128, 128) where each station's data is placed at its coordinates.
    
    Parameters:
        station_names_radar_indexs (dict): Dictionary mapping (x, y) coordinates to station names or lists of names.
        ims (dict): Dictionary where keys are station names and values are pandas Series with time-series data.
    
    Returns:
        torch.Tensor: Mask tensor of shape (12699, 128, 128).
    """
    mask_values_128 = torch.zeros(12699, 128, 128)
    
    for coord, names in station_names_radar_indexs.items():
        x, y = coord
        values = torch.zeros(12699)
        
        if isinstance(names, list):
            for name in names:
                values += torch.tensor(ims[name].fillna(0).values, dtype=torch.float)
        else:
            values += torch.tensor(ims[names].fillna(0).values, dtype=torch.float)
        
        mask_values_128[:, x, y] = values
    
    return mask_values_128

def save_tensor(tensor, filename):
    """
    Saves a tensor to a file using torch.save.
    
    Parameters:
        tensor (torch.Tensor): The tensor to save.
        filename (str): The file path to save the tensor.
    """
    torch.save(tensor, filename)

    
def create_mask_values_64_with_999(station_names_radar_indexs,radar, ims, radar_size_64_timestamps):    
    mask_values_64_999 = torch.full((radar.shape[0], 128,128), -999, dtype=torch.float64)
    for coord, names in station_names_radar_indexs.items():
        x, y = coord

        # # Initialize the values tensor
        values = torch.zeros(radar.shape[0], dtype=torch.float64)

        # Check if the names are lists
        if isinstance(names, list):
            for name in names:
                values = torch.tensor(ims[name].fillna(-999).values, dtype=torch.float64)
    
        else:
            values = torch.tensor(ims[names].fillna(-999).values, dtype=torch.float64)
        mask_values_64_999[:, x, y] = values
        
    return mask_values_64_999



def categorize_stations(lon_lat_stns, gedera_lat=31.5, yokneam_lat=32.5):
    """
    Categorizes stations into north, center, and south based on latitude.

    Parameters:
        lon_lat_stns (pd.DataFrame): DataFrame with 'stn_name' and 'stn_lat' columns.
        gedera_lat (float): Latitude threshold for the south region.
        yokneam_lat (float): Latitude threshold for the north region.

    Returns:
        tuple: Lists of station names categorized into north, center, and south.
    """
    center_stns, north_stns, south_stns = [], [], []
    
    for i in range(len(lon_lat_stns['stn_name'])):
        lat = lon_lat_stns['stn_lat'][i]
        name = lon_lat_stns['stn_name'][i]

        if lat > yokneam_lat:
            north_stns.append(name)
        elif lat < gedera_lat:
            south_stns.append(name)
        else:
            center_stns.append(name)

    return north_stns, center_stns, south_stns

def create_station_dicts(station_names_radar_indexs,lon_lat_stns):
    """
    Creates dictionaries mapping coordinates to station names for each region.

    Parameters:
        station_names_radar_indexs (dict): Dictionary with coordinates as keys and station names as values.
        north_stns (list): List of station names in the north region.
        center_stns (list): List of station names in the center region.
        south_stns (list): List of station names in the south region.

    Returns:
        tuple: Dictionaries for north, center, and south stations.
    """
    dict_north, dict_center, dict_south ={}, {}, {} 
    north_stns, center_stns, south_stns = categorize_stations(lon_lat_stns)

    for coord, name in station_names_radar_indexs.items():
        if name in north_stns:
            dict_north[coord] = name
        elif name in center_stns:
            dict_center[coord] = name  
        elif name in south_stns:
            dict_south[coord] = name

    return dict_north, dict_center, dict_south
