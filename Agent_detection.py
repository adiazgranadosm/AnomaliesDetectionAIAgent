# %% [markdown]
# #### Libraries

# %%
import os
import pandas as pd
from nixtla import NixtlaClient
from datetime import datetime
from dateutil.parser import parse
import numpy as np
import time
from datetime import timezone

# %% [markdown]
# #### General parameters and configurations

# %%
# Initialize an empty list to store DataFrames
data_frames_to_merge = []


# Base folder path (replace with your actual path)
base_folder = './data'
time_col = 'timestamp'
target_col = 'target'
number_anomalies_predict=20    

data_columns = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP', 'P-JUS-CKGL', 'T-JUS-CKGL', 'QGL']

# Get the Nixtla API (TimeGPT) key from environment variable
nixtla_client = NixtlaClient(
    api_key=os.environ.get("NIXTLA_API_KEY")
)


# %% [markdown]
# #### Call function to detect anomalies online

# %%
def detect_anomalies_online(df, time_col, target_col):
    """Call the TimeGPT Nixtla API to detect anomalies in the target column of a DataFrame."""
    return nixtla_client.detect_anomalies_online(
        df = df,
        time_col=time_col,
        target_col=target_col,
        freq='5s',                      # Specify the frequency of the data
        h=30,                           # Specify the forecast horizon
        level=80,                       # Set the confidence level for anomaly detection
        detection_size=number_anomalies_predict,              # How many steps you want for analyzing anomalies
        threshold_method = 'multivariate',  # Specify the threshold_method as 'multivariate'    
        
    )

# %% [markdown]
# #### Functions to detect anomalies in a range of rows that correspond to a timeseries rows

# %%

def detect_anomalies(df, ini_row, end_row):
    """General function to identify an anomaly in a dataset using TimeGPT Nixtla API"""
    data_frame_target = pd.DataFrame()
    type_anomaly = 0
    # Convert numeric columns back to float
    for col in df.columns:
        # Check if the column can be converted to numeric (including decimals)
        if (col in data_columns):
            try:
                df.loc[:, col] = pd.to_numeric(df[col])
            except ValueError:
                pass        
            df.loc[:, col]= df[[col]].replace('', 0) 
            #scaler = MinMaxScaler()              
            #df.loc[:, col] = scaler.fit_transform(df[[col]])
        if (col in ['timestamp']): 
            try:
                df.loc[:, col] = pd.to_datetime(df[col])
            except ValueError:
                pass                                     

                       
    common_cols = list(set(df.columns) & set(data_columns))
        
    df.loc[:, 'target'] = df[common_cols].sum(axis=1, min_count=1)   
            
    df = df.iloc[ini_row:end_row].reset_index(drop=True)    
    data_frame_target= data_frame_target.iloc[ini_row:end_row].tail(number_anomalies_predict).reset_index(drop=True)    
    
    data_frame_anomaly = detect_anomalies_online(df, time_col, target_col)
    
    # Temporary method to simulate the type of anomaly, this function should be replaced with the actual logic to determine the type of anomaly
    np.random.seed(int(time.time()))
    type_anomaly = np.random.randint(1, 5)

    return data_frame_anomaly, type_anomaly


def detect_anomalies_dates(type_anomaly, start_date, end_date):
    """Function to identify an anomaly in a specific range of dates and specific id or type of anomaly""" 

    print(f'Function detect_anomalies_dates: {type_anomaly}, {start_date}, {end_date}')
  
    folder_path = os.path.join(base_folder)

    for file_name in os.listdir(folder_path):   
        if file_name.startswith('WELL-'):
            print(f'Processing file: {file_name}')
            file_path = os.path.join(folder_path, file_name)            
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Define the start and end dates and times
            start_datetime = parse(start_date)
            end_datetime = parse(end_date)
            
             # Ensure start_datetime and end_datetime are timezone-aware (UTC)
            if start_datetime.tzinfo is None:
                start_datetime = start_datetime.replace(tzinfo=timezone.utc)
            if end_datetime.tzinfo is None:
                end_datetime = end_datetime.replace(tzinfo=timezone.utc)

            # Convert 'timestamp' column to datetime and make it timezone-aware (UTC)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

            # Filter the dataframe
            filtered_df = df[(df['timestamp'] >= start_datetime) & (df['timestamp'] <= end_datetime)]  

            if filtered_df.shape[0] > 0:
                return detect_anomalies(filtered_df, 0, filtered_df.shape[0])
            else:
                return None         
                
def detect_all_anomalies_dates(start_date, end_date):
    """Function to identify an anomaly in a specific range of dates""" 

    print(f'Function detect_all_anomalies_dates: {start_date}, {end_date}')

    folder_path = os.path.join(base_folder)
    # Iterate over the files in the folder    
    for file_name in os.listdir(folder_path):   
        if file_name.startswith('WELL-'):
            print(f'Processing file: {file_name}')
            file_path = os.path.join(folder_path, file_name)            
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Define the start and end dates and times
            start_datetime = parse(start_date)
            end_datetime = parse(end_date)
            
            # Ensure start_datetime and end_datetime are timezone-aware (UTC)
            if start_datetime.tzinfo is None:
                start_datetime = start_datetime.replace(tzinfo=timezone.utc)
            if end_datetime.tzinfo is None:
                end_datetime = end_datetime.replace(tzinfo=timezone.utc)

            # Convert 'timestamp' column to datetime and make it timezone-aware (UTC)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

            # Filter the dataframe
            filtered_df = df[(df['timestamp'] >= start_datetime) & (df['timestamp'] <= end_datetime)] 

            if filtered_df.shape[0] > 0:
                return detect_anomalies(filtered_df, 0, filtered_df.shape[0])
            else:
                return None

def detect_all_anomalies_dates_by_parameter(start_date, end_date, parameter):
    """Function to identify an anomaly in a specific range of dates and using an specific parameter desviation""" 

    print(f'Function detect_all_anomalies_dates_by_parameter: {start_date}, {end_date}, {parameter}')
    
    folder_path = os.path.join(base_folder)
    # Iterate over the files in the folder    
    for file_name in os.listdir(folder_path):   
        if file_name.startswith('WELL-'):
            print(f'Processing file: {file_name}')
            file_path = os.path.join(folder_path, file_name)            
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Define the start and end dates and times
            start_datetime = parse(start_date)
            end_datetime = parse(end_date)
            
            # Ensure start_datetime and end_datetime are timezone-aware (UTC)
            if start_datetime.tzinfo is None:
                start_datetime = start_datetime.replace(tzinfo=timezone.utc)
            if end_datetime.tzinfo is None:
                end_datetime = end_datetime.replace(tzinfo=timezone.utc)

            # Convert 'timestamp' column to datetime and make it timezone-aware (UTC)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            
            # Filter the dataframe
            filtered_df = df[(df['timestamp'] >= start_datetime) & (df['timestamp'] <= end_datetime)]            

            if filtered_df.shape[0] > 0:
                # Select only timestamp and parameter columns                   
                selected_parameter_df = filtered_df[["timestamp", parameter]]
                return detect_anomalies(selected_parameter_df, 0, selected_parameter_df.shape[0])                



