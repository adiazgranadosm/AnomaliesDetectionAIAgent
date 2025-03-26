# %%
import os
import pandas as pd
from nixtla import NixtlaClient
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from datetime import datetime
from dateutil.parser import parse

# %%
# Initialize an empty list to store DataFrames
data_frames_to_merge = []


# Base folder path (replace with your actual path)
base_folder = './data'
time_col = 'timestamp'
target_col = 'target'
number_anomalies_predict=20    

# List of folders
#folders = ['1', '2', '3', '4', '5', '6', '7']
folders = ['1', '2', '3']

data_columns = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP', 'P-JUS-CKGL', 'T-JUS-CKGL', 'QGL']


nixtla_client = NixtlaClient(
    # defaults to os.environ.get("NIXTLA_API_KEY")
    api_key = 'nixak-ydVyUIawnVh68qhFTgxxWrCKzLNvOKmLVNpDr24m94DHR4e1sYYkntIF0iyIhjhUQLmjEUypp4vGORXe'
)

# %%
def detect_anomalies_online(df, time_col, target_col):
    """Call the TimeGPT Nixtla API to detect anomalies in the target column of a DataFrame."""
    return nixtla_client.detect_anomalies_online(
        df = df,
        time_col=time_col,
        target_col=target_col,
        freq='s',                      # Specify the frequency of the data
        h=30,                           # Specify the forecast horizon
        level=80,                       # Set the confidence level for anomaly detection
        detection_size=number_anomalies_predict,              # How many steps you want for analyzing anomalies
        threshold_method = 'multivariate',  # Specify the threshold_method as 'multivariate'    
        
    )

# %%
# Utility function to plot anomalies
def plot_anomaly(df, anomaly_df, time_col = 'ts', target_col = 'y'):
    """Plot anomaly detection."""
    merged_df = pd.merge(df.tail(300), anomaly_df[[time_col, 'anomaly', 'TimeGPT']], on=time_col, how='left')
    plt.figure(figsize=(12, 2))
    plt.plot(merged_df[time_col], merged_df[target_col], label='y', color='navy', alpha=0.8)
    plt.plot(merged_df[time_col], merged_df['TimeGPT'], label='TimeGPT', color='orchid', alpha=0.7)
    plt.scatter(merged_df.loc[merged_df['anomaly'] == True, time_col], merged_df.loc[merged_df['anomaly'] == True, target_col], color='orchid', label='Anomalies Detected')
    plt.legend()
    plt.tight_layout()
    plt.show()

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
            scaler = MinMaxScaler()              
            df.loc[:, col] = scaler.fit_transform(df[[col]])
        if (col in ['timestamp']): 
            try:
                df.loc[:, col] = pd.to_datetime(df[col])
            except ValueError:
                pass                                      
        if (col in ['class']):                              
            data_frame_target.loc[:, 'anomaly'] = df[col]
            max_val = data_frame_target.max(skipna=True).max()
            if pd.isna(max_val):
                type_anomaly = None  # or set a default like -1 or 0
            else:
                type_anomaly = int(max_val)
                if type_anomaly > 100:
                    type_anomaly = type_anomaly - 100

            data_frame_target.loc[:, 'anomaly']  = df[col].apply(lambda x: True if x != 0 and x != '' else False)  
                       
    common_cols = list(set(df.columns) & set(data_columns))

        
    df.loc[:, 'target'] = df[common_cols].sum(axis=1, min_count=1)
    df = df.drop('class', axis=1)    
            
    df = df.iloc[ini_row:end_row].reset_index(drop=True)    
    data_frame_target= data_frame_target.iloc[ini_row:end_row].tail(number_anomalies_predict).reset_index(drop=True)    
    
    data_frame_anomaly = detect_anomalies_online(df, time_col, target_col)
    
    print("Data frame actual anomalies")
    print(data_frame_target.shape)
    print("Data frame anomalies predicted")
    print(data_frame_anomaly.shape)

    accuracy = accuracy_score(data_frame_anomaly['anomaly'], data_frame_target['anomaly'])
    print(f'Accuracy: {accuracy}')

    return data_frame_anomaly, type_anomaly


# %%

def detect_anomalies_file(type_anomaly, file_name, ini_row, end_row):
    """Function to identify an anomaly in a specific log file and range of rows"""

    print("\nCALL FUNCTION TO DETECT ANOMALIES IN MACHINE LEARNING OR LLM MODEL\n")
    folder_path = os.path.join(base_folder, str(type_anomaly))
    print(f'Function detect_anomalies_file: {type_anomaly}, {folder_path}, {file_name}, {ini_row}, {end_row}')
    if file_name.startswith('WELL-'):
        print(f'Processing file: {file_name}')
        file_path = os.path.join(folder_path, file_name)    
        df = pd.DataFrame()        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        return detect_anomalies(df, ini_row, end_row)

# %%
def detect_anomalies_dates(type_anomaly, start_date, end_date):
    """Function to identify an anomaly in a specific range of dates and specific id or type of anomaly""" 

    print("\nCALL FUNCTION TO DETECT ANOMALIES IN MACHINE LEARNING OR LLM MODEL\n")
    print(f'Function detect_anomalies_dates: {type_anomaly}, {start_date}, {end_date}')
    type_anomaly_str = str(type_anomaly)
    folder_path = os.path.join(base_folder, type_anomaly_str)

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
                



# %%
def detect_all_anomalies_dates(start_date, end_date):
    """Function to identify an anomaly in a specific range of dates""" 

    print("\nCALL FUNCTION TO DETECT ANOMALIES IN MACHINE LEARNING OR LLM MODEL\n")
    print(f'Function detect_all_anomalies_dates: {start_date}, {end_date}')

    for folder in folders:    
        folder_path = os.path.join(base_folder, folder)
        print(f'Processing folder: {folder}')
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
                
                

# %%
def detect_all_anomalies_dates_by_parameter(start_date, end_date, parameter):
    """Function to identify an anomaly in a specific range of dates and using an specific parameter desviation""" 

    print("\nCALL FUNCTION TO DETECT ANOMALIES IN MACHINE LEARNING OR LLM MODEL\n")
    print(f'Function detect_all_anomalies_dates_by_parameter: {start_date}, {end_date}, {parameter}')

    for folder in folders:    
        folder_path = os.path.join(base_folder, folder)
        print(f'Processing folder: {folder}')
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
                    # Select only timestamp, parameter and class columns
                    selected_parameter_df = filtered_df[["timestamp", parameter, "class"]]
                    return detect_anomalies(selected_parameter_df, 0, selected_parameter_df.shape[0])






