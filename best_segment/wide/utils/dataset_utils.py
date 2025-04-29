import sys
import os
import pandas as pd
sys.path.insert(0, os.path.expanduser('~/git/ichor_analysis_evolved'))
from datasets.src.file_manager import FileManager
 
 
def file_name_to_office_path(file_name):
    file_name_df = FileManager.file_names_to_dataframe([file_name])
    recording_df = pd.read_csv(f"/mnt/disk1/office/recording_id.csv")
    recording_df = pd.merge(recording_df, file_name_df, on='recording_id')
 
    return recording_df['office_path'].iloc[0]