import pandas as pd

# Load CSV files
csv1 = pd.read_csv('path/to/csv1.csv')
csv2 = pd.read_csv('path/to/csv2.csv')

# Merge CSV files on 'frame_name' column
merged_df = pd.merge(csv2, csv1[['frame_name', 'class_name']], on='frame_name', how='left')

# Reorder columns as required
merged_df = merged_df[['frame_name', 'class_name', 'confidence', 'left', 'top', 'width', 'height']]

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('merged_output.csv', index=False)

print("CSV files merged successfully!")
