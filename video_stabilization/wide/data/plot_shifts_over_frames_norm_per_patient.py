import argparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler


def plot_normalized_shifts_by_patient_plotly(csv_file_path):
    data = pd.read_csv(csv_file_path)
    patient_ids = data['patient_id'].unique()
    color_map = {'good': 'blue', 'corrected': 'green', 'corrected_forced': 'red'}

    # all_shift_types = data['shift_type'].unique()

    n_patients = len(patient_ids)
    cols = 1
    rows = n_patients

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f'Patient {pid}' for pid in patient_ids])
    legend_added = set()

    for i, patient_id in enumerate(patient_ids, 1):
        patient_data = data[data['patient_id'] == patient_id].copy()

        scaler = MinMaxScaler()
        patient_data[['x_axis_shift', 'y_axis_shift']] = (
            scaler.fit_transform(patient_data[['x_axis_shift', 'y_axis_shift']]))
        patient_data['shift_norm'] = np.sqrt(patient_data['x_axis_shift'] ** 2 + patient_data['y_axis_shift'] ** 2)
        patient_data['shift_norm'] = scaler.fit_transform(patient_data[['shift_norm']])

        for shift_type in patient_data['shift_type'].unique():
            subset = patient_data[patient_data['shift_type'] == shift_type]
            show_legend = shift_type not in legend_added
            fig.add_trace(go.Scatter(
                x=subset['frame_number'],
                y=subset['shift_norm'],
                mode='markers',
                marker=dict(color=color_map[shift_type]),
                name=f'{shift_type}',
                hoverinfo='text',
                text=[f'Frame: {frame}, Shift Type: {shift_type}, Norm: {norm:.2f}'
                      for frame, norm in zip(subset['frame_number'], subset['shift_norm'])],
                showlegend=show_legend
            ), row=i, col=1)
            legend_added.add(shift_type)

    fig.update_layout(
        height=300 * n_patients,
        title='Normalized Shift Norm Analysis Over Frames for All Patients',
        xaxis_title='Frame Number',
        yaxis_title='Normalized Shift Norm',
        legend_title="Shift Types",
        showlegend=True
    )

    for j in range(1, n_patients + 1):
        fig['layout'][f'xaxis{j}']['title'] = 'Frame Number'
        fig['layout'][f'yaxis{j}']['title'] = 'Normalized Shift Norm'

    fig.show(config={'toImageButtonOptions': {
        'format': 'png'}})


def main():
    parser = argparse.ArgumentParser(description="Plot normalized shifts by patient.")
    parser.add_argument('csv_file_path', type=str, help="The path to the CSV file containing the shift data.")
    args = parser.parse_args()

    plot_normalized_shifts_by_patient_plotly(args.csv_file_path)


if __name__ == '__main__':
    main()
