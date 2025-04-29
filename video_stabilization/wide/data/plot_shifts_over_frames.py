import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_shifts_with_common_scale(csv_file_path):
    data = pd.read_csv(csv_file_path)
    patient_ids = data['patient_id'].unique()
    color_map = {'good': 'blue', 'corrected': 'green', 'corrected_forced': 'red'}

    fig = make_subplots(rows=len(patient_ids), cols=1,
                        subplot_titles=[f'Patient ID: {pid}' for pid in patient_ids])
    legend_added = set()

    for i, patient_id in enumerate(patient_ids, start=1):
        patient_data = data[data['patient_id'] == patient_id]

        patient_min = patient_data[['x_axis_shift', 'y_axis_shift']].min().min()
        patient_max = patient_data[['x_axis_shift', 'y_axis_shift']].max().max()

        range_buffer = (patient_max - patient_min) * 0.1
        y_range = [patient_min - range_buffer, patient_max + range_buffer]

        for shift_type in patient_data['shift_type'].unique():
            subset = patient_data[patient_data['shift_type'] == shift_type]

            show_legend = shift_type not in legend_added
            if show_legend:
                legend_added.add(shift_type)

            fig.add_trace(go.Scatter(
                x=subset['frame_number'],
                y=subset['x_axis_shift'],
                mode='markers',
                marker=dict(color=color_map[shift_type], size=6),
                name=f'X-Axis Shift ({shift_type})',
                legendgroup=shift_type,
                showlegend=show_legend
            ), row=i, col=1)

            fig.add_trace(go.Scatter(
                x=subset['frame_number'],
                y=subset['y_axis_shift'],
                mode='markers',
                marker=dict(symbol='x', color=color_map[shift_type], size=6),
                name=f'Y-Axis Shift ({shift_type})',
                legendgroup=shift_type,
                showlegend=False
            ), row=i, col=1)

            fig.update_yaxes(range=y_range, row=i, col=1)

        fig.update_layout(
            title='Shift Analysis Over Frames with Individual Scale for Each Patient',
            title_font_size=25,
            font_size=20,
            xaxis_title='Frame Number',
            yaxis_title='Shift Value',
            height=300 * len(patient_ids),
            showlegend=True
        )

    fig.show()


plot_shifts_with_common_scale('/home/dorozco/stabilization/presentation_all/csv_files/all_shifts_confidence.csv')
