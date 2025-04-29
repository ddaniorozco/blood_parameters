import argparse

import pandas as pd
import plotly.express as px


def plot_consecutive_frames_norm(csv_file_path):
    df = pd.read_csv(csv_file_path)
    df['patient_id'] = df['patient_id'].astype(str)
    df['stability_metric'] *= 100

    color_scale = [
        [0, 'rgba(255, 99, 71, 0.6)'],
        [1, 'rgba(135, 206, 250, 0.6)']
    ]
    color_scale_distance = [
        [0, 'rgba(135, 206, 250, 0.6)'],
        [1, 'rgba(255, 99, 71, 0.6)']
    ]

    fig1 = px.bar(df, x='patient_id', y='stability_metric', color='stability_metric',
                  title='Stability Metric per Patient (Scaled 0-100)',
                  color_continuous_scale=color_scale,
                  text='stability_metric', range_color=[0, 100])
    fig1.update_traces(texttemplate='%{text:.1f}', textposition='inside')
    fig1.update_layout(
        title_font_size=40,
        font_size=35,
        xaxis_title="Patients ID",
        yaxis_title="Stability Metric (0-100)",
        yaxis=dict(range=[0, 100])
    )
    fig1.show()

    fig2 = px.bar(df, x='patient_id', y='distances', color='distances',
                  title='Maximum Norm over Consecutive Frames per Patient',
                  color_continuous_scale=color_scale_distance,
                  text='distances', range_color=[0, 10])
    fig2.update_traces(texttemplate='%{text:.1f}', textposition='inside')
    fig2.update_layout(
        title_font_size=40,
        font_size=35,
        xaxis_title="Patients ID",
        yaxis_title="Max Norm (Pixels)",
        yaxis=dict(range=[0, 10])
    )
    fig2.show()


def main():
    parser = argparse.ArgumentParser(description="Plot normalized shifts by patient.")
    parser.add_argument('csv_file_path', type=str, help="The path to the CSV file containing the shift data.")

    args = parser.parse_args()

    plot_consecutive_frames_norm(args.csv_file_path)


if __name__ == '__main__':
    main()
