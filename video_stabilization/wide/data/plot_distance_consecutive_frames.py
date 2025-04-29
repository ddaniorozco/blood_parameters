import argparse

import pandas as pd
import plotly.express as px


def plot_consecutive_frames_norm(csv_file_path):
    df = pd.read_csv(csv_file_path)

    fig = px.line(df, x="frame_number", y="distance_consecutive_frames", color='patient_id',
                  title="Distance Between Consecutive Frames")
    fig.update_layout(
        title_font_size=40,
        font_size=35,
        xaxis_title="Frame Number",
        yaxis_title="Norm consecutive frames",
        yaxis=dict(range=[0, 10])
    )
    fig.show()


def main():
    parser = argparse.ArgumentParser(description="Plot normalized shifts by patient.")
    parser.add_argument('csv_file_path', type=str, help="The path to the CSV file containing the shift data.")

    args = parser.parse_args()

    plot_consecutive_frames_norm(args.csv_file_path)


if __name__ == '__main__':
    main()
