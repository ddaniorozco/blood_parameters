import argparse

import numpy as np
import pandas as pd
import plotly.express as px


def plot_shifts_over_frames_norm(csv_file_path):
    df = pd.read_csv(csv_file_path)
    df['shift_norm'] = df[['x_axis_shift', 'y_axis_shift']].apply(lambda x: np.sqrt(x["x_axis_shift"] ** 2 +
                                                                                    x["y_axis_shift"] ** 2), axis=1)
    fig = px.line(df, x='frame_number', y='shift_norm', color='patient_id', title="Norm Analysis Over Frames")
    fig.update_layout(
        title_font_size=40,
        font_size=35,
        xaxis_title="Frame Number",
        yaxis_title="Norm shifts"
    )
    fig.show()


def main():
    parser = argparse.ArgumentParser(description="Plot normalized shifts by patient.")
    parser.add_argument('csv_file_path', type=str, help="The path to the CSV file containing the shift data.")

    args = parser.parse_args()

    plot_shifts_over_frames_norm(args.csv_file_path)


if __name__ == '__main__':
    main()
