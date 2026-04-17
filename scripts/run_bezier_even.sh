#!/usr/bin/bash

uv run python3 ./test/draw_trace3d_with_bezier_even.py --csv_path ./data/trajectory_04.csv --smooth_window 21 --lift_height 20 --speed 0.001 --hop_res 1.0

