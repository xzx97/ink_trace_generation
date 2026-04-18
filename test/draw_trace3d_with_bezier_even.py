## Bezier 差值 优化

import pyvista as pv
import numpy as np
import argparse
import csv
import time
import sys
import os
from utils.smooth_2d_strokes import smooth_stroke_2d, resample_stroke_2d
from utils.smooth_3d_trace import generate_even_bezier_hop

class CoordinateFrame:
    """三维空间坐标系类"""
    def __init__(self, name, transform_matrix=None):
        self.name = name
        self.transform_matrix = transform_matrix if transform_matrix is not None else np.eye(4)

    def draw(self, plotter, scale=1.0, shaft_radius=0.01):
        for i, (color, axis) in enumerate(zip(['red', 'green', 'blue'], ['x', 'y', 'z'])):
            direction = [1 if j == i else 0 for j in range(3)]
            arrow = pv.Arrow(start=(0, 0, 0), direction=direction, scale=scale, shaft_radius=shaft_radius)
            arrow.transform(self.transform_matrix, inplace=True)
            plotter.add_mesh(arrow, color=color, name=f'{self.name}_{axis}')
        origin = self.transform_matrix[:3, 3]
        plotter.add_point_labels([origin], [self.name], point_size=10, font_size=16, text_color='white')

def main():
    parser = argparse.ArgumentParser(description="3D Trajectory Visualizer with Spatial Interpolation")
    parser.add_argument("--csv_path", type=str, help="输入包含笔迹轨迹的 CSV 文件路径")
    parser.add_argument("--speed", type=float, default=0.01, help="绘制动画刷新延迟")
    parser.add_argument("--hide_air", action="store_true", help="默认隐藏空中轨迹")
    parser.add_argument("--smooth_window", type=int, default=11, help="2D 平滑窗口大小")
    parser.add_argument("--lift_height", type=float, default=10.0, help="抬笔安全高度 (mm)")
    parser.add_argument("--hop_res", type=float, default=1.0, help="空中插值轨迹的空间分辨率/步长 (单位: mm)")
    # --- 新增：墨迹重采样参数 ---
    parser.add_argument("--trace_step", type=float, default=2.0, help="墨迹轨迹的空间等距采样步长 (mm)")
    parser.add_argument("--output_file", type=str, default=None, help="如果提供，将把最终的 3D 轨迹 (Base坐标系) 导出到此 CSV 文件")

    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        sys.exit(f"Error: 找不到文件 {args.csv_path}")

    # 1. 结构化读取：将连续的下笔点组合成独立的“笔画 (strokes)”
    strokes = []
    current_stroke = []
    
    with open(args.csv_path, 'r', encoding='utf-8') as f:
        for row in csv.reader(f):
            if not row: continue 
            try:
                x, y, z_csv = float(row[0])/1000.0, float(row[1])/1000.0, float(row[2])
                if z_csv > 0.5: # 落笔状态
                    current_stroke.append((x, y))
                else:           # 抬笔状态，截断当前笔画
                    if current_stroke:
                        strokes.append(current_stroke)
                        current_stroke = []
            except ValueError:
                continue
    if current_stroke: strokes.append(current_stroke) # 保存最后一笔

    print(f"成功解析，共提取到 {len(strokes)} 个独立的实线笔画。")

    # 3.1 对每个独立笔画进行 2D XY 平滑 (去除毛刺)
    if args.smooth_window > 0:
        strokes = [smooth_stroke_2d(s, window_length=args.smooth_window) for s in strokes]
        print("✅ 笔画 2D 平滑处理完成。")

    # 3.2 --- 新增：对平滑后的笔画进行等距重采样 (控制点密度) ---
    if args.trace_step > 0:
        trace_step_m = args.trace_step / 1000.0
        strokes = [resample_stroke_2d(s, step_size_m=trace_step_m) for s in strokes]
        print(f"✅ 笔画等距重采样完成，墨迹点距被强制控制在约 {args.trace_step} mm。")


    # 3. 核心：构建包含平滑落笔和 3D 空中贝塞尔插值的完整空间轨迹
    z_safe_m = args.lift_height / 1000.0
    full_local_trajectory = [] # [x, y, z]
    point_states = []          # 1: 墨迹, 0: 空中
    
    for i, stroke in enumerate(strokes):
        # 写入当前真实笔画 (Z = 0)
        for pt in stroke:
            full_local_trajectory.append([pt[0], pt[1], 0.0])
            point_states.append(1.0)
            
            # 如果不是最后一笔，生成飞向下一笔的空中插值轨迹
        if i < len(strokes) - 1:
            p_end = stroke[-1]
            p_next_start = strokes[i+1][0]
            
            # 把毫米转换为米传进去
            hop_res_m = args.hop_res / 1000.0 
            
            # --- 使用全新的等距重采样函数 ---
            hop_curve = generate_even_bezier_hop(p_end, p_next_start, z_safe_m, step_size_m=hop_res_m)
            
            # 掐头去尾，避免与真实笔画端点坐标重叠重复
            for pt in hop_curve[1:-1]:
                full_local_trajectory.append(pt)
                point_states.append(0.0)

    # 4. 全局矩阵映射
    theta = np.deg2rad(90)
    T_paper = np.array([
        [np.cos(theta),  0, -np.sin(theta),  0.288],
        [0,              1,  0            ,  0.330],
        [np.sin(theta),  0,  np.cos(theta),  0.277],
        [            0,              0, 0,   1]
    ])
    # T_paper = np.array([
    #     [np.cos(theta), -np.sin(theta), 0,  0.327],
    #     [np.sin(theta),  np.cos(theta), 0, -0.352],
    #     [            0,              0, 1, -0.403],
    #     [            0,              0, 0,   1]
    # ])

    trajectory_points_3d = []
    for pt in full_local_trajectory:
        P_local = np.array([pt[0], pt[1], pt[2], 1])
        P_base = T_paper @ P_local
        trajectory_points_3d.append(P_base[:3])
        
    points_array = np.array(trajectory_points_3d)
    num_points = len(points_array)
    num_segments = num_points - 1

    # --- 新增：保存轨迹至 CSV 文件模块 ---
    if args.output_file:
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(args.output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with open(args.output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # 写入表头：在机器人工程中，带有清晰表头的数据极大地减少了联调沟通成本
                writer.writerow(['x', 'y', 'z'])
                
                # 遍历所有 3D 点及其对应的绘画状态
                for pt, state in zip(trajectory_points_3d, point_states):
                    # 保留 6 位小数 (微米级精度对于大部分机械臂完全足够，避免文件过大)
                    writer.writerow([f"{pt[0]*1000:.4f}", f"{pt[1]*1000:.4f}", f"{pt[2]*1000:.4f}"])
            
            print(f"✅ 成功将 {len(trajectory_points_3d)} 个 3D 轨迹点保存至: {args.output_file}")
        except Exception as e:
            print(f"❌ 保存轨迹文件时发生错误: {e}")
    # -----------------------------------

    # 拓扑连线与状态判断
    all_lines = np.empty((num_segments, 3), dtype=int)
    all_lines[:, 0] = 2
    all_lines[:, 1] = np.arange(num_segments)
    all_lines[:, 2] = np.arange(1, num_points)
    all_lines_flat = all_lines.flatten()

    all_states = np.zeros(num_segments, dtype=float)
    for k in range(num_segments):
        # 只有线段两端都是落笔点，这条线才是实线墨迹
        if point_states[k] == 1.0 and point_states[k+1] == 1.0:
            all_states[k] = 1.0 

    states_show_air = all_states.copy()
    states_hide_air = np.where(all_states == 0.0, np.nan, all_states)

    # 5. 渲染部分 (与之前一致，使用 UI 和动画)
    plotter = pv.Plotter()
    plotter.set_background("#eeeeee")
    CoordinateFrame(name="Base_Frame").draw(plotter, scale=0.15, shaft_radius=0.02)
    paper_frame = CoordinateFrame(name="Paper_Origin", transform_matrix=T_paper)
    paper_frame.draw(plotter, scale=0.1, shaft_radius=0.01)

    paper_mesh = pv.Plane(center=(0.297/2, 0.210/2, 0), i_size=0.297, j_size=0.210)
    paper_mesh.transform(paper_frame.transform_matrix, inplace=True) 
    plotter.add_mesh(paper_mesh, color='white', opacity=0.4, show_edges=True)
    plotter.show_bounds(grid='front', location='outer', all_edges=True)
    plotter.add_axes()

    interactive_state = {"show_air": not args.hide_air, "current_step": 1}
    trace_mesh = pv.PolyData()

    def update_mesh():
        step = interactive_state["current_step"]
        temp_mesh = pv.PolyData()
        temp_mesh.points = points_array[:step+1]
        temp_mesh.lines = all_lines_flat[: step * 3]
        temp_mesh.cell_data["PenState"] = states_show_air[:step] if interactive_state["show_air"] else states_hide_air[:step]
        trace_mesh.shallow_copy(temp_mesh)

    update_mesh()

    def toggle_air_visibility(flag):
        interactive_state["show_air"] = flag
        update_mesh() 

    plotter.add_checkbox_button_widget(toggle_air_visibility, value=interactive_state["show_air"], position=(10, 10), size=30, color_on='gray', color_off='white')
    plotter.add_text("Show Spatial Bezier Trace", position=(50, 15), font_size=12, color="black")

    plotter.add_mesh(trace_mesh, scalars="PenState", cmap=['#c0c0c0', '#FF5722'], clim=[-0.1, 1.1], nan_opacity=0.0, line_width=3, render_lines_as_tubes=True, show_scalar_bar=False)

    plotter.show(interactive_update=True)
    for i in range(1, num_segments):
        interactive_state["current_step"] = i
        update_mesh()
        plotter.update()
        time.sleep(args.speed)

    print("渲染完毕！现在转动视角，看看那些完美的空中抛物线吧！")
    plotter.show()

if __name__ == "__main__":
    main()

