import pyvista as pv
import numpy as np
import argparse
import csv
import time
import sys
import os

class CoordinateFrame:
    """三维空间坐标系类，用于存储和绘制相对于 Base 的空间位姿。"""
    def __init__(self, name, transform_matrix=None):
        self.name = name
        if transform_matrix is None:
            self.transform_matrix = np.eye(4)
        else:
            self.transform_matrix = transform_matrix

    def update_pose(self, new_transform_matrix):
        self.transform_matrix = new_transform_matrix

    def draw(self, plotter, scale=1.0, shaft_radius=0.01):
        x_arrow = pv.Arrow(start=(0, 0, 0), direction=(1, 0, 0), scale=scale, shaft_radius=shaft_radius)
        x_arrow.transform(self.transform_matrix, inplace=True)
        plotter.add_mesh(x_arrow, color='red', name=f'{self.name}_x')

        y_arrow = pv.Arrow(start=(0, 0, 0), direction=(0, 1, 0), scale=scale, shaft_radius=shaft_radius)
        y_arrow.transform(self.transform_matrix, inplace=True)
        plotter.add_mesh(y_arrow, color='green', name=f'{self.name}_y')

        z_arrow = pv.Arrow(start=(0, 0, 0), direction=(0, 0, 1), scale=scale, shaft_radius=shaft_radius)
        z_arrow.transform(self.transform_matrix, inplace=True)
        plotter.add_mesh(z_arrow, color='blue', name=f'{self.name}_z')

        origin = self.transform_matrix[:3, 3]
        plotter.add_point_labels([origin], [self.name], point_size=10, font_size=16, text_color='white')


def main():
    # 1. 终端参数解析 (新增了 --hide_air 参数)
    parser = argparse.ArgumentParser(description="3D Trajectory Visualizer")
    parser.add_argument("--csv_path", type=str, help="输入包含笔迹轨迹的 CSV 文件路径 (x, y, z 格式)")
    parser.add_argument("--speed", type=float, default=0.01, help="绘制动画的刷新延迟时间 (秒)")
    parser.add_argument("--hide_air", action="store_true", help="如果带有此参数，默认隐藏灰色的提笔空中轨迹")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"Error: 找不到文件 {args.csv_path}")
        sys.exit(1)

    # 2. 读取 CSV 轨迹数据
    trajectory_in_paper = []
    z_states = []
    with open(args.csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue 
            try:
                x, y, z = float(row[0]), float(row[1]), float(row[2])
                trajectory_in_paper.append((x/1000, y/1000))
                z_states.append(z)
            except ValueError:
                continue
    
    if not trajectory_in_paper:
        print("Error: CSV 文件中没有读取到有效的轨迹数据。")
        sys.exit(1)

    print(f"成功加载 {len(trajectory_in_paper)} 个轨迹点。开始 3D 渲染...")

    # 3. 初始化渲染器与基础场景
    plotter = pv.Plotter()
    plotter.set_background("#eeeeee")

    base_frame = CoordinateFrame(name="Base_Frame")
    base_frame.draw(plotter, scale=0.15, shaft_radius=0.02)

    theta = np.deg2rad(-90)
    T_paper = np.array([
        [np.cos(theta), -np.sin(theta), 0,  0.331],
        [np.sin(theta),  np.cos(theta), 0, -0.174],
        [            0,              0, 1, -0.12],
        [            0,              0, 0,   1]
    ])
    paper_frame = CoordinateFrame(name="Paper_Origin", transform_matrix=T_paper)
    paper_frame.draw(plotter, scale=0.1, shaft_radius=0.01)

    PAPER_WIDTH = 0.297
    PAPER_HEIGHT = 0.210
    paper_mesh = pv.Plane(center=(PAPER_WIDTH/2, PAPER_HEIGHT/2, 0), i_size=PAPER_WIDTH, j_size=PAPER_HEIGHT)
    paper_mesh.transform(paper_frame.transform_matrix, inplace=True) 
    plotter.add_mesh(paper_mesh, color='white', opacity=0.4, show_edges=True)

    plotter.show_bounds(grid='front', location='outer', all_edges=True)
    plotter.add_axes()

    # ---------------- 核心优化：前置计算拓扑结构与状态 ----------------
    print("正在计算空间映射与笔尖状态...")
    
    trajectory_points_3d = []
    for x, y in trajectory_in_paper:
        P_local = np.array([x, y, 0, 1])
        P_base = T_paper @ P_local
        trajectory_points_3d.append(P_base[:3])
        
    points_array = np.array(trajectory_points_3d)
    num_points = len(points_array)
    num_segments = num_points - 1

    all_lines = np.empty((num_segments, 3), dtype=int)
    all_lines[:, 0] = 2
    all_lines[:, 1] = np.arange(num_segments)
    all_lines[:, 2] = np.arange(1, num_points)
    all_lines_flat = all_lines.flatten()

    # 【重要修改】：使用 float 类型，为后续传入 np.nan 做准备
    all_states = np.zeros(num_segments, dtype=float)
    for k in range(num_segments):
        if z_states[k] > 0.5 and z_states[k+1] > 0.5:
            all_states[k] = 1.0 # 下笔
        else:
            all_states[k] = 0.0 # 提笔

    # 预先准备两套状态数据，极速切换
    states_show_air = all_states.copy()
    states_hide_air = np.where(all_states == 0.0, np.nan, all_states) # 把 0 替换为 NaN

    # ---------------- UI 交互与动画刷新逻辑 ----------------
    
    interactive_state = {
        "show_air": not args.hide_air,
        "current_step": 1  # 记录当前动画画到了哪一段
    }

    trace_mesh = pv.PolyData()

    def update_mesh():
        step = interactive_state["current_step"]
        temp_mesh = pv.PolyData()
        temp_mesh.points = points_array[:step+1]
        temp_mesh.lines = all_lines_flat[: step * 3]
        
        if interactive_state["show_air"]:
            temp_mesh.cell_data["PenState"] = states_show_air[:step]
        else:
            temp_mesh.cell_data["PenState"] = states_hide_air[:step]
            
        trace_mesh.shallow_copy(temp_mesh)

    # 👇 -------- 新增这一行：在加入渲染器前，先强行初始化一次数据！ --------
    update_mesh() 
    # 👆 -----------------------------------------------------------------

    def toggle_air_visibility(flag):
        interactive_state["show_air"] = flag
        update_mesh() 

    plotter.add_checkbox_button_widget(
        toggle_air_visibility, 
        value=interactive_state["show_air"], 
        position=(10, 10), 
        size=30, 
        color_on='gray', 
        color_off='white'
    )
    plotter.add_text("Show Air Trace (Lift Pen)", position=(50, 15), font_size=12, color="black")

    # 现在执行 add_mesh 时，trace_mesh 里已经有 PenState 数组了，完美通过检查
    plotter.add_mesh(
        trace_mesh, 
        scalars="PenState", 
        cmap=['#c0c0c0', '#FF5722'], 
        clim=[-0.1, 1.1], 
        nan_opacity=0.0,  
        line_width=4, 
        render_lines_as_tubes=True,
        show_scalar_bar=False,
        name='dynamic_trace'
    )
    

    plotter.show(interactive_update=True)

    # 执行动画
    for i in range(1, num_segments):
        interactive_state["current_step"] = i
        update_mesh()
        
        plotter.update()
        time.sleep(args.speed)

    print("绘制完成！你可以点击左下角的按钮随时切换轨迹的显示状态。")
    plotter.show()


if __name__ == "__main__":
    main()
