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
    parser = argparse.ArgumentParser(description="3D Trajectory Visualizer")
    parser.add_argument("--csv_path", type=str, help="输入包含笔迹轨迹的 CSV 文件路径 (x, y, z 格式)")
    parser.add_argument("--speed", type=float, default=0.01, help="绘制动画的刷新延迟时间 (秒)")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"Error: 找不到文件 {args.csv_path}")
        sys.exit(1)

    # 2. 读取 CSV 轨迹数据 (同时捕获 z 轴状态)
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
    
    # 将所有的 2D 点一次性转换为 3D Base 坐标系点
    trajectory_points_3d = []
    for x, y in trajectory_in_paper:
        P_local = np.array([x, y, 0, 1])
        P_base = T_paper @ P_local
        trajectory_points_3d.append(P_base[:3])
        
    points_array = np.array(trajectory_points_3d)
    num_points = len(points_array)
    num_segments = num_points - 1

    # 预先构建所有【2点线段】的拓扑数组。格式为: [2, 0, 1,   2, 1, 2,   2, 2, 3 ...]
    # 这比 np.hstack 循环拼接要快几个数量级
    all_lines = np.empty((num_segments, 3), dtype=int)
    all_lines[:, 0] = 2
    all_lines[:, 1] = np.arange(num_segments)
    all_lines[:, 2] = np.arange(1, num_points)
    all_lines_flat = all_lines.flatten()

    # 预先计算每一截线段的物理状态：只有当前点和下一个点都是下笔(z>0.5)时，这段线才是墨迹(1)
    all_states = np.zeros(num_segments, dtype=int)
    for k in range(num_segments):
        if z_states[k] > 0.5 and z_states[k+1] > 0.5:
            all_states[k] = 1
        else:
            all_states[k] = 0

    # ---------------- 动画绘制逻辑 ----------------
    
    # 1. 创建一个空的基底对象，负责一直挂载在渲染器上
    trace_mesh = pv.PolyData() 

    # 2. 初始化第 1 段线的状态（使用干净的临时网格赋值，避免自动生成顶点）
    temp_mesh = pv.PolyData()
    temp_mesh.points = points_array[:2]
    temp_mesh.lines = all_lines_flat[:3]
    temp_mesh.cell_data["PenState"] = all_states[:1]
    
    # 将临时网格的数据指针“瞬间”替换给主网格
    trace_mesh.shallow_copy(temp_mesh)
    
    # 巧妙利用 cmap，给不同状态自动上色：
    # 0 -> 浅灰色 (提笔空中运动)
    # 1 -> 亮橙色 (下笔真实墨迹)
    plotter.add_mesh(
        trace_mesh, 
        scalars="PenState", 
        cmap=['#c0c0c0', '#FF5722'], 
        clim=[-0.1, 1.1], 
        line_width=4, 
        render_lines_as_tubes=True,
        show_scalar_bar=# ---------------- 动画绘制逻辑 ----------------
    
    # 1. 创建一个空的基底对象，负责一直挂载在渲染器上
    trace_mesh = pv.PolyData() 

    # 2. 初始化第 1 段线的状态（使用干净的临时网格赋值，避免自动生成顶点）
    temp_mesh = pv.PolyData()
    temp_mesh.points = points_array[:2]
    temp_mesh.lines = all_lines_flat[:3]
    temp_mesh.cell_data["PenState"] = all_states[:1]
    
    # 将临时网格的数据指针“瞬间”替换给主网格
    trace_mesh.shallow_copy(temp_mesh)
    
    # 巧妙利用 cmap，给不同状态自动上色：
    # 0 -> 浅灰色 (提笔空中运动)
    # 1 -> 亮橙色 (下笔真实墨迹)
    plotter.add_mesh(
        trace_mesh, 
        scalars="PenState", 
        cmap=['#c0c0c0', '#FF5722'], 
        clim=[-0.1, 1.1], 
        line_width=4, 
        render_lines_as_tubes=True,
        show_scalar_bar=False,  # 隐藏颜色条，保持画面干净
        name='dynamic_trace'
    )

    plotter.show(interactive_update=True)

    # 从第 2 个线段开始循环刷新画面
    for i in range(1, num_segments):
        # 每次循环都创建一个临时的干净网格装载新切片的数据
        temp_mesh = pv.PolyData()
        temp_mesh.points = points_array[:i+1]
        temp_mesh.lines = all_lines_flat[: i * 3]
        temp_mesh.cell_data["PenState"] = all_states[:i]
        
        # 核心修复：使用 shallow_copy 无缝替换主网格的拓扑结构和数据
        trace_mesh.shallow_copy(temp_mesh)
        
        plotter.update()
        time.sleep(args.speed)

    print("绘制完成！灰色线代表提笔空中位移，橙色代表实际下笔轨迹。")
    plotter.show()False,  # 隐藏颜色条，保持画面干净
        name='dynamic_trace'
    )

    plotter.show(interactive_update=True)

    # 从第 2 个线段开始循环刷新画面
    for i in range(1, num_segments):
        # 每次循环都创建一个临时的干净网格装载新切片的数据
        temp_mesh = pv.PolyData()
        temp_mesh.points = points_array[:i+1]
        temp_mesh.lines = all_lines_flat[: i * 3]
        temp_mesh.cell_data["PenState"] = all_states[:i]
        
        # 核心修复：使用 shallow_copy 无缝替换主网格的拓扑结构和数据
        trace_mesh.shallow_copy(temp_mesh)
        
        plotter.update()
        time.sleep(args.speed)

    print("绘制完成！灰色线代表提笔空中位移，橙色代表实际下笔轨迹。")
    plotter.show()


if __name__ == "__main__":
    main()
