import pyvista as pv
import numpy as np
import argparse
import csv
import time
import sys
import os

class CoordinateFrame:
    """
    三维空间坐标系类，用于存储和绘制相对于 Base 的空间位姿。
    """
    def __init__(self, name, transform_matrix=None):
        self.name = name
        if transform_matrix is None:
            self.transform_matrix = np.eye(4)
        else:
            self.transform_matrix = transform_matrix

    def update_pose(self, new_transform_matrix):
        """更新坐标系的位姿矩阵（用于后续动态轨迹）"""
        self.transform_matrix = new_transform_matrix

    def draw(self, plotter, scale=1.0, shaft_radius=0.01):
        """将该坐标系绘制到指定的 PyVista plotter 上。"""
        # X 轴 (红)
        x_arrow = pv.Arrow(start=(0, 0, 0), direction=(1, 0, 0), scale=scale, shaft_radius=shaft_radius)
        x_arrow.transform(self.transform_matrix, inplace=True)
        plotter.add_mesh(x_arrow, color='red', name=f'{self.name}_x')

        # Y 轴 (绿)
        y_arrow = pv.Arrow(start=(0, 0, 0), direction=(0, 1, 0), scale=scale, shaft_radius=shaft_radius)
        y_arrow.transform(self.transform_matrix, inplace=True)
        plotter.add_mesh(y_arrow, color='green', name=f'{self.name}_y')

        # Z 轴 (蓝)
        z_arrow = pv.Arrow(start=(0, 0, 0), direction=(0, 0, 1), scale=scale, shaft_radius=shaft_radius)
        z_arrow.transform(self.transform_matrix, inplace=True)
        plotter.add_mesh(z_arrow, color='blue', name=f'{self.name}_z')

        origin = self.transform_matrix[:3, 3]
        plotter.add_point_labels([origin], [self.name], point_size=10, font_size=16, text_color='white')


def main():
    # 1. 终端参数解析 (接收 CSV 文件路径)
    parser = argparse.ArgumentParser(description="3D Trajectory Visualizer")
    parser.add_argument("--csv_path", type=str, help="输入包含笔迹轨迹的 CSV 文件路径 (x, y, z 格式)")
    parser.add_argument("--speed", type=float, default=0.01, help="绘制动画的刷新延迟时间 (秒)")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"Error: 找不到文件 {args.csv_path}")
        sys.exit(1)

    # 2. 读取 CSV 轨迹数据
    trajectory_in_paper = []
    with open(args.csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue # 跳过空行
            try:
                # 尝试提取前两列 x, y。如果带有表头 (如 'x','y','z')，这里会抛出 ValueError 并被安全忽略
                x, y, z= float(row[0]), float(row[1]), float(row[2])
                trajectory_in_paper.append((x/1000, y/1000))
                # 注意：你提到的 z (下笔/抬笔) 状态目前暂存忽略，后续如果需要实现断笔，可以把 z 也读出来
            except ValueError:
                continue
    
    if not trajectory_in_paper:
        print("Error: CSV 文件中没有读取到有效的轨迹数据。")
        sys.exit(1)

    print(f"成功加载 {len(trajectory_in_paper)} 个轨迹点。开始 3D 渲染...")

    # 3. 初始化渲染器与基础场景
    plotter = pv.Plotter()
    BACKGROUND_COLOR = "#eeeeee"
    plotter.set_background(BACKGROUND_COLOR)

    # Base 坐标系
    base_frame = CoordinateFrame(name="Base_Frame")
    base_frame.draw(plotter, scale=0.15, shaft_radius=0.02)

    # Paper 坐标系变换矩阵
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

    # ---------------- 替换原来的 4 和 5 ----------------

    print("正在计算空间映射...")
    # 1. 同样先提前算好所有的 3D 点
    trajectory_points_3d = []
    for x, y in trajectory_in_paper:
        P_local = np.array([x, y, 0, 1])
        P_base = T_paper @ P_local
        trajectory_points_3d.append(P_base[:3])
        
    points_array = np.array(trajectory_points_3d)

    # 2. 避免空网格报错：用前两个点初始化网格
    trace_mesh = pv.PolyData(points_array[:2])
    trace_mesh.lines = np.hstack([[2], [0, 1]]) # 连接前两个点
    
    plotter.add_mesh(trace_mesh, color='#FF5722', line_width=4, name='dynamic_trace', render_lines_as_tubes=True)

    # 3. 开启非阻塞交互模式
    plotter.show(interactive_update=True)

    # 4. 从第 3 个点开始循环更新画面，产生动画效果
    for i in range(2, len(points_array)):
        # 每次取出前 i+1 个点覆盖进去
        current_points = points_array[:i+1]
        trace_mesh.points = current_points
        trace_mesh.lines = np.hstack([[len(current_points)], np.arange(len(current_points))])
        
        plotter.update()
        time.sleep(args.speed)

    print("绘制完成！")
    plotter.show()


if __name__ == "__main__":
    main()
