import pyvista as pv
import numpy as np

class CoordinateFrame:
    """
    三维空间坐标系类，用于存储和绘制相对于 Base 的空间位姿。
    """
    def __init__(self, name, transform_matrix=None):
        self.name = name
        # 如果不传入矩阵，默认初始化为单位阵（即与世界/Base坐标系重合）
        if transform_matrix is None:
            self.transform_matrix = np.eye(4)
        else:
            self.transform_matrix = transform_matrix

    def update_pose(self, new_transform_matrix):
        """更新坐标系的位姿矩阵（用于后续动态轨迹）"""
        self.transform_matrix = new_transform_matrix

    def draw(self, plotter, scale=1.0, shaft_radius=0.01):
        """
        将该坐标系绘制到指定的 PyVista plotter 上。
        """
        # X 轴 (红)
        x_arrow = pv.Arrow(start=(0, 0, 0), direction=(1, 0, 0), scale=scale, shaft_radius=shaft_radius)
        x_arrow.transform(self.transform_matrix, inplace=True)
        plotter.add_mesh(x_arrow, color='red', name=f'{self.name}_x')

        # Y 轴 (绿) - 直接指定方向，并只应用全局变换
        y_arrow = pv.Arrow(start=(0, 0, 0), direction=(0, 1, 0), scale=scale, shaft_radius=shaft_radius)
        y_arrow.transform(self.transform_matrix, inplace=True)
        plotter.add_mesh(y_arrow, color='green', name=f'{self.name}_y')

        # Z 轴 (蓝) - 直接指定方向，并只应用全局变换
        z_arrow = pv.Arrow(start=(0, 0, 0), direction=(0, 0, 1), scale=scale, shaft_radius=shaft_radius)
        z_arrow.transform(self.transform_matrix, inplace=True)
        plotter.add_mesh(z_arrow, color='blue', name=f'{self.name}_z')

        # 绘制坐标系原点标签
        origin = self.transform_matrix[:3, 3]
        plotter.add_point_labels([origin], [self.name], point_size=10, font_size=16, text_color='white')


def main():
    plotter = pv.Plotter()
    plotter.set_background("#eeeeee")

    # 1. 实例化 Base 坐标系并绘制
    base_frame = CoordinateFrame(name="Base_Frame")
    base_frame.draw(plotter, scale=0.15, shaft_radius=0.02)

    # 2. 定义纸张相对于 Base 的 4x4 变换矩阵
    theta = np.deg2rad(-90)
    T_paper = np.array([
        [np.cos(theta), -np.sin(theta), 0,  0.331],
        [np.sin(theta),  np.cos(theta), 0, -0.174],
        [            0,              0, 1, -0.12],
        [            0,              0, 0,   1]
    ])

    # 3. 实例化 Paper 坐标系并绘制
    paper_frame = CoordinateFrame(name="Paper_Origin", transform_matrix=T_paper)
    paper_frame.draw(plotter, scale=0.1, shaft_radius=0.01)

    PAPER_WIDTH = 0.297
    PAPER_HEIGHT = 0.210

    # 4. 附加绘制 A4 纸平面 (可选，为了视觉辅助)
    paper_mesh = pv.Plane(center=(PAPER_WIDTH/2, PAPER_HEIGHT/2, 0), i_size=PAPER_WIDTH, j_size=PAPER_HEIGHT)
    # 直接使用 paper_frame 实例中存储的矩阵来对齐纸张
    paper_mesh.transform(paper_frame.transform_matrix, inplace=True) 
    plotter.add_mesh(paper_mesh, color='white', opacity=0.4, show_edges=True)

    plotter.show_bounds(grid='front', location='outer', all_edges=True)
    plotter.add_axes()
    plotter.show()

if __name__ == "__main__":
    main()
