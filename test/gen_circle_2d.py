import numpy as np
import pandas as pd

def generate_3d_circle_csv(x_center, y_center, z_center, radius, num_points=100, filename='circle_trajectory.csv'):
    """
    生成一个 3D 圆轨迹并保存为 CSV。
    
    参数:
    x_center, y_center, z_center: 圆心坐标
    radius: 圆半径
    num_points: 轨迹点的数量（采样率）
    filename: 输出文件名
    """
    # 生成 theta 角，从 0 到 2*pi
    theta = np.linspace(0, 2 * np.pi, num_points)
    
    # 计算圆上各点的坐标
    x = x_center + radius * np.cos(theta)
    y = y_center + radius * np.sin(theta)
    # 在 z 平面，所以 z 坐标为常数
    z = np.full_like(theta, z_center)
    
    # 创建 DataFrame 并保存为 CSV
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'z': z
    })
    
    df.to_csv(filename, index=False)
    print(f"成功生成轨迹文件: {filename}")
    return df

# --- 自定义参数区域 ---
CX, CY, CZ = 0, 0, 10  # 圆心坐标
R = 50                  # 半径
POINTS = 800          # 点数

# 执行生成
df_circle = generate_3d_circle_csv(CX, CY, CZ, R, POINTS)
