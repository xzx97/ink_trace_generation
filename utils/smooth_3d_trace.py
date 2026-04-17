import numpy as np

def generate_bezier_hop(p_start, p_end, z_safe, num_points=20):
    """
    生成三阶贝塞尔曲线，模拟机械臂完美的提笔空中姿态
    """
    P0 = np.array([p_start[0], p_start[1], 0.0])
    P1 = np.array([p_start[0], p_start[1], z_safe]) # 保证垂直起飞
    P2 = np.array([p_end[0],   p_end[1],   z_safe]) # 保证垂直降落
    P3 = np.array([p_end[0],   p_end[1],   0.0])

    t = np.linspace(0, 1, num_points)[:, np.newaxis]
    
    # 三阶贝塞尔公式
    curve = (1-t)**3 * P0 + 3*(1-t)**2 * t * P1 + 3*(1-t)*t**2 * P2 + t**3 * P3
    return curve.tolist()


def generate_even_bezier_hop(p_start, p_end, z_safe, step_size_m=0.001):
    """
    生成三阶贝塞尔曲线，并按真实的弧长进行绝对等距重采样 (消除阶跃)
    
    参数:
    - p_start: 起飞点 [x, y]
    - p_end: 降落点 [x, y]
    - z_safe: 安全抬笔高度 (米)
    - step_size_m: 期望的插值点空间间隔 (默认 0.001m 即 1mm)
    """
    P0 = np.array([p_start[0], p_start[1], 0.0])
    P1 = np.array([p_start[0], p_start[1], z_safe])
    P2 = np.array([p_end[0],   p_end[1],   z_safe])
    P3 = np.array([p_end[0],   p_end[1],   0.0])

    # 1. 估算空间大致总长度，决定“超采样”的密度
    approx_length = np.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1]) + 2 * z_safe
    # 保证每毫米至少有 10 个点作为基准，计算超采样的数量
    dense_N = max(100, int(approx_length / (step_size_m / 10)))

    # 2. 生成极高密度的基础贝塞尔曲线
    t = np.linspace(0, 1, dense_N)[:, np.newaxis]
    dense_curve = (1-t)**3 * P0 + 3*(1-t)**2 * t * P1 + 3*(1-t)*t**2 * P2 + t**3 * P3

    # 3. 计算真实的累积弧长 (里程计)
    # np.diff 计算相邻两点的向量，norm 计算欧氏距离
    segment_lengths = np.linalg.norm(np.diff(dense_curve, axis=0), axis=1)
    cum_lengths = np.insert(np.cumsum(segment_lengths), 0, 0.0) # 累加，起点为0
    total_length = cum_lengths[-1]

    # 4. 根据设定的物理步长 (step_size_m) 决定最终需要多少个点
    num_even_points = max(2, int(total_length / step_size_m))
    # 在 0 到 总长度 之间，生成完美的等差数列
    even_lengths = np.linspace(0, total_length, num_even_points)

    # 5. 分别对 X, Y, Z 进行一维线性插值重采样
    even_curve = np.empty((num_even_points, 3))
    even_curve[:, 0] = np.interp(even_lengths, cum_lengths, dense_curve[:, 0])
    even_curve[:, 1] = np.interp(even_lengths, cum_lengths, dense_curve[:, 1])
    even_curve[:, 2] = np.interp(even_lengths, cum_lengths, dense_curve[:, 2])

    return even_curve.tolist()
