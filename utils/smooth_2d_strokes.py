import numpy as np

def resample_stroke_2d(stroke_points, step_size_m=0.002):
    """
    将 2D 笔画按指定的空间绝对距离（弧长）进行等距重采样
    解决鼠标采样密集或稀疏的问题，强制点位均匀分布
    """
    pts = np.array(stroke_points)
    
    # 滤除异常数据
    if len(pts) < 2:
        return stroke_points
        
    # 计算每一小段的长度和累积长度（物理里程计）
    segment_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum_lengths = np.insert(np.cumsum(segment_lengths), 0, 0.0)
    total_length = cum_lengths[-1]
    
    # 如果这整个笔画的长度甚至不到我们设定的一个步长，保留头尾即可
    if total_length < step_size_m or total_length == 0:
        return [stroke_points[0], stroke_points[-1]]
        
    # 根据总长度和步长，计算需要多少个均匀的点
    num_even_points = max(2, int(total_length / step_size_m))
    
    # 生成完美的等差数列里程点
    even_lengths = np.linspace(0, total_length, num_even_points)
    
    # 对 X 和 Y 坐标分别基于里程点进行一维线性插值
    resampled = np.empty((num_even_points, 2))
    resampled[:, 0] = np.interp(even_lengths, cum_lengths, pts[:, 0])
    resampled[:, 1] = np.interp(even_lengths, cum_lengths, pts[:, 1])
    
    return resampled.tolist()

def smooth_stroke_2d(stroke_points, window_length=11, polyorder=3):
    """
    对单条独立笔画的 2D 坐标进行 Savitzky-Golay 平滑
    融合了严谨的边界校验，专门适配切分好的笔画列表 (List of Lists)
    """
    try:
        from scipy.signal import savgol_filter
    except ImportError:
        print("警告: 未找到 scipy 库，跳过平滑处理。")
        return stroke_points

    # 1. 极其严谨的算法参数自动修正 (沿用你的第一版逻辑)
    if window_length % 2 == 0: 
        window_length += 1
    if window_length <= polyorder: 
        window_length = polyorder + 2
        if window_length % 2 == 0: 
            window_length += 1

    # 2. 短笔画保护：如果这一个笔画的点数比滤波窗口还要少，强行滤波会报错
    # 所以对于太短的点集（比如只是拿笔点了一下），直接原样返回，不作处理
    if len(stroke_points) < window_length:
        return stroke_points

    # 3. 执行平滑
    pts = np.array(stroke_points)
    smoothed = np.empty_like(pts)
    smoothed[:, 0] = savgol_filter(pts[:, 0], window_length, polyorder)
    smoothed[:, 1] = savgol_filter(pts[:, 1], window_length, polyorder)
    
    return smoothed.tolist()


# --- 新增：平滑处理模块 ---
def smooth_strokes(trajectory_2d, z_states, window_length=11, polyorder=3):
    """
    按独立笔画对 2D 轨迹进行 Savitzky-Golay 平滑过滤
    
    参数:
    - trajectory_2d: [(x, y), ...] 原始 2D 坐标列表
    - z_states: [z, ...] 对应的下笔状态 (z > 0.5 表示下笔)
    - window_length: 滤波窗口大小，必须是奇数。越大越平滑，但也可能丢失微小细节
    - polyorder: 局部多项式拟合的阶数。通常 2 或 3 效果最好
    """
    try:
        from scipy.signal import savgol_filter
    except ImportError:
        print("警告: 未找到 scipy 库，跳过平滑处理。请执行: uv pip install scipy")
        return trajectory_2d

    # 算法强制要求 window_length 必须是正奇数，且严格大于 polyorder
    if window_length % 2 == 0:
        window_length += 1
    if window_length <= polyorder:
        window_length = polyorder + 2
        if window_length % 2 == 0: window_length += 1

    points = np.array(trajectory_2d)
    smoothed_points = points.copy() # 复制一份，以免改变原数据
    
    start_idx = -1
    
    # 遍历所有点，像状态机一样寻找每一段独立的笔画
    for i in range(len(z_states)):
        is_down = z_states[i] > 0.5
        
        if is_down and start_idx == -1:
            # 笔尖落下，记录当前笔画的起点
            start_idx = i
        elif not is_down and start_idx != -1:
            # 笔尖抬起，结算这一段连续笔画
            end_idx = i
            stroke_len = end_idx - start_idx
            
            # 只有当这一笔的采样点数量大于滤波窗口时，才进行平滑，否则原样保留
            if stroke_len >= window_length:
                smoothed_points[start_idx:end_idx, 0] = savgol_filter(points[start_idx:end_idx, 0], window_length, polyorder)
                smoothed_points[start_idx:end_idx, 1] = savgol_filter(points[start_idx:end_idx, 1], window_length, polyorder)
            
            start_idx = -1 # 重置状态，等待下一笔
            
    # 收尾处理：如果最后一段笔画一直下笔到文件末尾，也要进行结算
    if start_idx != -1:
        end_idx = len(z_states)
        stroke_len = end_idx - start_idx
        if stroke_len >= window_length:
            smoothed_points[start_idx:end_idx, 0] = savgol_filter(points[start_idx:end_idx, 0], window_length, polyorder)
            smoothed_points[start_idx:end_idx, 1] = savgol_filter(points[start_idx:end_idx, 1], window_length, polyorder)
            
    return smoothed_points.tolist()
# ------------------------
