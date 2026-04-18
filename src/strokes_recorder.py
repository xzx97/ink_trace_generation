import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import sys

class TraceRecorder:
    def __init__(self, image_path, output_csv, scale_x=1.0, scale_y=1.0):
        # 1. 设定 A4 横放物理比例 (单位：mm)
        self.width, self.height = 297, 210
        self.output_csv = output_csv
        
        # 2. 加载图片
        raw_img = cv2.imread(image_path)
        if raw_img is None:
            print(f"Error: Could not find image at {image_path}")
            sys.exit(1)
        self.image = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        
        # 3. 计算缩放后的图片显示范围 (extent)
        # 默认图片占据全屏，应用 scale 后会缩小并居中
        img_w = self.width * scale_x
        img_h = self.height * scale_y
        
        # 居中偏移量
        off_x = (self.width - img_w) / 2
        off_y = (self.height - img_h) / 2
        
        # extent=[左, 右, 下, 上] 
        # 为了符合左上角原点且 Y 向下，设置 [off_x, off_x + img_w, off_y + img_h, off_y]
        self.img_extent = [off_x, off_x + img_w, off_y, off_y + img_h]
        
        self.points = [] 
        self.is_drawing = False

        # 4. 创建画布
        self.fig, self.ax = plt.subplots(figsize=(10, 7)) 
        self.ax.imshow(self.image, extent=self.img_extent, aspect='auto', alpha=0.6)
        
        # 5. 坐标轴设置：左上角 (0,0)，横向 A4
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        # self.ax.invert_yaxis() 
        
        self.ax.set_title(f"A4 Landscape (Scale: x={scale_x}, y={scale_y})\nOrigin: Bot-Left (0,0)")
        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")

        # 绑定鼠标事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)

    def on_press(self, event):
        if event.inaxes:
            self.is_drawing = True
            self.add_point(event.xdata, event.ydata, 1)

    def on_move(self, event):
        if self.is_drawing and event.inaxes:
            self.add_point(event.xdata, event.ydata, 1)
            self.ax.plot(event.xdata, event.ydata, 'ro', markersize=1)
            self.fig.canvas.draw_idle()

    def on_release(self, event):
        if self.is_drawing:
            if event.xdata is not None and event.ydata is not None:
                self.add_point(event.xdata, event.ydata, 0)
            self.is_drawing = False

    def add_point(self, x, y, z):
        self.points.append([round(float(x), 2), round(float(y), 2), int(z)])

    def save_csv(self):
        if not self.points:
            print("No points recorded.")
            return
        df = pd.DataFrame(self.points, columns=['x', 'y', 'z'])
        df.to_csv(self.output_csv, index=False)
        print(f"\n--- SUCCESS ---")
        print(f"Recorded {len(self.points)} points at scale X:{self.img_extent[1]-self.img_extent[0]}mm")
        print(f"File saved: {self.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A4 Trace Recorder with Scaling")
    parser.add_argument("--img", type=str, required=True, help="Path to image")
    parser.add_argument("--out", type=str, default="trajectory.csv", help="Output CSV path")
    parser.add_argument("--sx", type=float, default=1.0, help="Scale factor for X axis (0.1 to 1.0)")
    parser.add_argument("--sy", type=float, default=1.0, help="Scale factor for Y axis (0.1 to 1.0)")

    args = parser.parse_args()

    recorder = TraceRecorder(args.img, args.out, scale_x=args.sx, scale_y=args.sy)
    plt.show()
    recorder.save_csv()
