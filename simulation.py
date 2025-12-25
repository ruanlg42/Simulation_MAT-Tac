import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2

import os

# 设置全局字体为 Arial
plt.rcParams['font.family'] = 'Arial'

class TactileSensorSim:
    """
    视触觉传感器光场与自适应阈值仿真类
    """
    def __init__(self):
        # --- 几何参数 (单位: mm) ---
        self.width = 40.0             # 底面宽度 (正方形)
        self.height = 30.0            # LED平面到底面的距离
        self.radius_led = 15.0        # LED分布圆半径 (近似放置在上方)
        
        # --- 光源参数 ---
        self.num_leds = 9             # LED数量
        self.led_fov = 120            # LED视场角 (度)
        self.led_power = 1000.0       # LED基础光强系数 (相对值)
        self.led_tilt_angle = 0       # LED倾斜角
        
        # --- 图像仿真参数 ---
        self.resolution = 0.02        # 仿真网格分辨率 (mm/pixel)
        self.noise_sigma = 2.0        # 图像随机噪声标准差 (模拟传感器噪声)
        self.output_dir = 'results'   # 输出目录
        
        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # --- 自适应阈值参数 ---
        self.block_size = 251         # 邻域大小
        self.C = 5                    # 阈值常数
        
        # --- 标记点 (Marker) 参数 ---
        # 如果是100个点，分布在40x40mm区域，大概是10x10的网格
        self.marker_pitch = 4.0       # 标记点间距 (mm) 40/10=4
        self.marker_radius = 0.5      # 标记点半径 (mm)
        self.bg_reflectivity = 1.0    # 背景材料反射率
        self.marker_reflectivity = 0.4 # 标记点反射率
        
        # 内部变量
        self.X = None
        self.Y = None
        self.light_map = None         # 纯光照分布 (背景)
        self.raw_image = None         # 模拟的相机原始图像 (含标记点)
        self.threshold_map = None
        self.binary_map = None

    def generate_light_field(self):
        """
        生成底面光照强度分布，并叠加标记点以模拟真实图像
        """
        # 1. 创建网格
        grid_size = int(self.width / self.resolution)
        x = np.linspace(-self.width/2, self.width/2, grid_size)
        y = np.linspace(-self.width/2, self.width/2, grid_size)
        self.X, self.Y = np.meshgrid(x, y)
        
        # 初始化光照图
        self.light_map = np.zeros_like(self.X)
        
        # 2. 计算LED参数 m 值 (根据FOV)
        # 朗伯辐射模式近似: I = I0 * cos(theta)^m / d^2
        alpha = np.deg2rad(self.led_fov / 2.0)
        m = -np.log(2) / np.log(np.cos(alpha))
        
        print(f"LED Simulation: m={m:.2f} for FOV={self.led_fov}°")

        # 3. 叠加每个LED的光照
        for i in range(self.num_leds):
            # 计算当前LED的位置 (极坐标转直角坐标)
            theta_led = 2 * np.pi * i / self.num_leds
            lx = self.radius_led * np.cos(theta_led)
            ly = self.radius_led * np.sin(theta_led)
            lz = self.height
            
            # LED 主光轴方向向量 (垂直向下)
            dir_vec = np.array([0, 0, -1])
            
            # 计算网格点到LED的向量
            dx = self.X - lx
            dy = self.Y - ly
            dz = 0 - lz      # 底面 z=0
            
            dist_sq = dx**2 + dy**2 + dz**2
            dist = np.sqrt(dist_sq)
            
            # 计算入射角/发射角的余弦值
            dot_product = dx*dir_vec[0] + dy*dir_vec[1] + dz*dir_vec[2]
            cos_theta = dot_product / (dist * np.linalg.norm(dir_vec))
            
            # 只保留 cos_theta > 0 的部分 (即在LED前方)
            cos_theta = np.maximum(cos_theta, 0)
            
            # 计算光强: I = P * cos(theta)^m / dist^2
            intensity = self.led_power * (cos_theta ** m) / dist_sq
            
            self.light_map += intensity

        # 4. 模拟标记点 (Markers)
        # 生成标记点中心坐标 (10x10网格)
        # 为了正好放下10个点，间距调整，起始位置调整
        # 范围 -20 到 20. 
        # 10个点: linspace(-18, 18, 10) -> 间距 4mm
        marker_lin = np.linspace(-self.width/2 + self.marker_pitch/2, self.width/2 - self.marker_pitch/2, 10)
        marker_X, marker_Y = np.meshgrid(marker_lin, marker_lin)
        marker_points = np.column_stack([marker_X.ravel(), marker_Y.ravel()])
        
        # 图像尺寸
        h, w = self.light_map.shape
        
        # 临时用 OpenCV 画圆
        # 使用 bg_reflectivity 初始化背景
        img_temp = np.full((h, w), self.bg_reflectivity, dtype=np.float32)
        
        center_pixel_x = w // 2
        center_pixel_y = h // 2
        pixels_per_mm = 1.0 / self.resolution
        marker_radius_px = int(self.marker_radius * pixels_per_mm)
        
        for px, py in marker_points:
            # 转换到图像坐标
            ix = int(px * pixels_per_mm) + center_pixel_x
            iy = int(py * pixels_per_mm) + center_pixel_y
            
            if 0 <= ix < w and 0 <= iy < h:
                # 矩形区域内直接画
                 cv2.circle(img_temp, (ix, iy), marker_radius_px, self.marker_reflectivity, -1) 
        
        self.reflectivity_map = img_temp
        
        # 5. 生成最终模拟图像 = 光照 * 反射率
        self.raw_image = self.light_map * self.reflectivity_map

        # 6. 归一化到 0-255
        max_val = np.max(self.light_map)
        if max_val > 0:
            scale = 255.0 / max_val
            self.raw_image_norm = self.raw_image * scale
            self.light_map_norm = self.light_map * scale
        else:
            self.raw_image_norm = self.raw_image
            self.light_map_norm = self.light_map
            
        # 7. 添加随机噪声 (高斯噪声)
        if self.noise_sigma > 0:
            noise = np.random.normal(0, self.noise_sigma, self.raw_image_norm.shape)
            self.raw_image_norm = self.raw_image_norm + noise
            # 确保不越界
            self.raw_image_norm = np.clip(self.raw_image_norm, 0, 255)

        self.raw_image_uint8 = np.clip(self.raw_image_norm, 0, 255).astype(np.uint8)
        self.light_map_uint8 = np.clip(self.light_map_norm, 0, 255).astype(np.uint8)
        
        # 不需要圆形遮罩了，本身就是矩形区域


    def calculate_adaptive_threshold(self):
        """
        计算自适应阈值面并进行二值化
        """
        if self.raw_image is None:
            self.generate_light_field()
            
        # 使用 OpenCV 的高斯模糊来模拟自适应阈值的“局部加权平均”部分
        # adaptiveThreshold 函数内部就是: src - (Gaussian(src) - C)
        # 这里我们想可视化阈值面本身，即 T = Gaussian(src) - C
        
        # 必须是奇数
        if self.block_size % 2 == 0:
            self.block_size += 1
            
        # 计算局部加权平均 (Mean map)
        # 对含有标记点的原图进行模糊，得到背景估计
        # 注意：通常为了更好的背景估计，会使用中值滤波或大核高斯，或者先膨胀消除黑点
        # 但标准的 adaptiveThreshold 就是直接对原图模糊
        mean_map = cv2.GaussianBlur(self.raw_image_norm, (self.block_size, self.block_size), 0)
        
        # 计算阈值面
        self.threshold_map = mean_map - self.C
        
        # 二值化
        # dst(x,y) = 255 if src(x,y) > T(x,y) else 0
        # 但标记点是黑的，所以我们想提取标记点，应该是 src < T
        # 或者：通常二值化是把前景变白。如果标记点是前景（黑），背景是白
        # cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY 默认是 src > T 为 255
        # 我们这里手动计算：
        
        # 标记点提取 (黑点变白)
        self.binary_map = np.zeros_like(self.raw_image_norm, dtype=np.uint8)
        self.binary_map[self.raw_image_norm < self.threshold_map] = 255
        
        # 遮罩处理
        # mask = (self.X**2 + self.Y**2) > self.radius_base**2
        # self.threshold_map[mask] = 0
        # self.binary_map[mask] = 0
        
        # --- 统计标记点数据 ---
        # 使用连通域分析统计提取到的标记点数量和面积
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.binary_map, connectivity=8)
        
        # 过滤掉背景 (label 0) 和过小的噪点
        # stats: [left, top, width, height, area]
        self.marker_stats = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # 简单的面积过滤，假设标记点至少有 5 个像素
            if area > 5:
                self.marker_stats.append(stats[i])
        
        print(f"Detected Markers: {len(self.marker_stats)}")

    def visualize(self, save_img=True):
        """
        可视化结果
        """
        if self.raw_image is None:
            self.generate_light_field()
        if self.threshold_map is None:
            self.calculate_adaptive_threshold()
            
        # --- Figure 1: 2D 分析面板 (4合1) ---
        fig1 = plt.figure(figsize=(15, 4))
        fig1.suptitle('Adaptive Thresholding Process Analysis', fontsize=14)
        
        # 1. 模拟原始图像 (Raw Image)
        ax1 = fig1.add_subplot(1, 4, 1)
        im1 = ax1.imshow(self.raw_image_norm, cmap='gray', vmin=0, vmax=255, 
                         extent=[-self.width/2, self.width/2, -self.width/2, self.width/2])
        ax1.set_title('1. Simulated Raw Image\n(Uneven Illumination)')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Pixel Value')
        
        # 2. 自适应阈值面 (Adaptive Threshold Map)
        ax2 = fig1.add_subplot(1, 4, 2)
        im2 = ax2.imshow(self.threshold_map, cmap='viridis', vmin=0, vmax=255,
                         extent=[-self.width/2, self.width/2, -self.width/2, self.width/2])
        ax2.set_title('2. Adaptive Threshold Surface\n(T = Gaussian - C)')
        ax2.set_xlabel('X (mm)')
        fig1.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Threshold Value')
        
        # 3. 二值化结果 (Binary Result)
        ax3 = fig1.add_subplot(1, 4, 3)
        im3 = ax3.imshow(self.binary_map, cmap='gray', vmin=0, vmax=255,
                         extent=[-self.width/2, self.width/2, -self.width/2, self.width/2])
        ax3.set_title(f'3. Binary Result\n(Extracted Markers: {len(self.marker_stats)})')
        ax3.set_xlabel('X (mm)')
        fig1.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Binary Value')
        
        # 4. 截面波形对比 (Cross Section)
        ax4 = fig1.add_subplot(1, 4, 4)
        # 找到最接近 y=0 的行索引，确保截面穿过标记点中心
        mid_idx = np.abs(self.Y[:, 0] - 0).argmin()
        
        # 提取中间行数据
        x_line = self.X[mid_idx, :]
        raw_line = self.raw_image_norm[mid_idx, :]
        thresh_line = self.threshold_map[mid_idx, :]
        
        ax4.plot(x_line, raw_line, 'k-', alpha=0.7, label='Raw Signal')
        ax4.plot(x_line, thresh_line, 'r--', linewidth=2, label='Threshold')
        
        # 填充被选中的区域 (低于阈值的部分)
        ax4.fill_between(x_line, 0, raw_line, where=(raw_line < thresh_line), 
                         color='cyan', alpha=0.5, label='Extracted Markers')
        
        ax4.set_title('4. Cross Section Analysis\n(Signal vs Threshold)')
        ax4.set_xlabel('X (mm)')
        ax4.set_ylabel('Pixel Value')
        ax4.legend(loc='lower center', fontsize='small')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_img:
            save_path = os.path.join(self.output_dir, 'simulation_result_2d.png')
            plt.savefig(save_path)
            print(f"Result saved to {save_path}")

        # --- Figure 2: 3D 可视化面板 ---
        fig2 = plt.figure(figsize=(12, 5))
        fig2.suptitle('3D Visualization Panel', fontsize=14)

        # 1. 3D 传感器几何模型与光锥
        ax4 = fig2.add_subplot(1, 2, 1, projection='3d')
        ax4.set_title('3D Sensor Model & Light Cones')
        
        # 重新计算 led_x, led_y 用于 3D 绘图
        led_x = [self.radius_led * np.cos(2*np.pi*i/self.num_leds) for i in range(self.num_leds)]
        led_y = [self.radius_led * np.sin(2*np.pi*i/self.num_leds) for i in range(self.num_leds)]
        
        # 绘制底面 (Sensor Surface)
        # 生成正方形
        x_base = [-self.width/2, self.width/2, self.width/2, -self.width/2, -self.width/2]
        y_base = [-self.width/2, -self.width/2, self.width/2, self.width/2, -self.width/2]
        z_base = [0, 0, 0, 0, 0]
        ax4.plot(x_base, y_base, z_base, 'k-', label='Sensor Surface')
        # 填充底面 (简单画个网格)
        grid_lines = np.linspace(-self.width/2, self.width/2, 5)
        for g in grid_lines:
            ax4.plot([g, g], [-self.width/2, self.width/2], [0, 0], 'k-', alpha=0.1)
            ax4.plot([-self.width/2, self.width/2], [g, g], [0, 0], 'k-', alpha=0.1)

        # 绘制LED点
        led_z = np.full_like(led_x, self.height)
        ax4.scatter(led_x, led_y, led_z, c='orange', s=50, marker='o', label='LEDs')

        # 绘制光锥
        # 计算光锥底面半径: R = H * tan(FOV/2)
        cone_radius = self.height * np.tan(np.deg2rad(self.led_fov / 2))
        
        # 绘制每个LED的光锥
        for i in range(self.num_leds):
            lx, ly, lz = led_x[i], led_y[i], self.height
            
            # 生成圆锥面的线
            # 我们画几条母线和底面圆
            num_lines = 8
            for j in range(num_lines):
                angle = 2 * np.pi * j / num_lines
                # 锥底上的点 (相对于LED投影点中心)
                # 注意：这里假设LED垂直向下。如果倾斜，光锥轴线也要倾斜。
                base_x = lx + cone_radius * np.cos(angle)
                base_y = ly + cone_radius * np.sin(angle)
                base_z = 0
                
                ax4.plot([lx, base_x], [ly, base_y], [lz, base_z], 'y-', alpha=0.2)
            
            # 绘制光锥在底面的投影圆 (FOV覆盖范围)
            theta = np.linspace(0, 2*np.pi, 50) # Need theta here as it was removed above
            cx = lx + cone_radius * np.cos(theta)
            cy = ly + cone_radius * np.sin(theta)
            cz = np.zeros_like(cx)
            ax4.plot(cx, cy, cz, 'y--', alpha=0.3)

        ax4.set_xlabel('X (mm)')
        ax4.set_ylabel('Y (mm)')
        ax4.set_zlabel('Z (mm)')
        ax4.set_zlim(0, self.height + 2)
        # 调整视角以看清结构
        ax4.view_init(elev=30, azim=45)

        # 2. Intensity vs Threshold Surface (3D)
        ax5 = fig2.add_subplot(1, 2, 2, projection='3d')
        # 为了渲染速度，降采样 (分辨率提高后需要增加stride)
        stride = 20
        surf = ax5.plot_surface(self.X[::stride, ::stride], self.Y[::stride, ::stride], self.light_map_norm[::stride, ::stride], 
                               cmap='viridis', alpha=0.8, label='Intensity')
        
        # 绘制阈值面 (半透明红色)
        surf2 = ax5.plot_surface(self.X[::stride, ::stride], self.Y[::stride, ::stride], self.threshold_map[::stride, ::stride], 
                                color='red', alpha=0.3)
        
        ax5.set_title('Intensity vs Threshold Surface (3D)')
        ax5.set_xlabel('X (mm)')
        ax5.set_ylabel('Y (mm)')
        ax5.set_zlabel('Value (0-255)')
        
        # 添加 Colorbar (对应 Intensity)
        fig2.colorbar(surf, ax=ax5, shrink=0.5, aspect=10, label='Intensity Value', pad=0.1)

        plt.tight_layout()
        if save_img:
            save_path = os.path.join(self.output_dir, 'simulation_result_3d.png')
            plt.savefig(save_path)
            print(f"Result saved to {save_path}")
        # plt.show() # Move show to the end of script or comparison method

    def compare_threshold_methods(self, save_img=True):
        """
        对比全局阈值法与自适应阈值法的效果
        展示"高阈值"和"低阈值"的困境，以及自适应方法的优势
        """
        if self.raw_image_uint8 is None:
            return

        # --- 生成理想均匀光照图像 (Ideal Uniform Image) ---
        # 假设光照是完美的均匀分布，强度为最大光强
        max_intensity = np.max(self.light_map) if self.light_map is not None else 1.0
        if max_intensity == 0: max_intensity = 1.0
        
        # 理想光照图 = 全局最大光强 * 反射率图
        ideal_light = np.full_like(self.light_map, max_intensity)
        ideal_image = ideal_light * self.reflectivity_map
        
        # 归一化
        ideal_image_norm = ideal_image * (255.0 / max_intensity)
        
        # 遮罩处理
        # mask = (self.X**2 + self.Y**2) > self.radius_base**2
        # ideal_image_norm[mask] = 0

        # --- 可视化对比 ---
        fig = plt.figure(figsize=(10, 9))
        fig.suptitle('Global (Otsu) vs Adaptive Thresholding Comparison', fontsize=14)

        # 1. Ideal Uniform Image (Reference)
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(ideal_image_norm, cmap='gray', vmin=0, vmax=255,
                   extent=[-self.width/2, self.width/2, -self.width/2, self.width/2])
        ax1.set_title('1. Ideal Uniform Light\n(Ground Truth)')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')

        # 2. Simulated Raw Image (Input)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(self.raw_image_norm, cmap='gray', vmin=0, vmax=255,
                   extent=[-self.width/2, self.width/2, -self.width/2, self.width/2])
        ax2.set_title('2. Simulated Input\n(Uneven Light)')
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')

        # 3. Otsu (Global Threshold)
        # Otsu calculation
        ret_otsu, binary_otsu = cv2.threshold(self.raw_image_uint8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # binary_otsu[mask] = 0
        
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(binary_otsu, cmap='gray', vmin=0, vmax=255,
                   extent=[-self.width/2, self.width/2, -self.width/2, self.width/2])
        ax3.set_title(f'3. Otsu Global Threshold (T={int(ret_otsu)})\n(Compromise: Misses Center or Noise at Edge)')
        ax3.set_xlabel('X (mm)')
        ax3.set_ylabel('Y (mm)')

        # 4. Adaptive (Our Method)
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(self.binary_map, cmap='gray', vmin=0, vmax=255,
                   extent=[-self.width/2, self.width/2, -self.width/2, self.width/2])
        ax4.set_title(f'4. Adaptive Thresholding\n(Robust Solution)')
        ax4.set_xlabel('X (mm)')
        ax4.set_ylabel('Y (mm)')

        plt.tight_layout()
        if save_img:
            save_path = os.path.join(self.output_dir, 'simulation_comparison.png')
            plt.savefig(save_path)
            print(f"Comparison result saved to {save_path}")
        plt.show()

if __name__ == "__main__":
    # 实例化并运行仿真
    sim = TactileSensorSim()
    
    # --- 用户可以在这里修改参数 ---
    
    # [1. 传感器几何结构参数]
    sim.width = 40.0              # 传感器底面宽度 (mm) - 正方形边长
    sim.height = 30.0             # LED平面到底面的垂直高度 (mm)
    sim.radius_led = 12.0         # LED阵列的分布半径 (mm)
    
    # [2. 光源参数]
    sim.num_leds = 9              # LED数量 (建议 6-12) - 决定光场的叠加平滑度
    sim.led_fov = 120             # LED发光角 (FOV, 度) - 120度为典型贴片LED参数
    sim.led_power = 1000.0        # LED发光强度 (相对值)
    
    # [3. 自适应阈值算法参数 (核心)]
    # block_size: 局部窗口大小 (像素)，必须是奇数。
    # 物理意义: 决定了背景估计的平滑程度。
    # 设置原则: 必须 > 标记点直径 (约50px)。建议取 3-5 倍直径 (约 150-250)。
    # 当前值 10 (0.2mm) 可能过小，会导致标记点内部被错误二值化。建议改为 251。
    sim.block_size = 151           
    
    sim.C = 10                    # 阈值常数 C - 决定了对噪声的敏感度 (建议 5-15)
    
    # [4. 材料光学属性]
    sim.bg_reflectivity = 1       # 背景反射率 (0.0-1.0) - 模拟高反光弹性体
    sim.marker_reflectivity = 0.4 # 标记点反射率 (0.0-1.0) - 模拟深色标记点
    
    # [5. 噪声参数]
    sim.noise_sigma = 2.0         # 图像随机噪声标准差 (建议 1.0-5.0) - 模拟相机传感器噪声
    
    # --- 开关 ---
    SAVE_IMAGES = False # 是否保存图片
    
    # 运行
    sim.generate_light_field()
    sim.calculate_adaptive_threshold()
    sim.visualize(save_img=SAVE_IMAGES)
    sim.compare_threshold_methods(save_img=SAVE_IMAGES)
