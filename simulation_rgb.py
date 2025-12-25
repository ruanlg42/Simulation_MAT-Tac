import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
import os
from scipy.ndimage import gaussian_filter

# 设置全局字体为 Arial
plt.rcParams['font.family'] = 'Arial'

class TactileSensorRGBSim:
    """
    视触觉传感器 RGB 多色光场仿真类
    用于仿真 9 个 LED（红、绿、蓝三色交替）在传感器底面形成的彩色光场及成像
    """
    def __init__(self):
        # --- 几何参数 (单位: mm) ---
        self.width = 40.0             # 底面宽度 (正方形)
        self.height = 30.0            # LED平面到底面的距离
        self.radius_led = 15.0        # LED分布圆半径
        
        # --- 接触物体参数 ---
        self.surface_type = 'flat'    # 表面类型: 'flat' (平整) 或 'sphere' (球体接触)
        self.sphere_radius = 10.0     # 接触球体半径 (mm)
        self.contact_depth = 3.0      # 压入深度 (mm)
        
        # --- 光源参数 ---
        self.num_leds = 9             # LED数量 (9个，每种颜色3个)
        self.led_fov = 120            # LED视场角 (度)
        self.led_power = 80000.0      # LED基础光强系数
        
        # --- 图像仿真参数 ---
        self.resolution = 0.05        # 仿真网格分辨率 (mm/pixel)
        self.noise_sigma = 1.5        # 图像随机噪声标准差
        self.output_dir = 'results'   # 输出目录
        
        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # --- 标记点 (Marker) 参数 ---
        self.marker_pitch = 4.0       # 标记点间距 (mm)
        self.marker_row = 10          # 标记点行数
        self.marker_col = 10          # 标记点列数
        self.marker_radius = 0.5      # 标记点半径 (mm)
        self.bg_reflectivity = 1.0    # 背景材料反射率
        self.marker_reflectivity = 0.3 # 标记点反射率 (模拟深色标记点)
        
        # 内部变量
        self.X = None
        self.Y = None
        self.rgb_light_map = None     # 彩色光照分布 (H x W x 3)
        self.rgb_raw_image = None     # 模拟的彩色原始图像
        self.reflectivity_map = None  # 反射率分布

    def generate_rgb_light_field(self):
        """
        生成 RGB 三色光场并模拟弹性体形变 (参考 PDMS 连续体物理特性)
        """
        grid_size = int(self.width / self.resolution)
        x = np.linspace(-self.width/2, self.width/2, grid_size)
        y = np.linspace(-self.width/2, self.width/2, grid_size)
        self.X, self.Y = np.meshgrid(x, y)
        
        # 1. 生成连续的形变高度图 (Z)
        if self.surface_type == 'sphere':
            # 计算纯几何压入深度
            dist_sq = self.X**2 + self.Y**2
            R = self.sphere_radius
            d = self.contact_depth
            
            # 接触区域的几何深度
            z_contact = np.sqrt(np.maximum(0, R**2 - dist_sq)) - (R - d)
            z_contact = np.maximum(0, z_contact)
            
            # 模拟 PDMS 的连续性：非接触区域也会由于泊松效应产生“塌陷”
            # 使用高斯平滑来模拟这种连续的表面沉降场
            # 弹性体传播半径增大 (模拟更广的全局形变影响)
            propagation_sigma = 4.0 / self.resolution 
            self.Z = gaussian_filter(z_contact, sigma=propagation_sigma)
            # 确保接触中心的深度保持一致，并模拟 PDMS 整体下陷
            if np.max(self.Z) > 0:
                self.Z = self.Z * (d / np.max(self.Z))
        else:
            self.Z = np.zeros_like(self.X)
            
        # 2. 计算法向量 (用于更真实的光度模型)
        dz_dy, dz_dx = np.gradient(self.Z, self.resolution)
        norm = np.sqrt(dz_dx**2 + dz_dy**2 + 1)
        Nx = -dz_dx / norm
        Ny = -dz_dy / norm
        Nz = 1.0 / norm

        # 3. 计算 9 个 LED 的光场叠加
        self.rgb_light_map = np.zeros((grid_size, grid_size, 3), dtype=np.float32)
        # ambient_light = 30.0  # 移除人工环境光，光照完全由 LED 提供
        # self.rgb_light_map.fill(ambient_light) 
        
        for i in range(self.num_leds):
            color_idx = i % 3
            theta_led = 2 * np.pi * i / self.num_leds
            lx = self.radius_led * np.cos(theta_led)
            ly = self.radius_led * np.sin(theta_led)
            lz = self.height
            
            dx = lx - self.X
            dy = ly - self.Y
            dz = lz - self.Z
            dist_sq = dx**2 + dy**2 + dz**2
            dist = np.sqrt(dist_sq)
            
            ix, iy, iz = dx/dist, dy/dist, dz/dist
            
            # LED 指向中心
            led_vec = np.array([-lx, -ly, -lz])
            led_vec = led_vec / np.linalg.norm(led_vec)
            cos_alpha = np.clip(-(ix*led_vec[0] + iy*led_vec[1] + iz*led_vec[2]), 0, 1)
            m = -np.log(2) / np.log(np.cos(np.radians(self.led_fov/2)))
            
            cos_theta = np.clip(Nx*ix + Ny*iy + Nz*iz, 0, 1)
            intensity = self.led_power * (cos_alpha ** m) * cos_theta / dist_sq
            self.rgb_light_map[:, :, color_idx] += intensity

        # 4. 叠加标记点 (参考 PDMS 弹性畸变)
        self.rgb_raw_image = self.rgb_light_map.copy()
        
        for r in range(self.marker_row):
            for c in range(self.marker_col):
                x0 = (c - (self.marker_col-1)/2) * (self.width / (self.marker_col+1))
                y0 = (r - (self.marker_row-1)/2) * (self.width / (self.marker_row+1))
                
                idx_x = np.clip(int((x0 + self.width/2) / self.resolution), 0, grid_size-1)
                idx_y = np.clip(int((y0 + self.width/2) / self.resolution), 0, grid_size-1)
                
                z_val = self.Z[idx_y, idx_x]
                gx, gy = dz_dx[idx_y, idx_x], dz_dy[idx_y, idx_x]
                
                # 位移系数：进一步降低，PDMS 的剪切模量使得标记点位移较为微妙
                displacement_k = 0.2 
                curr_x = x0 + gx * displacement_k
                curr_y = y0 + gy * displacement_k
                
                px = int((curr_x + self.width/2) / self.resolution)
                py = int((curr_y + self.width/2) / self.resolution)
                
                # 形变参数 (大幅降低比例，模拟 PDMS 的高泊松比带来的局部微小形变)
                base_radius_px = self.marker_radius / self.resolution
                # 放大系数调低
                scale_factor = 1.0 + (z_val / self.height) * 0.4 
                
                # 拉伸系数调低
                stretch_mag = np.sqrt(gx**2 + gy**2)
                stretch_factor = 1.0 + stretch_mag * 0.2
                
                axis_major = int(base_radius_px * scale_factor * stretch_factor)
                axis_minor = int(base_radius_px * scale_factor)
                angle_deg = np.degrees(np.arctan2(gy, gx))
                
                if 0 <= px < grid_size and 0 <= py < grid_size:
                    overlay = self.rgb_raw_image.copy()
                    # 模拟 PDMS 上的标记点反射率降低
                    marker_color = tuple((self.rgb_raw_image[py, px, :] * self.marker_reflectivity).tolist())
                    cv2.ellipse(overlay, (px, py), (axis_major, axis_minor), 
                               angle_deg, 0, 360, marker_color, -1, cv2.LINE_AA)
                    # 边缘羽化
                    self.rgb_raw_image = cv2.addWeighted(overlay, 0.75, self.rgb_raw_image, 0.25, 0)
        
        self.reflectivity_map = np.ones_like(self.Z) # 占位

        # 6. 归一化到 0-255
        max_val = np.max(self.rgb_light_map)
        if max_val > 0:
            scale = 255.0 / max_val
            self.rgb_raw_image_norm = self.rgb_raw_image * scale
            self.rgb_light_map_norm = self.rgb_light_map * scale
        
        # 7. 添加随机噪声
        if self.noise_sigma > 0:
            noise = np.random.normal(0, self.noise_sigma, self.rgb_raw_image_norm.shape)
            self.rgb_raw_image_norm = np.clip(self.rgb_raw_image_norm + noise, 0, 255)

        self.rgb_raw_image_uint8 = self.rgb_raw_image_norm.astype(np.uint8)
        self.rgb_light_map_uint8 = np.clip(self.rgb_light_map_norm, 0, 255).astype(np.uint8)

    def visualize_rgb(self, save_img=True):
        """
        可视化 RGB 仿真结果
        """
        if self.rgb_raw_image is None:
            self.generate_rgb_light_field()
            
        fig = plt.figure(figsize=(16, 5))
        title = f'RGB Multi-Color Simulation (Surface: {self.surface_type.capitalize()})'
        fig.suptitle(title, fontsize=16)
        
        # 1. 模拟的彩色原始图像
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(self.rgb_raw_image_uint8, extent=[-self.width/2, self.width/2, -self.width/2, self.width/2])
        ax1.set_title(f'1. Simulated RGB Image\n({self.surface_type} contact)')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        
        # 2. 纯光场分布 (不含标记点)
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(self.rgb_light_map_uint8, extent=[-self.width/2, self.width/2, -self.width/2, self.width/2])
        ax2.set_title('2. RGB Light Field Distribution\n(Combined R+G+B)')
        ax2.set_xlabel('X (mm)')
        
        # 3. 3D 几何与颜色示意
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        ax3.set_title('3. 3D Geometry & LED Layout')
        
        # 绘制底面 (形变后的表面)
        # 为了 3D 显示效果，对 Z 进行采样
        step = 5
        ax3.plot_surface(self.X[::step, ::step], self.Y[::step, ::step], self.Z[::step, ::step], 
                         cmap='viridis', alpha=0.3, antialiased=False)
        
        # 绘制 RGB LED
        colors = ['red', 'green', 'blue']
        for i in range(self.num_leds):
            theta_led = 2 * np.pi * i / self.num_leds
            lx = self.radius_led * np.cos(theta_led)
            ly = self.radius_led * np.sin(theta_led)
            color = colors[i % 3]
            ax3.scatter(lx, ly, self.height, c=color, s=100, marker='o', edgecolors='k')
            
        ax3.set_xlabel('X (mm)')
        ax3.set_ylabel('Y (mm)')
        ax3.set_zlabel('Z (mm)')
        ax3.set_zlim(0, max(self.height, self.contact_depth) + 5)
        ax3.view_init(elev=25, azim=45)

        plt.tight_layout()
        if save_img:
            save_name = f'simulation_result_rgb_{self.surface_type}.png'
            save_path = os.path.join(self.output_dir, save_name)
            plt.savefig(save_path)
            print(f"RGB Simulation result saved to {save_path}")
        plt.show()

if __name__ == "__main__":
    # 实例化并运行 RGB 仿真
    sim_rgb = TactileSensorRGBSim()
    
    # --- 1. 表面模式选择 ---
    # 选项: 'flat' (平面) 或 'sphere' (球体)
    sim_rgb.surface_type = 'sphere' 
    
    # --- 2. 标记点密度调整 ---
    # 您在这里可以自由调整标记点数量，例如 15x15 (225个) 或 20x20 (400个)
    sim_rgb.marker_row = 10
    sim_rgb.marker_col = 10
    
    # --- 3. 接触几何参数 ---
    if sim_rgb.surface_type == 'sphere':
        sim_rgb.sphere_radius = 12.0   # 球体半径 (mm)
        sim_rgb.contact_depth = 6    # 压入深度 (mm)，深度越大，位移越明显
        sim_rgb.height = 25.0          # 光源高度
    
    # 执行仿真并可视化
    sim_rgb.generate_rgb_light_field()
    sim_rgb.visualize_rgb()
