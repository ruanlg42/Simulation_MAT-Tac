import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import os

# 尝试导入 3D 绘图工具
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class DualCameraLayoutSim:
    """
    双摄像头与LED阵列布局及视触觉视野仿真 (3D 版 - 翻转坐标系)
    """
    def __init__(self):
        # --- 传感器外壳参数 ---
        self.sensor_w = 40.0  # 宽度 (mm)
        self.sensor_h = 40.0  # 长度 (mm)
        self.sensor_depth = 30.0 # 高度 (mm, PCB 到触觉面的距离)
        
        # --- LED 参数 ---
        # 0603 封装: 1.6mm x 0.8mm，这里用外接圆半径近似
        self.led_count = 9
        self.led_radius = 16.0 # 分布半径 (mm)
        self.led_size = 0.9    # 近似半径 (mm, 基于 0603 封装)
        self.led_angle_offset = 0.0 # LED 阵列旋转角度
        
        # LED 光学参数 (用于光照仿真)
        self.led_fov = 120.0   # LED 视场角
        self.led_power = 1.0   # 相对功率
        
        # --- 摄像头参数 ---
        # 1. RGB Camera (正方形 8.5x8.5mm)
        self.cam_rgb = {
            'name': 'RGB',
            'shape': 'rect',  # 形状标识
            'width': 8.5,     # 宽度 (mm)
            'height_size': 8.5, # 长度 (mm)
            'fov': 120.0,     # 视场角 (度)
            'height': 5.0,    # 摄像头物理高度 (mm)
            'color': 'blue',
            'pos': np.array([0.0, 0.0]) # 中心坐标
        }
        # 计算 RGB 的外接圆半径用于快速碰撞检测 (对角线的一半)
        self.cam_rgb['radius_outer'] = np.sqrt(self.cam_rgb['width']**2 + self.cam_rgb['height_size']**2) / 2.0
        
        # 2. Thermal Camera (圆形 r=5.0mm)
        self.cam_therm = {
            'name': 'Thermal',
            'shape': 'circle',
            'radius': 5.0,    # 物理半径 (mm)
            'fov': 110.0,     # 视场角 (度)
            'height': 6.5,    # 摄像头物理高度 (mm)
            'color': 'red',
            'pos': np.array([0.0, 0.0]) # 中心坐标
        }
        
        # --- 摄像头高度偏移参数 (新增) ---
        # 初始为 0 表示现在的状态 (镜头在 LED 平面下方)
        # 如果设置为摄像头高度 (如 RGB = 5.0，Thermal = 6.5)，则镜头与 LED 平面平齐
        self.rgb_offset = 0.0      # RGB 摄像头垂直偏移 (mm)
        self.thermal_offset = 0.0  # 热成像摄像头垂直偏移 (mm)
        
        self.output_dir = 'layout_results_3d'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def calculate_fov_radius_at_height(self, fov_deg, sensor_height, cam_height):
        """
        计算特定高度下的视野 (FOV) 半径。
        
        该函数基于针孔相机模型，计算摄像头在指定距离下的可视半径。
        
        参数:
            fov_deg (float): 摄像头的视场角 (Field of View)，单位为度。
            sensor_height (float): 传感器整体结构的高度 (LED 平面高度)，单位 mm。
            cam_height (float): 摄像头自身的物理高度 (从底座到镜头)，单位 mm。
            
        返回:
            float: 在目标平面 (通常为触觉面 Z=0) 上的视野半径，单位 mm。
            
        原理:
            根据三角函数关系: r = d * tan(FOV / 2)
            其中:
            - r 是视野半径
            - d 是有效成像距离 (sensor_height - cam_height)
            - FOV 是视场角
        """
        effective_dist = sensor_height - cam_height
        half_angle = np.deg2rad(fov_deg / 2.0)
        return effective_dist * np.tan(half_angle)

    def check_collision(self, p1, r1, p2, r2):
        """
        检查两个圆形物体是否发生物理碰撞。
        
        使用欧几里得距离判断两个圆是否重叠。
        
        参数:
            p1 (np.array): 第一个圆的圆心坐标 [x, y]。
            r1 (float): 第一个圆的半径。
            p2 (np.array): 第二个圆的圆心坐标 [x, y]。
            r2 (float): 第二个圆的半径。
            
        返回:
            bool: 如果两圆重叠 (发生碰撞) 返回 True，否则返回 False。
        """
        dist = np.linalg.norm(p1 - p2)
        return dist < (r1 + r2)

    def check_rect_circle_collision(self, rect_center, rect_size, circle_center, circle_radius):
        """
        检查矩形物体与圆形物体是否发生碰撞。
        
        用于检测方形元件 (如 RGB 摄像头) 与圆形元件 (如 LED) 之间的干涉。
        
        参数:
            rect_center (np.array): 矩形中心坐标 [x, y]。
            rect_size (tuple): 矩形尺寸 (宽, 高)。
            circle_center (np.array): 圆形中心坐标 [x, y]。
            circle_radius (float): 圆形半径。
            
        返回:
            bool: 如果发生碰撞返回 True，否则返回 False。
            
        算法逻辑:
            1. 坐标变换: 将圆心映射到以矩形为原点的第一象限 (利用对称性)。
            2. 区域判断: 
               - 如果圆心在矩形外部且距离超过半径，则无碰撞。
               - 如果圆心在矩形内部或边缘距离小于半径，则碰撞。
            3. 角点检测: 检查矩形角点是否在圆内。
        """
        # 矩形半宽和半高
        rw = rect_size[0] / 2.0
        rh = rect_size[1] / 2.0
        
        # 计算圆心到矩形中心在 x, y 轴上的距离
        dx = abs(circle_center[0] - rect_center[0])
        dy = abs(circle_center[1] - rect_center[1])

        # 如果圆心距离超过 (半宽 + 半径)，则肯定不相交
        if dx > (rw + circle_radius): return False
        if dy > (rh + circle_radius): return False

        # 如果圆心距离小于半宽，则肯定相交
        if dx <= rw: return True
        if dy <= rh: return True

        # 检查角点碰撞 (勾股定理)
        corner_dist_sq = (dx - rw)**2 + (dy - rh)**2
        return corner_dist_sq <= (circle_radius**2)

    def is_blocked_by_cameras(self, led_pos, target_point):
        """
        判断从 LED 到目标点的光线路径是否被任意摄像头遮挡。
        
        这是光照仿真的关键步骤，用于生成逼真的阴影效果。
        
        参数:
            led_pos (np.array): LED 光源的三维坐标 [x, y, z]。
            target_point (np.array): 目标点 (触觉面上) 的三维坐标 [x, y, z]。
            
        返回:
            bool: 如果光路被遮挡返回 True，否则返回 False。
            
        物理模型 (翻转坐标系):
            - LED 平面: Z = sensor_depth (30mm)
            - 触觉面: Z = 0mm
            - 光路: 从 Z=30 射向 Z=0
            - 摄像头: 视为悬挂在顶部的障碍物
            
        实现细节:
            通过调用 check_ray_occlusion 内部函数，分别检测光线是否与 RGB 摄像头 (近似为圆柱体) 
            或 Thermal 摄像头 (圆柱体) 相交。
        """
        # 摄像头在 Z 轴上的物理区间 (考虑了偏移量)
        rgb_z_min = self.sensor_depth - self.cam_rgb['height'] + self.rgb_offset
        rgb_z_max = self.sensor_depth + self.rgb_offset
        
        therm_z_min = self.sensor_depth - self.cam_therm['height'] + self.thermal_offset
        therm_z_max = self.sensor_depth + self.thermal_offset

        def check_ray_occlusion(ray_start, ray_end, cyl_center, cyl_radius, z_min, z_max):
            """
            内部辅助函数：检测射线是否穿过垂直圆柱体。
            
            参数:
                ray_start, ray_end: 射线起止点。
                cyl_center, cyl_radius: 圆柱体参数。
                z_min, z_max: 圆柱体的高度范围。
            """
            # 1. 高度筛选: 如果障碍物完全在光源平面之上，则不可能遮挡向下的光线
            if z_min >= self.sensor_depth - 1e-6:
                return False
                
            # 2. 2D 投影检测: 将 3D 射线投影到 XY 平面，计算线段到圆心的最短距离
            p0 = ray_start[:2]
            p1 = ray_end[:2]
            v = p1 - p0
            w = cyl_center - p0
            
            # 计算线段上距离圆心最近的点 (参数化表示: P = P0 + t*V, t in [0,1])
            c1 = np.dot(w, v)
            if c1 <= 0:
                dist = np.linalg.norm(w) # 最近点是起点
            else:
                c2 = np.dot(v, v)
                if c2 <= c1:
                    dist = np.linalg.norm(cyl_center - p1) # 最近点是终点
                else:
                    b = c1 / c2
                    pb = p0 + b * v # 最近点在线段中间
                    dist = np.linalg.norm(cyl_center - pb)
            
            # 如果投影距离小于半径，则认为水平方向发生遮挡
            # (此处简化模型：假设只要水平路径重叠即视为遮挡，对于俯视布局足够精确)
            return dist < cyl_radius

        # RGB (使用外接圆近似进行遮挡判断)
        if check_ray_occlusion(led_pos, target_point, self.cam_rgb['pos'], self.cam_rgb['radius_outer'], rgb_z_min, rgb_z_max):
            return True
            
        # Thermal
        if check_ray_occlusion(led_pos, target_point, self.cam_therm['pos'], self.cam_therm['radius'], therm_z_min, therm_z_max):
            return True
            
        return False

    def optimize_layout(self):
        """
        计算并应用最佳的摄像头布局策略。
        
        目标:
            在有限的 40x40mm 空间内，放置两个摄像头 (RGB 和 Thermal)，
            使其互不干涉，且尽可能利用空间。
            
        策略:
            沿 45 度对角线对称分布。这种布局方式在正方形传感器中最节省空间，
            并且能够让两个摄像头的视野中心尽可能靠近传感器中心。
        """
        # 计算最小中心距: Thermal 半径 + RGB 外接圆半径 + 0.5mm 安全间隙
        rgb_diag_half = self.cam_rgb['radius_outer']
        min_dist = self.cam_therm['radius'] + rgb_diag_half + 0.5 
        
        # 策略：沿对角线对称分布
        angle = np.deg2rad(45)
        offset = min_dist / 2.0
        
        dx = offset * np.cos(angle)
        dy = offset * np.sin(angle)
        
        # 更新摄像头坐标
        self.cam_rgb['pos'] = np.array([-dx, -dy])
        self.cam_therm['pos'] = np.array([dx, dy])
        
        print(f"Layout Optimized: Distance = {min_dist:.2f}mm")
        print(f"RGB Pos: {self.cam_rgb['pos']}")
        print(f"Therm Pos: {self.cam_therm['pos']}")

    def calculate_light_distribution(self, resolution=100):
        """
        计算触觉表面的光照分布强度，包含遮挡计算。
        
        参数:
            resolution (int): 输出光照图的分辨率 (resolution x resolution)。
            
        返回:
            tuple: (X坐标网格, Y坐标网格, 归一化光照强度图)
            
        物理模型 (Lambertian Model):
            光强 I 遵循公式: I = I0 * (cos(theta)^m) / d^2 * cos(alpha)
            
            其中:
            - d: 光源到目标点的距离 (遵循平方反比衰减)。
            - theta: 发光角度 (LED 法线与光线夹角)。
            - m: 朗伯体指数，由 LED 的半功率角 (FOV) 决定。
              公式: m = -ln(2) / ln(cos(FOV/2))
            
        过程:
            1. 遍历每个 LED 光源。
            2. 遍历触觉面上的每个像素点。
            3. 计算光线向量、距离和角度。
            4. 调用 is_blocked_by_cameras 检查遮挡，若被遮挡则光强为 0。
            5. 累加所有可见 LED 的光强贡献。
        """
        # 触觉面在 Z=0
        x = np.linspace(-self.sensor_w/2, self.sensor_w/2, resolution)
        y = np.linspace(-self.sensor_h/2, self.sensor_h/2, resolution)
        X, Y = np.meshgrid(x, y)
        light_map = np.zeros_like(X)
        
        # 计算朗伯体指数 m
        # m = -ln(2) / ln(cos(FOV/2))
        alpha = np.deg2rad(self.led_fov / 2.0)
        m = -np.log(2) / np.log(np.cos(alpha))
        
        print("Calculating light distribution with shadow casting (Flipped)...")
        
        for i in range(self.led_count):
            theta = 2 * np.pi * i / self.led_count + self.led_angle_offset
            lx = self.led_radius * np.cos(theta)
            ly = self.led_radius * np.sin(theta)
            # LED 在顶部 Z=30
            lz = self.sensor_depth 
            led_pos = np.array([lx, ly, lz])
            
            for r in range(resolution):
                for c in range(resolution):
                    tx, ty = X[r, c], Y[r, c]
                    # 目标点在 Z=0
                    tz = 0
                    target_point = np.array([tx, ty, tz])
                    
                    dx = tx - lx
                    dy = ty - ly
                    dz = tz - lz # dz 是负数，方向向下
                    dist_sq = dx**2 + dy**2 + dz**2
                    
                    # 检查遮挡 (Ray Casting)
                    if self.is_blocked_by_cameras(led_pos, target_point):
                        intensity = 0
                    else:
                        # cos_theta = |dz| / dist (假设 LED 垂直向下照射)
                        cos_theta = abs(dz) / np.sqrt(dist_sq)
                        intensity = self.led_power * (cos_theta ** m) / dist_sq
                        
                    light_map[r, c] += intensity
            
        light_map = light_map / np.max(light_map)
        return X, Y, light_map

    def visualize_3d(self):
        """生成 3D 可视化图"""
        # 使用白色背景样式
        plt.style.use('default')

        fig = plt.figure(figsize=(18, 8), facecolor='white')
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        # --- 子图 1: 2D 布局检查 (保持俯视图) ---
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_aspect('equal')
        # 英文标题: PCB Layout (Top View) & Occlusion Check
        ax1.set_title('PCB Layout (Top View) & Occlusion Check', color='black')
        ax1.set_xlim(-25, 25)
        ax1.set_ylim(-25, 25)
        
        # 外壳
        ax1.add_patch(Rectangle((-20, -20), 40, 40, fill=False, color='black', linewidth=2))
        
        # LED (0603)
        led_collisions = 0
        for i in range(self.led_count):
            theta = 2 * np.pi * i / self.led_count + self.led_angle_offset
            lx = self.led_radius * np.cos(theta)
            ly = self.led_radius * np.sin(theta)
            led_pos = np.array([lx, ly])
            
            # 检查遮挡
            is_blocked = False
            rgb_size = (self.cam_rgb['width'], self.cam_rgb['height_size'])
            if self.check_rect_circle_collision(self.cam_rgb['pos'], rgb_size, led_pos, self.led_size):
                is_blocked = True
            if self.check_collision(led_pos, self.led_size, self.cam_therm['pos'], self.cam_therm['radius']):
                is_blocked = True
            
            color = 'lime' if not is_blocked else 'red'
            if is_blocked: led_collisions += 1
            
            ax1.add_patch(Circle((lx, ly), self.led_size, color=color))
            
        # 摄像头 (2D 投影)
        rgb_w = self.cam_rgb['width']
        rgb_h = self.cam_rgb['height_size']
        rgb_xy = (self.cam_rgb['pos'][0] - rgb_w/2, self.cam_rgb['pos'][1] - rgb_h/2)
        ax1.add_patch(Rectangle(rgb_xy, rgb_w, rgb_h, color='blue', alpha=0.3, label='RGB'))
        ax1.add_patch(Circle(self.cam_therm['pos'], self.cam_therm['radius'], color='red', alpha=0.3, label='Thermal'))
        
        ax1.legend()
        ax1.grid(True, linestyle=':')
        if led_collisions > 0:
            # 英文警告: Warning: {n} LEDs blocked!
            ax1.text(0, -23, f"Warning: {led_collisions} LEDs blocked!", color='red', ha='center')

        # --- 子图 2: 3D 空间分布 (翻转) ---
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        # 英文标题: 3D Sensor Model: Camera Top-Down View (Z=30 -> Z=0)
        ax2.set_title('3D Sensor Model: Camera Top-Down View (Z=30 -> Z=0)')
        
        # 1. 绘制顶部的 PCB 和 LED (Z=30)
        led_xs, led_ys, led_zs = [], [], []
        for i in range(self.led_count):
            theta = 2 * np.pi * i / self.led_count + self.led_angle_offset
            led_xs.append(self.led_radius * np.cos(theta))
            led_ys.append(self.led_radius * np.sin(theta))
            led_zs.append(self.sensor_depth) # Z=30
        ax2.scatter(led_xs, led_ys, led_zs, c='lime', s=10, label='LEDs')
        
        # 2. 绘制摄像头实体 (从 Z=30+offset 向下延伸)
        # Thermal (圆柱体)
        def plot_cylinder_down(pos, radius, height, top_z, offset, color, ax):
            actual_top_z = top_z + offset
            z = np.linspace(actual_top_z - height, actual_top_z, 10)
            theta = np.linspace(0, 2*np.pi, 20)
            theta_grid, z_grid = np.meshgrid(theta, z)
            x_grid = radius * np.cos(theta_grid) + pos[0]
            y_grid = radius * np.sin(theta_grid) + pos[1]
            ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=0.3)
        
        plot_cylinder_down(self.cam_therm['pos'], self.cam_therm['radius'], self.cam_therm['height'], self.sensor_depth, self.thermal_offset, 'red', ax2)
        
        # RGB (长方体)
        def plot_box_down(pos, w, h, height, top_z, offset, color, ax):
            actual_top_z = top_z + offset
            dx = w/2
            dy = h/2
            x = [pos[0]-dx, pos[0]+dx, pos[0]+dx, pos[0]-dx, pos[0]-dx, pos[0]+dx, pos[0]+dx, pos[0]-dx]
            y = [pos[1]-dy, pos[1]-dy, pos[1]+dy, pos[1]+dy, pos[1]-dy, pos[1]-dy, pos[1]+dy, pos[1]+dy]
            z = [actual_top_z-height, actual_top_z-height, actual_top_z-height, actual_top_z-height, actual_top_z, actual_top_z, actual_top_z, actual_top_z]
            
            verts = [
                [0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7], 
                [0,1,2,3], [4,5,6,7]
            ]
            poly_verts = []
            for face in verts:
                poly_verts.append(list(zip([x[i] for i in face], [y[i] for i in face], [z[i] for i in face])))
            collection = Poly3DCollection(poly_verts, facecolors=color, linewidths=1, edgecolors=color, alpha=0.3)
            ax.add_collection3d(collection)
            
        plot_box_down(self.cam_rgb['pos'], self.cam_rgb['width'], self.cam_rgb['height_size'], self.cam_rgb['height'], self.sensor_depth, self.rgb_offset, 'blue', ax2)
        
        # 3. 绘制视锥 (从摄像头底部向下延伸到 Z=0)
        def plot_cone_fov_down(pos, fov, cam_height, total_height, offset, color, ax):
            # 摄像头镜头位置 (Z = total_height - cam_height + offset)
            lens_z = total_height - cam_height + offset
            effective_dist = max(0, lens_z - 0) # 到 Z=0 的距离
            r_bottom = effective_dist * np.tan(np.deg2rad(fov/2))
            
            # 绘制半透明视锥
            z = np.linspace(0, lens_z, 20)
            theta = np.linspace(0, 2*np.pi, 30)
            theta_grid, z_grid = np.meshgrid(theta, z)
            
            # 半径随 Z 减小而增大 (向下变大)
            r_grid = r_bottom * (lens_z - z_grid) / effective_dist if effective_dist > 0 else np.zeros_like(z_grid)
            
            x_grid = r_grid * np.cos(theta_grid) + pos[0]
            y_grid = r_grid * np.sin(theta_grid) + pos[1]
            
            ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=0.1, shade=False)
            
            # 绘制底部轮廓 (Z=0)
            xi = r_bottom * np.cos(theta) + pos[0]
            yi = r_bottom * np.sin(theta) + pos[1]
            zi = np.zeros_like(xi)
            ax.plot(xi, yi, zi, color=color, alpha=0.8, linewidth=2, label='FOV @ Bottom')

        plot_cone_fov_down(self.cam_rgb['pos'], self.cam_rgb['fov'], self.cam_rgb['height'], self.sensor_depth, self.rgb_offset, 'blue', ax2)
        plot_cone_fov_down(self.cam_therm['pos'], self.cam_therm['fov'], self.cam_therm['height'], self.sensor_depth, self.thermal_offset, 'red', ax2)
        
        # 4. 绘制底面触觉面 (光照分布) - Z=0
        X, Y, light_map = self.calculate_light_distribution(resolution=40)
        # 使用 YlOrRd colormap，适应白色背景，vmin=0 确保无光处为白色/浅黄
        cset = ax2.contourf(X, Y, light_map, zdir='z', offset=0, cmap='YlOrRd', alpha=0.8, levels=50, vmin=0)
        
        # 添加光强分布颜色条 (新增)
        cbar = fig.colorbar(cset, ax=ax2, shrink=0.5, aspect=10, pad=0.1)
        # 英文标签: Relative Intensity
        cbar.set_label('Relative Intensity', color='black')
        cbar.ax.yaxis.set_tick_params(color='black', labelcolor='black')
        
        # 设置坐标轴
        ax2.set_xlabel('X (mm)', color='black')
        ax2.set_ylabel('Y (mm)', color='black')
        ax2.set_zlabel('Z (mm)', color='black')
        ax2.set_xlim(-25, 25)
        ax2.set_ylim(-25, 25)
        ax2.set_zlim(0, 35)
        
        # 设置背景板颜色 (浅灰色增加空间感)
        ax2.xaxis.pane.set_facecolor((0.95, 0.95, 0.95, 1.0))
        ax2.yaxis.pane.set_facecolor((0.95, 0.95, 0.95, 1.0))
        ax2.zaxis.pane.set_facecolor((0.95, 0.95, 0.95, 1.0))
        
        # 恢复网格
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        # 视角调整 (俯视)
        ax2.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'dual_cam_layout_3d_flipped.png')
        plt.savefig(save_path)
        print(f"3D 仿真结果图已保存至: {save_path}")
        plt.show()

if __name__ == "__main__":
    sim = DualCameraLayoutSim()
    
    # --- 调整摄像头高度偏移 ---
    # 0.0: 默认状态 (镜头在 LED 下方)
    # 5.0: RGB 镜头与 LED 平齐 (因为 RGB 高度为 5mm)
    # 6.5: 热成像镜头与 LED 平齐 (因为热成像高度为 6.5mm)
    # sim.rgb_offset = 0.0      # 您可以修改这里
    # sim.thermal_offset = 0.0  # 您可以修改这里
    sim.rgb_offset = 5.0      # 您可以修改这里
    sim.thermal_offset = 6.5  # 您可以修改这里
    
    sim.optimize_layout()
    sim.visualize_3d()
