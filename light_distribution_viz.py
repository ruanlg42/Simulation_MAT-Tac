import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def simulate_light_distribution():
    """
    专门模拟视触觉传感器底面的光强分布，并绘制光锥边界
    """
    # --- 1. 参数设置 (保持与 simulation.py 一致) ---
    width = 40.0           # 底面宽度 (mm)
    height = 30.0          # LED高度 (mm)
    num_leds = 9           # LED数量
    radius_led = 12.0      # LED分布半径 (mm)
    led_fov = 120.0        # LED视场角 (度)
    resolution = 0.2       # 绘图分辨率 (mm/pixel)
    margin = 50          # 绘图边距
    
    # --- 2. 创建底面网格 ---
    grid_size = int(width / resolution)
    x = np.linspace(-width/2, width/2, grid_size)
    y = np.linspace(-width/2, width/2, grid_size)
    X, Y = np.meshgrid(x, y)
    light_map = np.zeros_like(X)

    # --- 3. 计算光照分布 (朗伯模型) ---
    alpha = np.deg2rad(led_fov / 2.0)
    m = -np.log(2) / np.log(np.cos(alpha))
    
    led_positions = []
    for i in range(num_leds):
        theta_led = 2 * np.pi * i / num_leds
        lx = radius_led * np.cos(theta_led)
        ly = radius_led * np.sin(theta_led)
        lz = height
        led_positions.append((lx, ly, lz))
        
        dx = X - lx
        dy = Y - ly
        dz = 0 - lz
        dist_sq = dx**2 + dy**2 + dz**2
        dist = np.sqrt(dist_sq)
        
        cos_theta = lz / dist 
        cos_theta = np.maximum(cos_theta, 0)
        
        intensity = (cos_theta ** m) / dist_sq
        light_map += intensity

    # --- 4. 绘图 ---
    plt.style.use('dark_background') # 使用深色背景以便看清光圈边界
    plt.figure(figsize=(10, 8))
    
    # 绘制底面光强分布热力图
    im = plt.imshow(light_map, extent=[-width/2, width/2, -width/2, width/2], 
                    cmap='magma', origin='lower') # 使用 magma 这种深色底的色表
    plt.colorbar(im, label='Relative Intensity')
    
    # 计算光锥在底面的投影半径: R_cone = H * tan(FOV/2)
    cone_radius_at_base = height * np.tan(alpha)
    print(f"Calculated Cone Radius at Base: {cone_radius_at_base:.2f} mm")
    
    # 绘制每个LED的投影位置和光锥边界(虚线圆)
    for i, (lx, ly, lz) in enumerate(led_positions):
        # 绘制LED中心位置
        plt.plot(lx, ly, 'wo', markersize=4)
        
        # 绘制光锥边界 (青色虚线圆，代表该LED光照覆盖的最远范围)
        circle = Circle((lx, ly), cone_radius_at_base, color='cyan', 
                        fill=False, linestyle='--', linewidth=1.2, alpha=0.5)
        plt.gca().add_patch(circle)
        
        # 标注LED编号
        plt.text(lx, ly, f'#{i+1}', color='white', fontsize=9, fontweight='bold')

    # 绘制传感器底面边界 (40x40mm 正方形)
    plt.plot([-width/2, width/2, width/2, -width/2, -width/2], 
             [-width/2, -width/2, width/2, width/2, -width/2], 
             'w-', linewidth=2, label='Sensor Base (40x40mm)')
    
    # 在图上添加半径说明
    plt.text(-width/2-margin+5, -width/2-margin+5, 
             f'LED FOV Circle Radius: {cone_radius_at_base:.1f}mm\n(H={height}mm, FOV={led_fov}°)', 
             color='cyan', fontsize=10, bbox=dict(facecolor='black', alpha=0.5))

    plt.title(f'Light Distribution & FOV Boundaries on Sensor Base\n(H={height}mm, FOV={led_fov}°, LEDs={num_leds})', fontsize=14)
    plt.xlabel('X (mm)', fontsize=12)
    plt.ylabel('Y (mm)', fontsize=12)
    plt.grid(True, alpha=0.2)
    plt.axis('equal')
    
    # 设置显示范围以完整展示光锥圆
    plt.xlim(-width/2 - margin, width/2 + margin)
    plt.ylim(-width/2 - margin, width/2 + margin)
    
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('light_distribution_2d_only.png', dpi=300)
    print("Combined visualization saved to light_distribution_2d_only.png")
    plt.show()

if __name__ == "__main__":
    simulate_light_distribution()
