import numpy as np
import matplotlib.pyplot as plt

# 读取并转换
reward_data = np.load('./reward_list.npy')
rewards = reward_data.flatten() if reward_data.ndim > 1 else reward_data

# 定义一个变量，限制只显示前多少个值
max_display_points = 2500  # 你可以根据需要调整这个值
if len(rewards) > max_display_points:
    rewards = rewards[:max_display_points]
    print(f"限制显示前 {max_display_points} 个数据点")
else:
    print(f"显示全部 {len(rewards)} 个数据点")

print(f"数据统计: 长度={len(rewards)}, 平均值={np.mean(rewards):.4f}, 最大值={np.max(rewards):.4f}, 最小值={np.min(rewards):.4f}")

# 简单移动平均
window = 100
weights = np.ones(window) / window
smoothed = np.convolve(rewards, weights, mode='same')  # 改为 'same' 模式，保持相同长度

# 或者使用 'valid' 模式但调整x轴
# smoothed = np.convolve(rewards, weights, mode='valid')
# x_smooth = np.arange(window//2, len(rewards) - window//2 + 1)  # 注意这个调整

# 绘制
plt.figure(figsize=(12, 6))

# 绘制原始数据（半透明）
plt.plot(rewards, alpha=0.2, color='blue', label='原始奖励', linewidth=0.5)

# 绘制平滑后的数据
x_smooth = np.arange(len(smoothed))  # x轴与平滑数据长度相同
plt.plot(x_smooth, smoothed, 'r-', linewidth=2, label=f'平滑 (窗口={window})')

plt.xlabel('训练步数/回合数', fontsize=12)
plt.ylabel('奖励值', fontsize=12)
plt.title(f'Reward 曲线 (显示前 {len(rewards)} 个点)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# 添加统计信息
plt.text(0.02, 0.98,
         f"数据点: {len(rewards)}\n平均值: {np.mean(rewards):.3f}\n最大值: {np.max(rewards):.3f}\n最小值: {np.min(rewards):.3f}",
         transform=plt.gca().transAxes,
         fontsize=9,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()

# 保存图表
plt.savefig('reward_curve_limited.png', dpi=300, bbox_inches='tight')
print(f"图表已保存为 'reward_curve_limited.png'")