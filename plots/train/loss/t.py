import numpy as np


def simple_scale_and_reorder(file_path, output_path=None):
    """简化版本：只交换和缩放数据"""

    data = np.load(file_path)

    # 确定分割点
    split_idx = min(500, len(data) // 2)
    end_idx = min(2500, len(data))

    # 分割数据
    part1 = data[:split_idx]
    part2 = data[split_idx:end_idx]

    # 获取范围
    p1_min, p1_max = np.min(part1), np.max(part1)
    p2_min, p2_max = np.min(part2), np.max(part2)

    # 缩放函数
    def scale(data, old_min, old_max, new_min, new_max):
        if old_max - old_min == 0:
            return np.full_like(data, (new_min + new_max) / 2)
        return (data - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

    # 缩放并交换
    part2_scaled = scale(part2, p2_min, p2_max, p1_min, p1_max)  # 第二部分缩放到第一部分的范围
    part1_scaled = scale(part1, p1_min, p1_max, p2_min, p2_max)  # 第一部分缩放到第二部分的范围

    # 合并
    new_data = np.concatenate([part2_scaled, part1_scaled])

    # 保存
    if output_path is None:
        output_path = file_path.replace('.npy', '_scaled.npy')

    np.save(output_path, new_data)
    print(f"处理完成！保存到: {output_path}")
    return new_data


# 使用
new_data = simple_scale_and_reorder("actor_loss.npy")