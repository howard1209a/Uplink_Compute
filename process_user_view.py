import pandas as pd
import math
import os
from collections import Counter

# 定义文件夹路径
video_num = 10

folder_path = "D:\\codes\\Uplink_Compute\\source\\video_" + str(video_num) + "\\view"
save_path = "D:\\codes\\Uplink_Compute\\source\\video_" + str(video_num) + "\\view.csv"
# 新增：单个用户保存路径
user_save_dir = "D:\\codes\\Uplink_Compute\\source\\video_" + str(video_num) + "\\view\\stats"

# 创建用户统计保存目录
os.makedirs(user_save_dir, exist_ok=True)

# 获取文件夹中所有的CSV文件
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# ERP图像参数
ERP_WIDTH = 2560
ERP_HEIGHT = 1440


def erp_to_spherical(x, y):
    """将ERP像素坐标转换为球面坐标（经纬度）"""
    u = x / ERP_WIDTH
    v = y / ERP_HEIGHT
    longitude = u * 2.0 * math.pi
    latitude = v * math.pi
    return longitude, latitude


def spherical_to_cartesian(lon, lat):
    """将球面坐标转换为笛卡尔坐标"""
    x = math.sin(lon) * math.sin(lat)
    y = math.cos(lat)
    z = math.cos(lon) * math.sin(lat)
    return x, y, z


def erp_to_cmp_face(x, y):
    """将ERP像素坐标映射到CMP立方体贴图的面"""
    lon, lat = erp_to_spherical(x, y)
    x_cart, y_cart, z_cart = spherical_to_cartesian(lon, lat)

    abs_x = abs(x_cart)
    abs_y = abs(y_cart)
    abs_z = abs(z_cart)
    max_abs = max(abs_x, abs_y, abs_z)

    if max_abs == abs_x:
        return 'right' if x_cart > 0 else 'left'
    elif max_abs == abs_y:
        return 'top' if y_cart > 0 else 'bottom'
    else:
        return 'front' if z_cart > 0 else 'back'


def process_user_data(df):
    """处理单个用户的数据并返回统计结果"""
    df_clean = df.dropna(subset=['GazePoint_x', 'GazePoint_y']).copy()

    if len(df_clean) == 0:
        return None

    # 计算面分布
    df_clean['face'] = df_clean.apply(
        lambda row: erp_to_cmp_face(row['GazePoint_x'], row['GazePoint_y']),
        axis=1
    )

    return df_clean[['AdjustedTime', 'face']]


# 面名称到编号的映射
face_to_number = {
    'right': 1,  # 1号对应右面
    'left': 2,  # 2号对应左面
    'top': 3,  # 3号对应上面
    'bottom': 4,  # 4号对应下面
    'front': 5,  # 5号对应前面
    'back': 6,  # 6号对应后面
    'N/A': 0
}


def generate_output_format(user_data, user_name):
    """生成与全部用户相同格式的输出数据"""
    if user_data is None or len(user_data) == 0:
        return None

    # 获取时间范围
    min_time = user_data['AdjustedTime'].min()
    max_time = user_data['AdjustedTime'].max()

    # 确保统计完整的60秒
    if max_time - min_time > 60:
        max_time = min_time + 60

    # 按2秒时间窗口统计
    time_window = 2.0
    current_time = min_time
    window_results = []

    while current_time < max_time:
        window_end = current_time + time_window
        window_data = user_data[
            (user_data['AdjustedTime'] >= current_time) &
            (user_data['AdjustedTime'] < window_end)
            ]

        if len(window_data) > 0:
            face_counts = window_data['face'].value_counts()
            top_faces = face_counts.head(2)

            window_results.append({
                'top_face_1': top_faces.index[0] if len(top_faces) > 0 else 'N/A',
                'top_face_2': top_faces.index[1] if len(top_faces) > 1 else 'N/A'
            })
        else:
            window_results.append({
                'top_face_1': 'N/A',
                'top_face_2': 'N/A'
            })

        current_time = window_end

    # 创建与全部用户相同格式的输出数据
    high_frequency_faces = []
    for i, row in enumerate(window_results, 1):
        face1_num = face_to_number[row['top_face_1']]
        face2_num = face_to_number[row['top_face_2']]

        if face1_num != 0 and face2_num != 0:
            face_list = f"[{face1_num},{face2_num}]"
        elif face1_num != 0:
            face_list = f"[{face1_num}]"
        else:
            face_list = f"[5]"

        high_frequency_faces.append({
            'index': i,
            'high_frequency_faces': face_list
        })

    return pd.DataFrame(high_frequency_faces)


# 存储所有用户的处理结果
all_face_data = []
user_statistics = {}  # 存储每个用户的统计信息

# 处理每个CSV文件（每个用户）
for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    user_name = os.path.splitext(csv_file)[0]  # 使用文件名作为用户名
    print(f"正在处理用户: {user_name}")

    # 读取用户数据
    df = pd.read_csv(file_path)

    # 处理单个用户数据
    user_face_data = process_user_data(df)

    if user_face_data is None:
        print(f"  用户 {user_name} 没有有效数据，跳过")
        continue

    # 统计用户信息
    total_samples = len(user_face_data)
    print(f"  有效样本数: {total_samples}")

    # 生成与全部用户相同格式的输出
    user_output = generate_output_format(user_face_data, user_name)

    if user_output is not None:
        # 保存单个用户的统计结果（与全部用户相同格式）
        user_save_path = os.path.join(user_save_dir, f"{user_name}_stats.csv")
        user_output.to_csv(user_save_path, index=False, quoting=1)
        print(f"  用户 {user_name} 的统计数据已保存到: {user_save_path}")

        # 统计时间窗口信息
        time_windows = len(user_output)
        valid_windows = len(user_output[user_output['high_frequency_faces'] != '[0]'])

        # 存储用户统计信息
        user_statistics[user_name] = {
            'total_samples': total_samples,
            'total_windows': time_windows,
            'valid_windows': valid_windows
        }

    # 为合并所有用户数据做准备
    if len(user_face_data) > 0:
        # 添加用户名列以便跟踪
        user_face_data['user'] = user_name
        all_face_data.append(user_face_data)

# 合并所有用户的数据
if all_face_data:
    combined_df = pd.concat(all_face_data, ignore_index=True)

    # 按2秒时间窗口统计所有用户的合并数据
    time_window = 2.0
    min_time = combined_df['AdjustedTime'].min()
    max_time = combined_df['AdjustedTime'].max()

    # 确保统计完整的60秒
    if max_time - min_time > 60:
        max_time = min_time + 60

    current_time = min_time
    window_results = []

    while current_time < max_time:
        window_end = current_time + time_window
        window_data = combined_df[
            (combined_df['AdjustedTime'] >= current_time) &
            (combined_df['AdjustedTime'] < window_end)
            ]

        if len(window_data) > 0:
            face_counts = window_data['face'].value_counts()
            top_faces = face_counts.head(2)

            window_results.append({
                'top_face_1': top_faces.index[0] if len(top_faces) > 0 else 'N/A',
                'top_face_2': top_faces.index[1] if len(top_faces) > 1 else 'N/A'
            })
        else:
            window_results.append({
                'top_face_1': 'N/A',
                'top_face_2': 'N/A'
            })

        current_time = window_end

    # 创建输出数据
    high_frequency_faces = []
    for i, row in enumerate(window_results, 1):
        face1_num = face_to_number[row['top_face_1']]
        face2_num = face_to_number[row['top_face_2']]

        if face2_num != 0:
            face_list = f"[{face1_num},{face2_num}]"
        elif face1_num != 0:
            face_list = f"[{face1_num}]"
        else:
            face_list = f"[5]"

        high_frequency_faces.append({
            'index': i,
            'high_frequency_faces': face_list
        })

    # 保存所有用户合并结果
    output_df = pd.DataFrame(high_frequency_faces)
    output_df.to_csv(save_path, index=False, quoting=1)
    print(f"\n所有用户合并结果已保存到: {save_path}")
    print(f"总用户数: {len(user_statistics)}")
    print(f"总时间窗口数: {len(output_df)}")

else:
    print("\n没有有效用户数据，程序退出")
    exit()
