# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from nuscenes.nuscenes import NuScenes
# from nuscenes.utils.data_classes import Box
# from nuscenes.utils.splits import create_splits_scenes
#
#
# def visualize_first_and_last_frame(nusc, scene_token):
#     """
#     根据场景的token可视化第一帧和最后一帧的前视相机（CAM_FRONT）图像，并输出时间戳
#     Args:
#         nusc: NuScenes 数据集实例
#         scene_token: 场景的token
#     """
#     # 获取场景信息
#     scene = nusc.get('scene', scene_token)
#
#     # 获取该场景的第一个样本token
#     sample_token = scene['first_sample_token']
#
#     # 获取场景中最后一帧的token
#     last_sample_token = scene['last_sample_token']
#
#     # 获取第一帧数据
#     first_sample = nusc.get('sample', sample_token)
#     first_cam_front_token = first_sample['data']['CAM_FRONT']
#     first_cam_front_data = nusc.get('sample_data', first_cam_front_token)
#     first_image_path = os.path.join(nusc.dataroot, first_cam_front_data['filename'])
#     first_image = plt.imread(first_image_path)
#     first_timestamp = first_cam_front_data['timestamp']
#
#     # first_lidar_token = first_sample['data']['LIDAR_TOP']
#     # first_lidar_data = nusc.get('sample_data', first_lidar_token)
#     # first_lidar_timestamp = first_lidar_data['timestamp']
#
#     # 获取最后一帧数据
#     last_sample = nusc.get('sample', last_sample_token)
#     last_cam_front_token = last_sample['data']['CAM_FRONT']
#     last_cam_front_data = nusc.get('sample_data', last_cam_front_token)
#     last_image_path = os.path.join(nusc.dataroot, last_cam_front_data['filename'])
#     last_image = plt.imread(last_image_path)
#     last_timestamp = last_cam_front_data['timestamp']
#
#     # last_lidar_token = last_sample['data']['LIDAR_TOP']
#     # last_lidar_data = nusc.get('sample_data', last_lidar_token)
#     # last_lidar_timestamp = last_lidar_data['timestamp']
#
#     # # 输出时间戳
#     # print(f"First frame timestamp: {first_lidar_timestamp}")
#     # print(f"Last frame timestamp: {last_lidar_timestamp}")
#
#     # 可视化第一帧和最后一帧的前视相机图像
#     fig, axes = plt.subplots(1, 2, figsize=(15, 10))
#
#     # 显示第一帧图像
#     axes[0].imshow(first_image)
#     axes[0].axis('off')
#     axes[0].set_title(f'First Frame')
#
#     # 显示最后一帧图像
#     axes[1].imshow(last_image)
#     axes[1].axis('off')
#     axes[1].set_title(f'Last Frame')
#
#     plt.show()
#
#
# # 使用示例
# if __name__ == "__main__":
#     # 初始化 NuScenes 数据集
#     nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes', verbose=True)
#
#     # 'scene-0170'
#     scene_token = '73030fb67d3c46cfb5e590168088ae39'
#     # lidar top timestamp: first-1526915617047315; last-1526915636396486
#     # 'scene-0171'
#     # scene_token = '2ffd7e2a1daf4b928464ddb2ed3dca59'
#     # lidar top timestamp: first-1526915637046794; last-1526915656447362
#
#     # 可视化第一帧和最后一帧的前视相机图像
#     visualize_first_and_last_frame(nusc, scene_token)


# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from nuscenes.nuscenes import NuScenes
#
#
# def visualize_all_frames(nusc, scene_token):
#     """
#     根据场景的 token 可视化该场景所有的前视相机（CAM_FRONT）图像，并输出时间戳。
#     Args:
#         nusc: NuScenes 数据集实例
#         scene_token: 场景的 token
#     """
#     # 获取场景信息
#     scene = nusc.get('scene', scene_token)
#
#     # 获取该场景的第一个样本 token
#     current_sample_token = scene['first_sample_token']
#
#     # 遍历整个场景
#     while current_sample_token != '':
#         # 获取当前帧数据
#         current_sample = nusc.get('sample', current_sample_token)
#
#         # 获取当前帧的前视相机数据
#         cam_front_token = current_sample['data']['CAM_FRONT']
#         cam_front_data = nusc.get('sample_data', cam_front_token)
#         image_path = os.path.join(nusc.dataroot, cam_front_data['filename'])
#         timestamp = cam_front_data['timestamp']
#
#         # 加载图像
#         image = plt.imread(image_path)
#
#         # 显示图像和时间戳
#         plt.figure(figsize=(8, 6))
#         plt.imshow(image)
#         plt.axis('off')
#         plt.title(f"Timestamp: {timestamp}")
#         plt.show()
#
#         # 获取下一帧 token
#         current_sample_token = current_sample['next']
#
#
# # 使用示例
# if __name__ == "__main__":
#     # 初始化 NuScenes 数据集
#     nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes', verbose=True)
#
#     # 指定场景 token
#     scene_token = '73030fb67d3c46cfb5e590168088ae39'
#
#     # 可视化该场景所有的前视相机图像
#     visualize_all_frames(nusc, scene_token)

import numpy as np
import open3d as o3d

# 加载存储在 .npy 文件中的点云数据
npy_file = ("/home/chunyu/WorkSpace/BugStudio/SurroundOcc/output/dense_points/"
            "n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530449377/pc1.npy")
point_cloud_data = np.load(npy_file)  # 加载点云数据

# 检查点云数据的形状
print(f"Point cloud data shape: {point_cloud_data.shape}")

# 如果点云数据是 Nx3 或 Nx4，提取前三列作为 (x, y, z) 坐标
if point_cloud_data.shape[1] >= 3:
    points = point_cloud_data[:, :3]
else:
    raise ValueError("点云数据的形状不正确，应该是 Nx3 或 Nx4 的格式。")

# 创建 Open3D 的点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 可视化点云
o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Visualization")
