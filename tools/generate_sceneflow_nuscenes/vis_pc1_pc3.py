import open3d as o3d
import numpy as np
import os.path as osp

dataroot = '/mnt/data/scene_flow/3DSFLabelling/SF_Pose_nuScenes_with_ground/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243697666/'
# 加载第一组点云
points1 = np.load(osp.join(dataroot, 'pc1.npy'))
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(points1)
# 设置第一组点云的颜色，例如红色
pcd1.paint_uniform_color([1, 0, 0])  # RGB，值在0-1之间

# 加载第二组点云
points2 = np.load(osp.join(dataroot, 'pc3.npy'))
pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(points2)
# 设置第二组点云的颜色，例如绿色
pcd2.paint_uniform_color([0, 1, 0])  # RGB，值在0-1之间

# 将两个点云合并到一个可视化窗口中
o3d.visualization.draw_geometries([pcd2],
                                  window_name="双点云可视化",
                                  width=800,
                                  height=600,
                                  left=50,
                                  top=50,
                                  point_show_normal=False)
