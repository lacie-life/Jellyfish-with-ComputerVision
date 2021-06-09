import open3d as o3d
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pointcloud = o3d.io.read_point_cloud("../data/frame_00000.pcd")
    print(pointcloud)
    print(np.asarray(pointcloud.points))
    o3d.visualization.draw_geometries([pointcloud],
                                      zoom = 0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

