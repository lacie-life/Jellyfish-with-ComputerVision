import open3d as o3d
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pointcloud = o3d.io.read_point_cloud("../data/test_1.pcd")
    print(pointcloud)
    print(np.asarray(pointcloud.points))
    o3d.visualization.draw_geometries([pointcloud])

