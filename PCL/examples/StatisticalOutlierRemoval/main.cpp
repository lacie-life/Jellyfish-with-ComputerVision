/*
 * Laser scans typically generate point cloud datasets of varying point densities.
 * Additionally, measurement errors lead to sparse outliers which corrupt the results even more.
 * This complicates the estimation of local point cloud characteristics such as surface normals or curvature changes,
 * leading to erroneous values, which in turn might cause point cloud registration failures.
 * Some of these irregularities can be solved by performing a statistical analysis on each point’s neighborhood,
 * and trimming those which do not meet a certain criterion.
 * Our sparse outlier removal is based on the computation of the distribution of point to neighbors distances in the input dataset.
 * For each point, we compute the mean distance from it to all its neighbors.
 * By assuming that the resulted distribution is Gaussian with a mean and a standard deviation,
 * all points whose mean distances are outside an interval defined by the global distances mean
 * and standard deviation can be considered as outliers and trimmed from the dataset.
 */

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>

int main(int argc, char** argv) {
    std::cout << "Hello, World!" << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);

    // Fill in the cloud data
    pcl::PCDReader reader;
    reader.read<pcl::PointXYZ> ("../../data/table_scene_lms400.pcd", *cloud);

    std::cerr << "Cloud before filtering: " << std::endl;
    std::cerr << *cloud << std::endl;

    // Create filter
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud (cloud);
    sor.setMeanK (50);
    sor.setStddevMulThresh (1.0);
    sor.filter(*cloud_filtered);

    std::cerr << "Cloud after filtering: " << std::endl;
    std::cerr << *cloud_filtered << std::endl;

    pcl::PCDWriter writer;
    writer.write<pcl::PointXYZ> ("../table_scene_lms400_inliers.pcd", *cloud_filtered, false);

    sor.setNegative (true);
    sor.filter (*cloud_filtered);
    writer.write<pcl::PointXYZ> ("../table_scene_lms400_outliers.pcd", *cloud_filtered, false);

    return 0;
}
