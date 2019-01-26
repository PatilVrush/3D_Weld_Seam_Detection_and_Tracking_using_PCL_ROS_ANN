
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <mutex>
#include <thread>
#include <chrono>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <ros/spinner.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <kinect2_bridge/kinect2_definitions.h>
#include <std_msgs/Float32.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/pcl_config.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/organized_edge_detection.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/common/common_headers.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/ransac.h>
#include <geometry_msgs/Quaternion.h>
#include <pcl/segmentation/extract_clusters.h>
#include <std_msgs/Float64.h>
#include <unistd.h>
#include <pcl/io/ply_io.h>

int w=0;
int a=0;
int g=0;
int cloud_initialise=0;
int mm=0;

cv::Mat color_img, depth_img;
cv::Mat lookupX, lookupY;
cv::Mat cameraMatrixDepth = cv::Mat::zeros(3, 3, CV_64F);
cv::Mat cameraMatrixColor = cv::Mat::zeros(3, 3, CV_64F);

sensor_msgs::CameraInfo cameraInfoColor;
sensor_msgs::CameraInfo cameraInfoDepth;


Eigen::Vector4f centroid_c1; 
pcl::PointXYZRGBA centre_c1;

Eigen::Vector4f centroid_c2; 
pcl::PointXYZRGBA centre_c2;

pcl::PointXYZRGBA centre2;
Eigen::Vector4f centroid2;

pcl::PointXYZRGBA to_push;
pcl::PointXYZRGBA searchPoint;

ros::Publisher pub_weld_line;

pcl::visualization::PCLVisualizer::Ptr visualizer(new pcl::visualization::PCLVisualizer("Cloud Viewer"));

pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ef(new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr boundary_edges (new pcl::PointCloud<pcl::PointXYZRGBA>);  
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr occluding_edges (new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr rgb_edges(new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_end(new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr high_curvature_edges (new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr occluded_edges (new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered_fin (new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_cluster1 (new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_cluster2 (new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered_out (new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered_o (new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr segmented_edge (new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr semi_edge (new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZRGBA> ());
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_plane_fin (new pcl::PointCloud<pcl::PointXYZRGBA> ());


pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
pcl::PointCloud<pcl::Normal>::Ptr cloud_filtered_normals (new pcl::PointCloud<pcl::Normal>);

pcl::ModelCoefficients::Ptr coefficients_plane (new pcl::ModelCoefficients), coefficients_cylinder (new pcl::ModelCoefficients);
pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices), inliers_cylinder (new pcl::PointIndices);


void cloud_cb1 (const sensor_msgs::ImageConstPtr& image_store)
{   
    cv_bridge::CvImageConstPtr pCvImage;
    pCvImage = cv_bridge::toCvShare(image_store, image_store->encoding);
    pCvImage->image.copyTo(color_img);
    OUT_INFO("found color image...");
    if(color_img.type() == CV_16U)
    {
        cv::Mat tmp;
        color_img.convertTo(tmp, CV_8U, 0.02);
        cv::cvtColor(tmp, color_img, CV_GRAY2BGR);
    }
}


void cloud_cb2 (const sensor_msgs::ImageConstPtr& depth_store)
{
    cv_bridge::CvImageConstPtr pCvImage2;
    pCvImage2 = cv_bridge::toCvShare(depth_store, depth_store->encoding);
    pCvImage2->image.copyTo(depth_img);
    OUT_INFO("found depth image...");
}


void cloud_cb3 (const sensor_msgs::CameraInfoConstPtr& cameraInfo1)
{
    double *itC = cameraMatrixColor.ptr<double>(0, 0);
    for(size_t i = 0; i < 9; ++i, ++itC)
    {
        *itC = cameraInfo1->K[i];
    }
    cameraInfoColor=*cameraInfo1;
}
 

void cloud_cb4 (const sensor_msgs::CameraInfoConstPtr& cameraInfo2)
{

    double *itC = cameraMatrixDepth.ptr<double>(0, 0);
    for(size_t i = 0; i < 9; ++i, ++itC)
    {
        *itC = cameraInfo2->K[i];
    }
    cameraInfoDepth=*cameraInfo2;
}


void createLookup(size_t width, size_t height)
{
    const float fx = 1.0f / cameraMatrixColor.at<double>(0, 0);
    const float fy = 1.0f / cameraMatrixColor.at<double>(1, 1);
    const float cx = cameraMatrixColor.at<double>(0, 2);
    const float cy = cameraMatrixColor.at<double>(1, 2);
    float *it;
    lookupY = cv::Mat(1, height, CV_32F);
    it = lookupY.ptr<float>();
    for(size_t r = 0; r < height; ++r, ++it)
    {
        *it = (r - cy) * fy;
    }
    lookupX = cv::Mat(1, width, CV_32F);
    it = lookupX.ptr<float>();
    for(size_t c = 0; c < width; ++c, ++it)
    {
        *it = (c - cx) * fx;
    }
}


void createcloud ()
{
    if(cloud_initialise==0)
    {
    cloud = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBA>());
    cloud->height = color_img.rows;
    cloud->width = color_img.cols;
    cloud->is_dense = false;
    cloud->points.resize(cloud->height * cloud->width);
    createLookup(color_img.cols, color_img.rows);
    cloud_initialise=cloud_initialise+1;
    } 

    const float badPoint = std::numeric_limits<float>::quiet_NaN();

    #pragma omp parallel for
    for(int r = 0; r < depth_img.rows; ++r)
    {
        pcl::PointXYZRGBA *itP = &cloud->points[r * depth_img.cols];
        const uint16_t *itD = depth_img.ptr<uint16_t>(r);
        const cv::Vec3b *itC = color_img.ptr<cv::Vec3b>(r);
        const float y = lookupY.at<float>(0, r);
        const float *itX = lookupX.ptr<float>();

        for(size_t c = 0; c < (size_t)depth_img.cols; ++c, ++itP, ++itD, ++itC, ++itX)
        {
            register const float depthValue = *itD / 1000.0f;
            // Check for invalid measurements
            if(*itD == 0)
            {
            // not valid
            itP->x = itP->y = itP->z = badPoint;
            itP->rgba = 0;
            continue;
            }
            itP->z = depthValue;
            itP->x = *itX * depthValue;
            itP->y = y * depthValue;
            itP->b = itC->val[0];
            itP->g = itC->val[1];
            itP->r = itC->val[2];
            itP->a = 255;
        }
    }
}


void processing ()
{

    createcloud();


    pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGBA>);
    pcl::ExtractIndices<pcl::PointXYZRGBA> extract;

    // **************** Emptying all the cloud ************** //

    pcl::PointIndices::Ptr inliers_1 (new pcl::PointIndices);

    for (int i = 0; i < (*cloud_cluster1).size(); i++)
    {
        if (g==0) //remove all points
        {
            inliers_1->indices.push_back(i);
        }

    }

    extract.setInputCloud(cloud_cluster1);
    extract.setIndices(inliers_1);
    extract.setNegative(true);
    extract.filter(*cloud_cluster1);


     pcl::PointIndices::Ptr inliers_r (new pcl::PointIndices);

    for (int i = 0; i < (*cloud_ef).size(); i++)
    {
        if (g==0) //remove all points
        {
            inliers_r->indices.push_back(i);
        }

    }

    extract.setInputCloud(cloud_ef);
    extract.setIndices(inliers_r);
    extract.setNegative(true);
    extract.filter(*cloud_ef);

        
        
    pcl::PointIndices::Ptr inliers_2 (new pcl::PointIndices);

    for (int i = 0; i < (*cloud_cluster2).size(); i++)
    {
        
        if (g==0) 
        {
            inliers_2->indices.push_back(i);
        }

    }


    extract.setInputCloud(cloud_cluster2);
    extract.setIndices(inliers_2);
    extract.setNegative(true);
    extract.filter(*cloud_cluster2);



    pcl::PointIndices::Ptr inliers_e (new pcl::PointIndices);

    for (int i = 0; i < (*segmented_edge).size(); i++)
       {

        if (g==0) 
        {
            inliers_e->indices.push_back(i);
        }

    }


    extract.setInputCloud(segmented_edge);
    extract.setIndices(inliers_e);
    extract.setNegative(true);
    extract.filter(*segmented_edge);


   // ************* Segmenting the weld piece ***************** //    
  
    pcl::PassThrough<pcl::PointXYZRGBA> pass;
    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0.0, 1.0);
    pass.filter (*cloud_filtered_o);
    std::cerr << "PointCloud has: " << cloud->points.size () << " data points." << std::endl;
    std::cerr << "PointCloud after Pass through filtering has: " << cloud_filtered_o->points.size () << " data points." << std::endl;

     

    pcl::VoxelGrid<pcl::PointXYZRGBA> sr;
    sr.setInputCloud (cloud_filtered_o);
    sr.setLeafSize (0.006, 0.006, 0.006);
    sr.filter (*cloud_filtered);
    std::cerr << "Voxel Filtered PointCloud has: " << cloud_filtered->points.size () << " data points." << std::endl;

    pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> nes;
    nes.setKSearch (15);
    nes.setSearchMethod (tree);
    nes.setInputCloud (cloud_filtered);
    nes.compute (*cloud_filtered_normals);

    pcl::SACSegmentationFromNormals<pcl::PointXYZRGBA, pcl::Normal> seg;
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_CYLINDER);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setNormalDistanceWeight (0.1);
    seg.setMaxIterations (200);
    seg.setDistanceThreshold (0.05);
    seg.setRadiusLimits (0, 0.2);
    seg.setInputCloud (cloud_filtered);
    seg.setInputNormals (cloud_filtered_normals);
    seg.segment (*inliers_cylinder, *coefficients_cylinder);     

    extract.setInputCloud (cloud_filtered);
    extract.setIndices (inliers_cylinder);
    extract.setNegative (false);
    extract.filter (*cloud_plane_fin);

    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBA> sor;
    sor.setInputCloud (cloud_plane_fin);
    sor.setMeanK (450);
    sor.setStddevMulThresh (1.0);
    sor.filter (*cloud_plane);
    

    // ************* Clustering the segmneted plane in two halves ***************** //
     
    tree->setInputCloud (cloud_filtered);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGBA> ec;
    ec.setClusterTolerance (0.008); // 1cm
    ec.setMinClusterSize (100);
    ec.setMaxClusterSize (2500000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud_plane);
    ec.extract (cluster_indices);
            
            

    int indx =0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
        if(indx==0)
        {
            for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
                cloud_cluster1->points.push_back (cloud_plane->points[*pit]); //*
                cloud_cluster1->width = cloud_cluster1->points.size ();
                cloud_cluster1->height = 1;
                cloud_cluster1->is_dense = true;

                std::cout << "PointCloud representing first cluster: " << cloud_cluster1->points.size () << " data points." << std::endl;
                std::stringstream ss;
                indx=indx+1;
        
        }

        else if(indx==1)
        {
            for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
            cloud_cluster2->points.push_back (cloud_plane->points[*pit]); //*
            cloud_cluster2->width = cloud_cluster2->points.size ();
            cloud_cluster2->height = 1;
            cloud_cluster2->is_dense = true;    

            std::cout << "PointCloud representing second cluster: " << cloud_cluster2->points.size () << " data points." << std::endl;
            indx=indx+1; // for verifying
            
        }

    }

    pcl::compute3DCentroid(*cloud_cluster1,centroid_c1); 
    centre_c1.x=centroid_c1[0];
    centre_c1.y=centroid_c1[1];
    centre_c1.z=centroid_c1[2];
    centre_c1.r=0;
    centre_c1.g=0;
    centre_c1.b=255;
    cloud_cluster1->points.push_back(centre_c1);
    OUT_INFO("Green Centroid");
    cout<<centre_c1<<endl;
    OUT_INFO("_______________________________________________");

    pcl::compute3DCentroid(*cloud_cluster2,centroid_c2); 
    centre_c2.x=centroid_c2[0];
    centre_c2.y=centroid_c2[1];
    centre_c2.z=centroid_c2[2];
    centre_c2.r=0;
    centre_c2.g=0;
    centre_c2.b=255;
    cloud_cluster2->points.push_back(centre_c2);
    OUT_INFO("Red Centroid");
    cout<<centre_c2<<endl;
    OUT_INFO("_____________________________________________");

    // ************* Edge Detection ***************** //


    pcl::IntegralImageNormalEstimation<pcl::PointXYZRGBA, pcl::Normal> ne;
    ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
    ne.setNormalSmoothingSize (10.0f);
    ne.setBorderPolicy (ne.BORDER_POLICY_MIRROR);
    ne.setInputCloud (cloud);
    ne.compute (*cloud_normals);

    pcl::OrganizedEdgeFromRGBNormals<pcl::PointXYZRGBA, pcl::Normal, pcl::Label> oed;
    oed.setInputNormals (cloud_normals);
    oed.setInputCloud (cloud);
    oed.setDepthDisconThreshold (0.01); // 1cm
    oed.setMaxSearchNeighbors (100);
    pcl::PointCloud<pcl::Label> labels;
    std::vector<pcl::PointIndices> label_indices;
    oed.compute (labels, label_indices);

    pcl::copyPointCloud (*cloud, label_indices[0].indices, *boundary_edges);
    pcl::copyPointCloud (*cloud, label_indices[1].indices, *occluding_edges);
    pcl::copyPointCloud (*cloud, label_indices[2].indices, *occluded_edges);
    pcl::copyPointCloud (*cloud, label_indices[3].indices, *high_curvature_edges);
    pcl::copyPointCloud (*cloud, label_indices[4].indices, *rgb_edges);

    //std::cerr << "Scene PointCloud has: " << cloud->points.size () << " points." << std::endl;
    std::cerr << "RGB has: " << rgb_edges->points.size () << " points." << std::endl;
    //std::cerr << "occluded has: " << occluding_edges->points.size () << " points." << std::endl;
    //std::cerr << "High Curvature Edges has: " << high_curvature_edges->points.size () << " points." << std::endl;


    // ************* Weld Line Detection ***************** //

  

    if(indx==2) //check if two clusters were detected 
    {   


        if(centre_c1.x > centre_c2.x) // if blue on right
        {


            pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
            kdtree.setInputCloud (cloud_cluster1);
            float r=0.1;    //0.1
            std::vector<int> pointIdxNKNSearch(r);
            std::vector<float> pointNKNSquaredDistance(r);

        for (size_t j = 0; j < rgb_edges->points.size (); ++j)
        {
            searchPoint.x = rgb_edges->points[j].x;
            searchPoint.y = rgb_edges->points[j].y;
            searchPoint.z = rgb_edges->points[j].z;
            //std::cerr << "Edge point is : " << rgb_edges->points[j].x  << "  " << rgb_edges->points[j].y << "  " << rgb_edges->points[j].z << std::endl;

            if (kdtree.radiusSearch (searchPoint, r, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
                {
         
                    for (size_t i = 0; i < pointIdxNKNSearch.size (); ++i)
                    {

                        to_push.x = cloud_cluster1->points[ pointIdxNKNSearch[i] ].x;
                        to_push.y = cloud_cluster1->points[ pointIdxNKNSearch[i] ].y;
                        to_push.z = cloud_cluster1->points[ pointIdxNKNSearch[i] ].z;
                        //cloud_ef->points.push_back(to_push); //To find edges
                        cloud_end->points.push_back(to_push);
                    }
                 
                     
                    pcl::compute3DCentroid(*cloud_end,centroid2); 
                    centre2.x= centroid2[0];
                    centre2.y=centroid2[1];
                    centre2.z=centroid2[2];
                    centre2.r=0;
                    centre2.g=0;
                    centre2.b=0;
                    cloud_ef->points.push_back(centre2);
                   //if( (centroid2[0]- rgb_edges->points[j].x > 0.01)  && (centroid2[0] - rgb_edges->points[j].x < 0.055)  &&  (rgb_edges->points[j].y -centroid2[1] < 0.036) && (centroid2[1]- rgb_edges->points[j].y < 0.019))
                   if((centroid2[0]- rgb_edges->points[j].x > 0.025) && (centroid2[0] - rgb_edges->points[j].x < 0.05) &&  (rgb_edges->points[j].y -centroid2[1] < 0.035) && (centroid2[1]- rgb_edges->points[j].y < 0.026))   
                        segmented_edge->points.push_back(rgb_edges->points[j]);

                    //std::cerr << "Corresponding Centroid point is : " << centroid2[0] << "  " << centroid2[1] <<"  "<< centroid2[2] << std::endl;
                    //std::cerr << "************************* "  << std::endl;
                }    


    
            pcl::PointIndices::Ptr inliers_t (new pcl::PointIndices);
        
            for (int i = 0; i < (*cloud_end).size(); i++)
            {
                if (g==0)
                {
                inliers_t->indices.push_back(i);
                }

            }
            extract.setInputCloud(cloud_end);
            extract.setIndices(inliers_t);
            extract.setNegative(true);
            extract.filter(*cloud_end); // Flushing cloud_end so as to store new neighbouring points 

            }
        }


         else // if red on right
        {


            pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
            kdtree.setInputCloud (cloud_cluster2);
            float r=0.1;  
            std::vector<int> pointIdxNKNSearch(r);
            std::vector<float> pointNKNSquaredDistance(r);

        for (size_t j = 0; j < rgb_edges->points.size (); ++j)
        {
            searchPoint.x = rgb_edges->points[j].x;
            searchPoint.y = rgb_edges->points[j].y;
            searchPoint.z = rgb_edges->points[j].z;
            //std::cerr << "Edge point is : " << rgb_edges->points[j].x  << "  " << rgb_edges->points[j].y << "  " << rgb_edges->points[j].z << std::endl;

            if (kdtree.radiusSearch (searchPoint, r, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
                {
         
                    for (size_t i = 0; i < pointIdxNKNSearch.size (); ++i)
                    {

                        to_push.x = cloud_cluster2->points[ pointIdxNKNSearch[i] ].x;
                        to_push.y = cloud_cluster2->points[ pointIdxNKNSearch[i] ].y;
                        to_push.z = cloud_cluster2->points[ pointIdxNKNSearch[i] ].z;
                        //cloud_ef->points.push_back(to_push); //To find edges
                        cloud_end->points.push_back(to_push);
                    }
                 
                     
                    pcl::compute3DCentroid(*cloud_end,centroid2); 
                    centre2.x= centroid2[0];
                    centre2.y=centroid2[1];
                    centre2.z=centroid2[2];
                    centre2.r=0;
                    centre2.g=255;
                    centre2.b=255;
                    cloud_ef->points.push_back(centre2);
                 //if((centroid2[0]- rgb_edges->points[j].x > 0.01) && (centroid2[0] - rgb_edges->points[j].x < 0.055) &&  (rgb_edges->points[j].y -centroid2[1] < 0.036) && (centroid2[1]- rgb_edges->points[j].y < 0.019))
                    if((centroid2[0]- rgb_edges->points[j].x > 0.025) && (centroid2[0] - rgb_edges->points[j].x < 0.05) && (rgb_edges->points[j].y -centroid2[1] < 0.035) && (centroid2[1]- rgb_edges->points[j].y < 0.026))
                        segmented_edge->points.push_back(rgb_edges->points[j]);

                    //std::cerr << "Corresponding Centroid point is : " << centroid2[0] << "  " << centroid2[1] <<"  "<< centroid2[2] << std::endl;
                    //std::cerr << "************************* "  << std::endl;
                }


    
            pcl::PointIndices::Ptr inliers_t (new pcl::PointIndices);
        
            for (int i = 0; i < (*cloud_end).size(); i++)
            {
                if (g==0) 
                {
                inliers_t->indices.push_back(i);
                }

            }
            extract.setInputCloud(cloud_end);
            extract.setIndices(inliers_t);
            extract.setNegative(true);
            extract.filter(*cloud_end); // Flushing cloud_end so as to store new neighbouring points 

            }
        }


    }

if (mm==0 && indx==2)
{
    int l=0;
            std::cout<< "****************************************************************************************************" <<std::endl;

            std::cerr << "Weld has: " << segmented_edge->points.size () << " points." << std::endl;

            pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
            kdtree.setInputCloud (segmented_edge);
            float r2=0.01;    //0.1
            std::vector<int> pointIdxNKNSearcht(r2);
            std::vector<float> pointNKNSquaredDistancet(r2);
            for (size_t j = 0; j < segmented_edge->points.size (); ++j)
            {
            searchPoint.x = segmented_edge->points[j].x;
            searchPoint.y = segmented_edge->points[j].y;
            searchPoint.z = segmented_edge->points[j].z;
            //std::cerr << "Edge point is : " << rgb_edges->points[j].x  << "  " << rgb_edges->points[j].y << "  " << rgb_edges->points[j].z << std::endl;

           
            if (kdtree.radiusSearch (searchPoint, r2, pointIdxNKNSearcht, pointNKNSquaredDistancet) > 1)
                {
                    
                    
                        to_push.x = segmented_edge->points[j].x;
                        to_push.y = segmented_edge->points[j].y;
                        to_push.z = segmented_edge->points[j].z;
                        semi_edge->points.push_back(to_push);
                    std::cerr << semi_edge->points[j].x << "," << semi_edge->points[j].y << "," << semi_edge->points[j].z<< std::endl;
                    l=l+1;

                    //std::cout<< l <<std::endl;
                

         
                   }


            }
             std::cout<< l <<std::endl;
            std::cout<< "****************************************************************************************************" <<std::endl;
            mm=mm+1;
}

/*




            if (indx==2 && segmented_edge->points.size()>30)
            {   
                if(w==0)
                {
                    pcl::io::savePLYFileBinary("rgb_edges_curve_2.ply" , *rgb_edges);
                    for (size_t j = 0; j < segmented_edge->points.size (); ++j)
                    {
                    
                    geometry_msgs::Quaternion msg2;
                    msg2.x = segmented_edge->points[j].x;
                    msg2.y = segmented_edge->points[j].y;
                    msg2.z = segmented_edge->points[j].z;
                    msg2.w = 0;

                    std::cerr << segmented_edge->points[j].x << "," << segmented_edge->points[j].y << "," << segmented_edge->points[j].z<< std::endl;
                    //std::cerr << "y:" << segmented_edge->points[j].y  << std::endl;
                    //std::cerr << "z:" << segmented_edge->points[j].z  << std::endl;

                    pub_weld_line.publish(msg2);
                   // std::cout<< j <<std::endl;
                    usleep(10000);
                    }  
                    w=w+1;  
                     
                }
            }  

        //std::cerr << "Weld line has: " << segmented_edge->points.size () << " points." << std::endl;
        //std::cerr << "Weld line points are: " << segmented_edge->points[j].x  << "  " << segmented_edge->points[j].y << "  " << segmented_edge->points[j].z << std::endl;

        
     OUT_INFO("********************************************************");

}


*/
}

void cloudViewer(const std_msgs::Float32ConstPtr& number)
{
    if(a==0)
    {
        //OUT_INFO("In cloud viewer");
        processing();

       // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> cloud_1 (rgb_edges, 1, 0, 0);
       // visualizer->addPointCloud (rgb_edges, cloud_1, "cloud");
       // visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "cloud");

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> c1 (cloud_cluster1, 0, 0, 255);
        visualizer->addPointCloud (cloud_cluster1, c1, "cloud_cluster1");
        visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "cloud_cluster1");

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> c2 (cloud_cluster2, 255, 0, 0);
        visualizer->addPointCloud (cloud_cluster2, c2, "cloud_cluster2");
        visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "cloud_cluster2");

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> wl (semi_edge, 0, 0, 0);
        visualizer->addPointCloud (semi_edge, wl, "weld");
        visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "weld");
/*
         pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> wl (cloud_ef, 0, 0, 0);
        visualizer->addPointCloud (cloud_ef, wl, "centroids");
        visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "centroids");
*/
        visualizer->initCameraParameters();
        visualizer->setBackgroundColor(255, 255, 255);
        visualizer->setShowFPS(true);
        visualizer->setCameraPosition(0, 0, 0, 0, -1, 0);

        //visualizer->addArrow<pcl::PointXYZRGBA, pcl::PointXYZRGBA> (centre_c1, centre_c2, 0, 0,0,"line");
        //visualizer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH , 7, "line");
        a=a+1;
   }

    processing();
   
      // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> cloud_1 (rgb_edges, 1, 0, 0);
      // visualizer->updatePointCloud (rgb_edges, cloud_1, "cloud");
      // visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "cloud");

    
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> c1 (cloud_cluster1, 0, 0, 255);
    visualizer->updatePointCloud (cloud_cluster1, c1, "cloud_cluster1");
    visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "cloud_cluster1");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> c2 (cloud_cluster2, 255, 0, 0);
    visualizer->updatePointCloud (cloud_cluster2, c2, "cloud_cluster2");
    visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "cloud_cluster2");
/*
    visualizer->removeShape("line");
    visualizer->addArrow<pcl::PointXYZRGBA, pcl::PointXYZRGBA> (centre_c1, centre_c2, 0, 0,0, "line");
    visualizer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH , 7, "line");
    //visualizer->updateLine<pcl::PointXYZRGBA, pcl::PointXYZRGBA> (centre_c1, centre_c2, 0, 0,0, "line2");
*/

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> wl (semi_edge, 0, 0, 0);
    visualizer->updatePointCloud (semi_edge, wl, "weld");
    visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "weld");
/*
     pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> wl (cloud_ef, 0, 0, 0);
    visualizer->updatePointCloud (cloud_ef, wl, "centroids");
    visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "centroids");
  */ 
    visualizer->spinOnce(5);

}


int main (int argc, char** argv)
{ 
    ros::init (argc, argv, "my_pcl_tutorial"); // Initialize ROS
    ros::NodeHandle nh;

    ros::Subscriber sub1 = nh.subscribe ("/kinect2/sd/image_ir_rect", 1000, cloud_cb1);
    ros::Subscriber sub2 = nh.subscribe ("/kinect2/sd/image_depth_rect", 1000, cloud_cb2);
    ros::Subscriber sub3 = nh.subscribe ("/kinect2/sd/camera_info", 1000, cloud_cb3);
    ros::Subscriber sub4 = nh.subscribe ("/kinect2/sd/camera_info", 1000, cloud_cb4);
    ros::Subscriber sub5 = nh.subscribe ("invoke_visualizer", 1000, cloudViewer);
    pub_weld_line = nh.advertise<geometry_msgs::Quaternion>("weld_line", 1000);
    ros::spin ();
}
