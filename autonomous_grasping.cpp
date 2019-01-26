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
#include <pcl/features/shot_omp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>


int i=0;
int a=0;
int t=0;
int cloud_initialise=0;

cv::Mat color_img, depth_img;
cv::Mat lookupX, lookupY;
cv::Mat cameraMatrixDepth = cv::Mat::zeros(3, 3, CV_64F);
cv::Mat cameraMatrixColor = cv::Mat::zeros(3, 3, CV_64F);

sensor_msgs::CameraInfo cameraInfoColor;
sensor_msgs::CameraInfo cameraInfoDepth;

pcl::visualization::PCLVisualizer::Ptr visualizer(new pcl::visualization::PCLVisualizer("Cloud Viewer"));
pcl::PointXYZRGBA centre;
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_temp (new pcl::PointCloud<pcl::PointXYZRGBA>);  
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_scene (new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_scene_filtered (new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered_out (new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_scene_filtered_out (new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scene_keypoints (new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered_out_one (new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_identified (new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered_out_two (new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered_out_three (new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered_out_four (new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
pcl::PointCloud<pcl::Normal>::Ptr cloud_scene_normals (new pcl::PointCloud<pcl::Normal>);
pcl::PointCloud<pcl::SHOT352>::Ptr descriptors (new pcl::PointCloud<pcl::SHOT352> ());
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr rotated_model (new pcl::PointCloud<pcl::PointXYZRGBA> ());
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr off_scene_model (new pcl::PointCloud<pcl::PointXYZRGBA> ());
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr off_scene_model_keypoints (new pcl::PointCloud<pcl::PointXYZRGBA> ());
std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
std::vector<pcl::Correspondences> clustered_corrs;

const std::string cloudName = "rendered";


void 
cloud_cb1 (const sensor_msgs::ImageConstPtr& image_store)
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


void 
cloud_cb2 (const sensor_msgs::ImageConstPtr& depth_store)
{
    cv_bridge::CvImageConstPtr pCvImage2;
    pCvImage2 = cv_bridge::toCvShare(depth_store, depth_store->encoding);
    pCvImage2->image.copyTo(depth_img);
    OUT_INFO("found depth image...");
}

void 
cloud_cb3 (const sensor_msgs::CameraInfoConstPtr& cameraInfo1)
{
    
    double *itC = cameraMatrixColor.ptr<double>(0, 0);
    for(size_t i = 0; i < 9; ++i, ++itC)
    {
      *itC = cameraInfo1->K[i];
    }
     cameraInfoColor=*cameraInfo1;
}
   
void 
cloud_cb4 (const sensor_msgs::CameraInfoConstPtr& cameraInfo2)
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

 // pcl::io::savePCDFileASCII ("hammer_scene_pcd.pcd", *cloud);
 // std::cerr << "Saved " << cloud->points.size () << " data points to hammer_pcd.pcd." << std::endl;
  
  
  
   //read pcd file
  if (t==0)
  {

    if(pcl::io::loadPCDFile ("sphere_pcd.pcd", *target_cloud) == -1){
      std::cout << "pcd file not found" << std::endl;
      exit(-1);
    }

    //if(pcl::io::loadPCDFile ("scene_pcd.pcd", *cloud) == -1){
     // std::cout << "pcd file not found" << std::endl;
      //exit(-1);
   // }

    pcl::PassThrough<pcl::PointXYZRGBA> pass;
    pass.setInputCloud (target_cloud);
    pass.setFilterFieldName ("x");
    pass.setFilterLimits (-0.5, 0.4);       
    pass.filter (*cloud_filtered_out_two);

    pcl::PassThrough<pcl::PointXYZRGBA> pass1;
    pass1.setInputCloud (cloud_filtered_out_two);
    pass1.setFilterFieldName ("z");
    pass1.setFilterLimits (0.0, 0.8);       
    pass1.filter (*cloud_filtered_out_one);


    pcl::VoxelGrid<pcl::PointXYZRGBA> srm;
    srm.setInputCloud (cloud_filtered_out_one);
    srm.setLeafSize (0.004, 0.004, 0.004);
    srm.filter (*cloud_filtered_out);
   
    pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGBA>);
    pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
  
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBA> sor;
    sor.setInputCloud (cloud_filtered_out);
    sor.setMeanK (50);
    sor.setStddevMulThresh (1.0);
    sor.filter (*cloud_filtered); 

    pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> ne;
    ne.setKSearch (50);
    ne.setInputCloud (cloud_filtered);
    ne.setSearchMethod (tree);
    ne.compute (*cloud_normals);

    pcl::UniformSampling<pcl::PointXYZRGBA> uniform_samplingm;
    uniform_samplingm.setInputCloud (cloud_filtered);
    uniform_samplingm.setRadiusSearch (0.001);
    uniform_samplingm.filter (*cloud_temp); 
    
    
    pcl::SHOTEstimationOMP<pcl::PointXYZRGBA, pcl::Normal, pcl::SHOT352> fpfh;
    fpfh.setInputCloud (cloud_temp);
    fpfh.setSearchSurface(cloud_filtered) ;
    fpfh.setInputNormals (cloud_normals);
    fpfh.setSearchMethod (tree);
    fpfh.setRadiusSearch (0.02);
    fpfh.compute (*descriptors);
    
    std::cerr << "Model PointCloud has: " << descriptors->points.size () << " descriptors." << std::endl;

    

    pcl::PassThrough<pcl::PointXYZRGBA> pass_s;
    pass_s.setInputCloud (cloud);
    pass_s.setFilterFieldName ("z");
    pass_s.setFilterLimits (0, 0.7);
    pass_s.filter (*cloud_scene_filtered);



    pcl::VoxelGrid<pcl::PointXYZRGBA> sr;
    sr.setInputCloud (cloud_scene_filtered);
    sr.setLeafSize (0.004, 0.004, 0.004);
    sr.filter (*cloud_scene_filtered_out);

    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBA> sor_s;
    sor_s.setInputCloud (cloud_scene_filtered_out);
    sor_s.setMeanK (50);
    sor_s.setStddevMulThresh (1.0);
    sor_s.filter (*cloud_scene);

    
    
    pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> nes;
    nes.setKSearch (10);
    nes.setSearchMethod (tree);
    nes.setInputCloud (cloud_scene);
    nes.compute (*cloud_scene_normals);
    
    pcl::UniformSampling<pcl::PointXYZRGBA> uniform_sampling;
    uniform_sampling.setInputCloud (cloud_scene);
    uniform_sampling.setRadiusSearch (0.001);
    uniform_sampling.filter (*scene_keypoints);

  
    
    pcl::PointCloud<pcl::SHOT352>::Ptr scene_descriptors (new pcl::PointCloud<pcl::SHOT352> ());
    pcl::SHOTEstimationOMP<pcl::PointXYZRGBA, pcl::Normal, pcl::SHOT352> fpfhs;
    fpfhs.setSearchMethod (tree);
    fpfhs.setRadiusSearch (0.02f);
    fpfhs.setInputCloud (scene_keypoints);
    fpfhs.setSearchSurface (cloud_scene);
    fpfhs.setInputNormals (cloud_scene_normals);
    fpfhs.compute (*scene_descriptors);
    std::cerr << "Scene PointCloud has: " << scene_descriptors->points.size () << " descriptors." << std::endl;
  
    
    pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());
    pcl::KdTreeFLANN<pcl::SHOT352> match_search;
    
    match_search.setInputCloud (descriptors);

  //  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
    for (size_t i = 0; i < scene_descriptors->size (); ++i)
    {
    std::vector<int> neigh_indices (1);
    std::vector<float> neigh_sqr_dists (1);
    if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0])) //skipping NaNs
    {
      continue;
    }
    int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
    if(found_neighs == 1 && neigh_sqr_dists[0] < 0.25f) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
    {
      pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
      model_scene_corrs->push_back (corr);
    }
    }
    std::cout << "Correspondences found: " << model_scene_corrs->size () << std::endl;
    std::cout << "********************************************************" << std::endl;

    pcl::PointCloud<pcl::ReferenceFrame>::Ptr model_rf (new pcl::PointCloud<pcl::ReferenceFrame> ());
    pcl::PointCloud<pcl::ReferenceFrame>::Ptr scene_rf (new pcl::PointCloud<pcl::ReferenceFrame> ());

    pcl::BOARDLocalReferenceFrameEstimation<pcl::PointXYZRGBA, pcl::Normal, pcl::ReferenceFrame> rf_est;
    rf_est.setFindHoles (true);
    rf_est.setRadiusSearch (0.015f);

    rf_est.setInputCloud (cloud_temp);
    rf_est.setInputNormals (cloud_normals);
    rf_est.setSearchSurface (cloud_filtered);
    rf_est.compute (*model_rf);

    rf_est.setInputCloud (scene_keypoints);
    rf_est.setInputNormals (cloud_scene_normals);
    rf_est.setSearchSurface (cloud_scene);
    rf_est.compute (*scene_rf);

    //  Clustering
    pcl::Hough3DGrouping<pcl::PointXYZRGBA, pcl::PointXYZRGBA, pcl::ReferenceFrame, pcl::ReferenceFrame> clusterer;
    clusterer.setHoughBinSize (0.01f);
    clusterer.setHoughThreshold (5.0f);
    clusterer.setUseInterpolation (true);
    clusterer.setUseDistanceWeight (false);

    clusterer.setInputCloud (cloud_temp);
    clusterer.setInputRf (model_rf);
    clusterer.setSceneCloud (scene_keypoints);
    clusterer.setSceneRf (scene_rf);
    clusterer.setModelSceneCorrespondences (model_scene_corrs);

    //clusterer.cluster (clustered_corrs);
    clusterer.recognize (rototranslations, clustered_corrs);
    std::cout << "Model instances found: " << rototranslations.size () << std::endl;

    for (size_t i = 0; i < rototranslations.size (); ++i)
    {
    std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
    std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size () << std::endl;

    // Print the rotation matrix and translation vector
    Eigen::Matrix3f rotation = rototranslations[i].block<3,3>(0, 0);
    Eigen::Vector3f translation = rototranslations[i].block<3,1>(0, 3);

    printf ("\n");
    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
    printf ("        R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
    printf ("\n");
    printf ("        t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));
    }

    pcl::transformPointCloud (*cloud_filtered, *off_scene_model, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
    pcl::transformPointCloud (*cloud_temp, *off_scene_model_keypoints, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));


    for (size_t i = 0; i < off_scene_model->points.size (); ++i)
    {
    off_scene_model->points[i].r=255;
    off_scene_model->points[i].g=0;
    off_scene_model->points[i].b=0;
    }

    std::cout << "off_scene_model size is: " << off_scene_model->points.size () << std::endl;

    t = t+1;

  }
}



void cloudViewer(const std_msgs::Float32ConstPtr& number)
{
   if(a==0)
   {
   OUT_INFO("In cloud viewer");
   const std::string cloudName = "rendered";

   createcloud();
   processing();
   
   //visualizer->addPointCloud(cloud_temp, cloudName);
  visualizer->addPointCloud(cloud_scene, "scene");

   //visualizer->addPointCloudNormals<pcl::PointXYZRGBA, pcl::Normal> (cloud_filtered, cloud_normals, 100, 0.05, "scene_normals");
   //visualizer->addPointCloudNormals<pcl::PointXYZRGBA, pcl::Normal> (cloud_scene, cloud_scene_normals, 100, 0.05, "normals")

   //visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloudName);


   //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> off_scene_model_color_handler (off_scene_model, 0, 0, 255);
   //visualizer->addPointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model");

   for (size_t i = 0; i < rototranslations.size (); ++i)
  {
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr rotated_model (new pcl::PointCloud<pcl::PointXYZRGBA> ());
    pcl::transformPointCloud (*cloud_filtered, *rotated_model, rototranslations[i]);

    std::stringstream ss_cloud;
    ss_cloud << "instance" << i;

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> rotated_model_color_handler (cloud_filtered, 255, 0, 0);
    visualizer->addPointCloud (cloud_filtered, rotated_model_color_handler, ss_cloud.str ()); //should be rotated 

    
      for (size_t j = 0; j < clustered_corrs[i].size (); ++j)
      {
        std::stringstream ss_line;
        std::stringstream ss_point;
        ss_line << "correspondence_line" << i << "_" << j;
        pcl::PointXYZRGBA& model_point = cloud_filtered->at (clustered_corrs[i][j].index_query);
        pcl::PointXYZRGBA& scene_point = scene_keypoints->at (clustered_corrs[i][j].index_match);

        //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
        visualizer->addLine<pcl::PointXYZRGBA, pcl::PointXYZRGBA> (model_point, scene_point, 0, 255,0, ss_line.str ());


        cloud_identified->points.push_back(scene_point);
      }
  }

   std::cout << "Identified object has: " << cloud_identified->points.size () << "points" << std::endl;

   visualizer->initCameraParameters();
   visualizer->setBackgroundColor(0, 0, 0);
   visualizer->setShowFPS(true);
   visualizer->setCameraPosition(0, 0, 0, 0, -1, 0);
   a=a+1;
   }

   createcloud();
   processing();

   
    
   //visualizer->updatePointCloud(cloud, cloudName);

   //visualizer->updatePointCloud(scene_keypoints, "scene");
   //visualizer->updatePointCloud(rotated_model, "one");

   //visualizer->removePointCloud("normals");
  // visualizer->addPointCloudNormals<pcl::PointXYZRGBA,pcl::Normal> (cloud_filtered,cloud_normals,100,0.05,"normals");
   //visualizer->removePointCloud("scene_normals");
   //visualizer->addPointCloudNormals<pcl::PointXYZRGBA,pcl::Normal> (cloud_scene,cloud_scene_normals,100,0.05,"scene_normals");
   visualizer->spinOnce(5);

   OUT_INFO("Cloud name is: " FG_CYAN << cloudName << NO_COLOR);

}


int
main (int argc, char** argv)
{ 
  // Initialize ROS
  ros::init (argc, argv, "my_pcl_tutorial");
  ros::NodeHandle nh;

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub1 = nh.subscribe ("/kinect2/sd/image_ir_rect", 1000, cloud_cb1);
  ros::Subscriber sub2 = nh.subscribe ("/kinect2/sd/image_depth_rect", 1000, cloud_cb2);
  ros::Subscriber sub3 = nh.subscribe ("/kinect2/sd/camera_info", 1000, cloud_cb3);
  ros::Subscriber sub4 = nh.subscribe ("/kinect2/sd/camera_info", 1000, cloud_cb4);
  
  ros::Subscriber sub5 = nh.subscribe ("invoke_visualizer", 1000, cloudViewer);
  ros::spin ();
}

