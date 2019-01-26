#include <pcl/filters/voxel_grid.h>
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
#include <time.h>
#include <std_msgs/Float32.h>
#include <geometry_msgs/Point.h>
#include <pcl/filters/passthrough.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/console/parse.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/model_outlier_removal.h>
#include "ros/ros.h"
#include <geometry_msgs/Quaternion.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>


cv::Mat color_img, depth_img;
cv::Mat lookupX, lookupY;
cv::Mat cameraMatrixDepth = cv::Mat::zeros(3, 3, CV_64F);
cv::Mat cameraMatrixColor = cv::Mat::zeros(3, 3, CV_64F);

sensor_msgs::CameraInfo cameraInfoColor;
sensor_msgs::CameraInfo cameraInfoDepth;

pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::visualization::PCLVisualizer::Ptr visualizer(new pcl::visualization::PCLVisualizer("Cloud Viewer"));

const std::string cloudName = "rendered";

   Eigen::Vector4f centroid; 
 pcl::PointXYZRGBA centre;

pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered_out (new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered2(new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2 (new pcl::PointCloud<pcl::Normal>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_cylinder (new pcl::PointCloud<pcl::PointXYZRGBA> ());
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZRGBA> ());
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_final (new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::ModelCoefficients::Ptr coefficients_plane (new pcl::ModelCoefficients), coefficients_cylinder (new pcl::ModelCoefficients);
pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices), inliers_cylinder (new pcl::PointIndices);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered_fin (new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_centroid (new pcl::PointCloud<pcl::PointXYZRGBA>);
//pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
pcl::ExtractIndices<pcl::Normal> extract_normals;



int a=0;
int g=0;
int cloud_initialise=0;


class SubscribeAndPublish
{
public:
    SubscribeAndPublish()

    {
       pub_coeff = nh.advertise<geometry_msgs::Quaternion>("n_coefficients", 100);	
       pub_centre = nh.advertise<geometry_msgs::Quaternion>("centre_c", 100);
       sub1 = nh.subscribe ("/kinect2/sd/image_ir_rect", 1000, &SubscribeAndPublish::cloud_cb1,this);
  	   sub2 = nh.subscribe ("/kinect2/sd/image_depth_rect", 1000, &SubscribeAndPublish::cloud_cb2,this);
       sub3 = nh.subscribe ("/kinect2/sd/camera_info", 1000, &SubscribeAndPublish::cloud_cb3,this);
       sub4 = nh.subscribe ("/kinect2/sd/camera_info", 1000, &SubscribeAndPublish::cloud_cb4,this);
       sub5 = nh.subscribe ("invoke_visualizer", 1000, &SubscribeAndPublish::cloudViewer,this);
       ros::spin ();
    }



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


		void cylinder()


		{
  
 			createcloud();


          pcl::PointIndices::Ptr inliers_2 (new pcl::PointIndices);

    for (int i = 0; i < (*cloud_centroid).size(); i++)
    {
        
        if (g==0) 
        {
            inliers_2->indices.push_back(i);
        }

    }


    extract.setInputCloud(cloud_centroid);
    extract.setIndices(inliers_2);
    extract.setNegative(true);
    extract.filter(*cloud_centroid);

  			pcl::PassThrough<pcl::PointXYZRGBA> pass;
  			pass.setInputCloud (cloud);
  			pass.setFilterFieldName ("z");
  			pass.setFilterLimits (0, 1.2);
  			pass.filter (*cloud_filtered_out);
  			std::cerr << "PointCloud has: " << cloud->points.size () << " data points." << std::endl;
  			std::cerr << "PointCloud after Pass through filtering has: " << cloud_filtered_out->points.size () << " data points." << std::endl;

        pcl::io::savePCDFileASCII ("scene_pcd.pcd", *cloud_filtered_out);
        std::cerr << "Saved " << cloud->points.size () << " data points to test_pcd.pcd." << std::endl;

  			pcl::VoxelGrid<pcl::PointXYZRGBA> sor;
  			sor.setInputCloud (cloud_filtered_out);
 			  sor.setLeafSize (0.009, 0.009, 0.009);
  			sor.filter (*cloud_filtered);
  			std::cerr << "Voxel Filtered PointCloud has: " << cloud_filtered->points.size () << " data points." << std::endl;

 
  //pcl::SACSegmentationFromNormals<pcl::PointXYZRGBA, pcl::Normal> seg;
 
  			pcl::SACSegmentation<pcl::PointXYZRGBA> seg;
  
  			seg.setOptimizeCoefficients (true);
  			seg.setModelType (pcl::SACMODEL_PLANE);
  			seg.setMethodType (pcl::SAC_RANSAC);
  //seg.setNormalDistanceWeight (0.1);
  			seg.setMaxIterations (1000);
  			seg.setDistanceThreshold (0.01);
  //seg.setRadiusLimits (0, 0.1);
 			seg.setInputCloud (cloud_filtered);
 // seg.setInputNormals (cloud_normals);
 
  // Obtain the cylinder inliers and coefficients
  			seg.segment (*inliers_cylinder, *coefficients_cylinder);

  
  			extract.setInputCloud (cloud_filtered);
  			extract.setIndices (inliers_cylinder);
  			extract.setNegative (false);
  			extract.filter (*cloud_plane);

  			pcl::ModelOutlierRemoval<pcl::PointXYZRGBA> filter;
  			filter.setModelCoefficients (*coefficients_cylinder);
  			filter.setThreshold (0.1);
  			filter.setModelType (pcl::SACMODEL_PLANE);
  			filter.setInputCloud (cloud_plane);
  			filter.filter (*cloud_filtered_fin);

        pcl::io::savePLYFileBinary("hexagon.ply" , *cloud_filtered_fin);

  			 pcl::compute3DCentroid(*cloud_filtered_fin,centroid); 

  			//cout <<centroid <<endl;
  			centre.x=centroid[0];
  			centre.y=centroid[1];
  			centre.z=centroid[2];
  			centre.r=255;
  			centre.g=0;
  			centre.b=0;
  			cloud_centroid->points.push_back(centre);
  			cout<<centre<<endl;




  			//std::cerr << "Plane Filtered PointCloud has: " << cloud_filtered_fin->points.size () << " data points." << std::endl;
  			
  			if (cloud_filtered_fin->points.size ()>400)
  			{
            std::cerr << "Model coefficients: " << coefficients_cylinder->values[0] << " " 
                                      << coefficients_cylinder->values[1] << " "
                                      << coefficients_cylinder->values[2] << " " 
                                      << coefficients_cylinder->values[3] << std::endl;



            geometry_msgs::Quaternion msg;
            msg.x = centroid[0];
            msg.y = centroid[1];
            msg.z = centroid[2];
            msg.w = 0;
            pub_centre.publish(msg);
            
            geometry_msgs::Quaternion msg2;
            msg2.x = coefficients_cylinder->values[0];
            msg2.y = coefficients_cylinder->values[1];
            msg2.z = coefficients_cylinder->values[2];
            msg2.w = coefficients_cylinder->values[3];
            pub_coeff.publish(msg2);
        }	


/*
            geometry_msgs::Quaternion msg2;
            msg2.x = 1;
            msg2.y = 2;
            msg2.z = 3;
            msg2.w = 4;
            pub_coeff.publish(msg2);

            }
 
			int t1 =  (rand() % cloud_filtered_fin->points.size ());
			int t2 =  (rand() % cloud_filtered_fin->points.size ());
 			int t3 =  (rand() % cloud_filtered_fin->points.size ());


			float p1_x = cloud_filtered_fin->points[t1].x;
			float p1_y = cloud_filtered_fin->points[t1].y;
			float p1_z = cloud_filtered_fin->points[t1].z;

			float p2_x = cloud_filtered_fin->points[t2].x;
			float p2_y = cloud_filtered_fin->points[t2].y;
			float p2_z = cloud_filtered_fin->points[t2].z;

			float p3_x = cloud_filtered_fin->points[t3].x;
			float p3_y = cloud_filtered_fin->points[t3].y;
			float p3_z = cloud_filtered_fin->points[t3].z;

			std_msgs::Float32 msg;
			msg.data = 32;
			pub.publish(msg);
			
*/



/*
OUT_INFO(p1_x);
OUT_INFO(p1_y);
OUT_INFO(p1_z);

OUT_INFO(p2_x);
OUT_INFO(p2_y);
OUT_INFO(p2_z);

OUT_INFO(p3_x);
OUT_INFO(p3_y);
OUT_INFO(p3_z);
*/




/*
  for (size_t i = 0; i < inliers_cylinder->indices.size (); ++i)
  {
    std::cerr << inliers_cylinder->indices[i] << "    " << cloud_filtered_fin->points[inliers_cylinder->indices[i]].x << " "
                                               << cloud_filtered_fin->points[inliers_cylinder->indices[i]].y << " "
                                               << cloud_filtered_fin->points[inliers_cylinder->indices[i]].z << std::endl;
    //cloud_filtered->points[inliers_cylinder->indices[i]].r = 255; 
    //cloud_filtered->points[inliers_cylinder->indices[i]].g = 0; 
    //cloud_filtered->points[inliers_cylinder->indices[i]].b = 0; 
   }
*/


		}



		void cloudViewer(const std_msgs::Float32ConstPtr& number)
		{

			

   			if(a==0)
  			 {
   				OUT_INFO("In cloud viewer");
   				const std::string cloudName = "rendered";
   
  				cylinder();
   				OUT_INFO("I'm Back!!");

          //visualizer->addPointCloud(cloud_filtered, cloudName);
          pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> c1 (cloud_filtered_fin, 0, 0, 0);
   				visualizer->addPointCloud(cloud_filtered_fin, c1, cloudName);
          visualizer->addPointCloud(cloud_centroid, "centroid");
   //visualizer->addPointCloudNormals<pcl::PointXYZRGBA, pcl::Normal> (cloud_filtered, cloud_normals, 100, 0.05, "normals");
   				visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, cloudName);
          visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, "centroid");
   				visualizer->initCameraParameters();
  				visualizer->setBackgroundColor(255, 255, 255);
   				visualizer->setSize(color_img.cols, color_img.rows);
   				visualizer->setShowFPS(true);
   				visualizer->setCameraPosition(0, 0, 0, 0, -1, 0);
   //visualizer->addCoordinateSystem (1.0);
   //visualizer->addLine<pcl::PointXYZRGB> (cloud_filtered_fin->points[0], cloud_filtered_fin->points[cloud_filtered_fin->size() - 1], "line");
  				a=a+1;
   			}

   				cylinder();
   				OUT_INFO("Updating Cloudddddddddddd");
          pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> c1 (cloud_filtered_fin, 0, 0, 0);
   				visualizer->updatePointCloud(cloud_filtered_fin, c1, cloudName);
          visualizer->updatePointCloud(cloud_centroid, "centroid");
          visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, cloudName);
          visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, "centroid");
          //visualizer->updatePointCloud(cloud_filtered, cloudName);
   //visualizer->removePointCloud("normals");
   //visualizer->addPointCloudNormals<pcl::PointXYZRGBA, pcl::Normal> (cloud_filtered, cloud_normals, 100, 0.05, "normals");
   				visualizer->spinOnce(5);

		}



private:
  ros::NodeHandle nh; 
  ros::Publisher pub_centre;
  ros::Publisher pub_coeff;
  ros::Subscriber sub1;
  ros::Subscriber sub2;
  ros::Subscriber sub3;
  ros::Subscriber sub4;
  ros::Subscriber sub5;
};



int main(int argc, char **argv)
{
  //Initiate ROS
  ros::init(argc, argv, "subscribe_and_publish");

  //Create an object of class SubscribeAndPublish that will take care of everything
  SubscribeAndPublish SAPObject;

  ros::spin();

  return 0;
}


/*
int
main (int argc, char** argv)
{ 
  // Initialize ROS
  ros::init (argc, argv, "my_pcl_tutorial");
  ros::NodeHandle nh;

  ros::Publisher p1 = nh.advertise<geometry_msgs::Point>("point_one", 1000);
  // ros::Rate loop_rate(10);

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub1 = nh.subscribe ("/kinect2/sd/image_ir_rect", 1000, cloud_cb1);
  ros::Subscriber sub2 = nh.subscribe ("/kinect2/sd/image_depth_rect", 1000, cloud_cb2);
  ros::Subscriber sub3 = nh.subscribe ("/kinect2/sd/camera_info", 1000, cloud_cb3);
  ros::Subscriber sub4 = nh.subscribe ("/kinect2/sd/camera_info", 1000, cloud_cb4);
  ros::Subscriber sub5 = nh.subscribe ("invoke_visualizer", 1000, cloudViewer);
  ros::spin ();
 }
*/

/*
void not_required_but_imp()
{
pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
pcl::SACSegmentation<pcl::PointXYZRGBA> seg;

  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.01);

  seg.setInputCloud (cloud_filtered);
  seg.segment (*inliers, *coefficients);
    if (inliers->indices.size () == 0)
  {
    PCL_ERROR ("Could not estimate a planar model for the given dataset.");
    
  }

  std::cerr << "Model inliers: " << inliers->indices.size () << std::endl;
  for (size_t i = 0; i < inliers->indices.size (); ++i)
  {
    std::cerr << inliers->indices[i] << "    " << cloud->points[inliers->indices[i]].x << " "
                                               << cloud->points[inliers->indices[i]].y << " "
                                               << cloud->points[inliers->indices[i]].z << std::endl;
    cloud_filtered->points[inliers->indices[i]].r = 255; 
    cloud_filtered->points[inliers->indices[i]].g = 0; 
    cloud_filtered->points[inliers->indices[i]].b = 0; 
   }
//to find x,y,z values

for(pcl::PointCloud<pcl::PointXYZRGBA>::iterator it=cloud->begin(); it!=cloud->end();it++)
{
    std::cerr <<it->x<<","<<it->y<<","<<it->z<< std::endl;
}



}
*/
