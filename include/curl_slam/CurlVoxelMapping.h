#ifndef CURLVOXEL_MAPPING_H
#define CURLVOXEL_MAPPING_H
#include "curl_slam/PatchProcession.h"
#include "curl_slam/Scancontext.h"
#include "curl_slam/curl_tools_light.h"
#include "curl_slam/light_structure.h"
#include "curl_slam/load_config.h"
#include "curl_slam/optimization_types.h"
#include "curl_slam/spatial_hashing.h"
#include "curl_slam/types.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <memory>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <mutex>
#include <omp.h>
#include <pcl/common/common_headers.h>
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <thread>
#include <tuple>
#include <visualization_msgs/Marker.h>

template <typename BasicType> class CurlVoxelMapping {
  public:
    CurlVoxelMapping() = default;
    // constructor for odometry
    CurlVoxelMapping(ros::NodeHandle *nh, std::shared_ptr<CURL_VOXEL_MAPPING_CONFIG> _curl_voxel_mapping_config_ptr,
                     std::shared_ptr<CURL_TRACKING_CONFIG> _curl_tracking_config_ptr,
                     std::shared_ptr<SH_TABLE_CONFIG<BasicType>> _SH_table_config_ptr,
                     std::shared_ptr<DIRECT_METHOD_CONFIG> _direct_method_config_ptr,
                     std::shared_ptr<AABB_CONFIG> _aabb_config_ptr,
                     std::shared_ptr<CURL_LOOP_CLOSURE_CONFIG> _curl_loop_closure_config_ptr,
                     std::shared_ptr<DEBUG_CONFIG> _debug_config_ptr);

    std::shared_ptr<CURL_VOXEL_MAPPING_CONFIG> curl_voxel_mapping_config_ptr;
    std::queue<std::tuple<double, pcl::PointCloud<PointT>::Ptr, pcl::PointCloud<PointT>::Ptr>> pcl_ptr_pair_queue;
    omp_lock_t writelock;

    bool rt_fix_voxel_initialization(bool is_new_keyframe, bool is_add_trajectory_segment,
                                     const std::vector<Eigen::MatrixX<BasicType>> &patches_lidar_vec,
                                     const std::vector<bool> &is_ground_cloud_vec, const double time,
                                     const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &T_w_lidar_curr,
                                     const Eigen::Isometry3d &_T_lastKeyframe_keyframe, bool is_active, int frame_idx,
                                     const pcl::PointCloud<PointT>::Ptr &seg_cloud_ptr,
                                     const pcl::PointCloud<PointT>::Ptr &ground_cloud_ptr,
                                     std::shared_ptr<TrajectoryLabel> &_history_trajectory_label_ptr);

    // for local BA
    std::mutex local_BA_lock;
    std::condition_variable local_BA_cv;

    std::mutex preprocessing_queue_lock;

    std::shared_ptr<SpatialHashing<BasicType>> spatial_hashing_ptr;

    bool is_tracking_wait_BA;
    std::mutex BA_wait_lock;
    std::condition_variable tracking_wait_BA_cv;
    bool is_loop_closure_detected, is_strict_loop_closure_detected;
    int frame_counter, robust_initialization_counter;

    visualization_msgs::Marker set_default_marker(const std::string &frame_id, const std::string &ns, const int &id,
                                                  const std::array<float, 3> &color,
                                                  int32_t type = visualization_msgs::Marker::POINTS,
                                                  const float scale[3] = (float[3]){0.05, 0.05, 0.05},
                                                  const float position[3] = (float[3]){0, 0, 0},
                                                  const std::string &text = "");
    void rt_publish_landmark_patch_recons(const std::shared_ptr<PatchInfo<BasicType>> &patch_info_ptr);
    void refresh_dense_reconstruction_markers();

    void makeAndSaveScancontextAndKeys(pcl::PointCloud<PointT> &scan_down);
    std::pair<int, float> detectLoopClosureID(void);
    std::pair<int, float> detectLoopClosureID(const int query_idx);

    std::pair<std::vector<double>, std::vector<double>>
    preprocessing(const double time, const pcl::PointCloud<PointT>::Ptr &seg_cloud_ptr,
                  const pcl::PointCloud<PointT>::Ptr &ground_cloud_ptr, const Eigen::Matrix4f _T_w_j,
                  std::vector<std::vector<Eigen::MatrixX<BasicType>>> &_point_cloud_vec,
                  PointCloudInfo<BasicType> &_point_cloud_info);
    std::pair<std::vector<double>, std::vector<double>>
    preprocessing(const double time, const pcl::PointCloud<PointT>::Ptr &seg_cloud_ptr,
                  const pcl::PointCloud<PointT>::Ptr &ground_cloud_ptr, const Eigen::Matrix4d &_T_j_1_j,
                  const Eigen::Matrix4f _T_w_j, std::vector<std::vector<Eigen::MatrixX<BasicType>>> &_point_cloud_vec,
                  PointCloudInfo<BasicType> &_point_cloud_info);
    bool curl_registration_method(
        const double cloud_time, const int _last_keyframe_idx, const pcl::PointCloud<PointT>::Ptr &seg_cloud_ptr,
        const pcl::PointCloud<PointT>::Ptr &ground_cloud_ptr, const int _current_label_frame_num,
        const std::shared_ptr<TrajectoryLabel> &_curr_trajectory_label_ptr, const bool is_use_all_associated_patches,
        const double highest_IoU_new_landmark_thres, const int minimum_observation_num, const bool is_voxel_grid_filter,
        const double leaf_size, const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_w_j_keyframe,
        const Eigen::Matrix4d &_T_j_1_j, Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_w_j,
        std::vector<std::vector<Eigen::MatrixX<BasicType>>> &_point_cloud_vec,
        PointCloudInfo<BasicType> &_point_cloud_info,
        std::vector<std::tuple<Eigen::MatrixX<BasicType>, std::shared_ptr<PatchInfo<BasicType>>, double>>
            &succeed_associations,
        std::pair<std::vector<double>, std::vector<double>> &bounds_w,
        std::unordered_set<std::array<int, 2>, Voxel2DHashFuncPrimeArray> &new_box_map, int &_number_patches,
        bool &_is_add_keyframe, bool &_is_add_trajectory_segment, std::vector<BasicType> &_our_costs,
        std::vector<std::pair<std::vector<double>, std::vector<double>>> &query_bounding_box_vec,
        std::vector<std::pair<std::vector<double>, std::vector<double>>> &map_bounding_box_vec);

    bool curl_registration_method_minimum(
        const pcl::PointCloud<PointT>::Ptr &seg_cloud_ptr, const pcl::PointCloud<PointT>::Ptr &ground_cloud_ptr,
        const bool is_voxel_grid_filter, const double leaf_size,
        const std::unordered_set<PatchId> &associated_patches,
        Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_w_j);

    // template <class Archive> void serialize(Archive &ar, const unsigned int version) { ar &spatial_hashing_ptr; }

  private:
    // friend class boost::serialization::access;

    void remove_points_inside_bounding_box(
        const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe_info_ptr,
        const Eigen::MatrixX<BasicType> &merged_seg_point_cloud_world,
        const Eigen::MatrixX<BasicType> &merged_ground_point_cloud_world,
        std::vector<std::pair<std::vector<int>, Eigen::MatrixX<BasicType>>> &seg_points_vec_world,
        std::vector<std::pair<std::vector<int>, Eigen::MatrixX<BasicType>>> &ground_points_vec_world);
    void re_clustering(
        const int frame_idx, const Eigen::MatrixX<BasicType> &merged_seg_point_cloud_world,
        const Eigen::MatrixX<BasicType> &merged_ground_point_cloud_world,
        std::unordered_map<std::vector<int>, Eigen::MatrixX<BasicType>, VoxelHashFuncPrime> &seg_points_vec_world,
        std::unordered_map<std::vector<int>, Eigen::MatrixX<BasicType>, VoxelHashFuncPrime> &ground_points_vec_world);

    bool get_local_coordinate_with_eig(bool is_ground, const Eigen::Isometry3d &T_w_lidar,
                                       const Eigen::MatrixX<BasicType> &patch_cloud_w,
                                       const std::vector<double> &lower_bound_w,
                                       const std::vector<double> &upper_bound_w, Eigen::Isometry3d &T_obj_lidar,
                                       Eigen::Matrix4<BasicType> &cov_4, PROJECTION_AXIS &projection_axis);
    void rt_callback_point_cloud_segments_seg_only(const sensor_msgs::PointCloud2::ConstPtr &segment_msg);
    void rt_callback_point_cloud_segments(const sensor_msgs::PointCloud2::ConstPtr &segment_msg,
                                          const sensor_msgs::PointCloud2::ConstPtr &ground_msg);
    std::shared_ptr<SH_TABLE_CONFIG<BasicType>> SH_table_config_ptr;
    std::shared_ptr<DIRECT_METHOD_CONFIG> direct_method_config_ptr;
    std::shared_ptr<AABB_CONFIG> aabb_config_ptr;
    std::shared_ptr<CURL_LOOP_CLOSURE_CONFIG> curl_loop_closure_config_ptr;
    std::shared_ptr<DEBUG_CONFIG> debug_config_ptr;

    ros::Publisher pub_landmark_patch;
    message_filters::Subscriber<sensor_msgs::PointCloud2> rt_segment_pt_sub;
    message_filters::Subscriber<sensor_msgs::PointCloud2> rt_ground_pt_sub;
    message_filters::Subscriber<sensor_msgs::PointCloud2> rt_non_ground_pt_sub;
    ros::Subscriber segment_pt_sub;
    typedef message_filters::sync_policies::ExactTime<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2> MySyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync_ptr;
    std::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> non_ground_sync_ptr;

    std::vector<std::array<float, 3>> colors = {
        //            {1.0f, 0.0f, 0.0f}, // Red
        //            {0.0f, 1.0f, 0.0f}, // Green
        {0.0f, 0.0f, 1.0f} // Blue
                           //            {1.0f, 1.0f, 0.0f}, // Yellow
                           //            {0.0f, 1.0f, 1.0f}  // Cyan
    };
    bool is_init;
    SCManager scManager;
    std::shared_ptr<CURL_TRACKING_CONFIG> curl_tracking_config_ptr;
    // StopCallback stop_callback;
};

#endif // CURLVOXEL_MAPPING_H
