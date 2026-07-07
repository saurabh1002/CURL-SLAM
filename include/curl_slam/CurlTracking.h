//
// Created by zkc on 06/06/23.
//

#ifndef SRC_CURLTRACKING_H
#define SRC_CURLTRACKING_H

#include "curl_slam/CurlLocalBA.h"
#include "curl_slam/CurlPoseGraph.h"
#include "curl_slam/CurlVoxelMapping.h"
#include "curl_slam/FileReaderBase.h"
#include "curl_slam/ThreadSafeQueue.h"
#include "curl_slam/curl_tools_light.h"
#include "curl_slam/light_structure.h"
#include "curl_slam/lock_manager.h"
#include "curl_slam/optimization_types.h"
#include "curl_slam/spatial_hashing.h"
#include "curl_slam/types.h"
#include <Eigen/Dense>
#include <bitset>
#include <condition_variable>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <limits>
#include <atomic>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <shared_mutex>
#include <nav_msgs/Path.h>
#include <omp.h>
#include <pcl/common/common_headers.h>
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <thread>
#include <tuple>
#include <unordered_set>

template <typename BasicType> class CurlTracking {
  private:
    bool is_initial;
    std::shared_ptr<CURL_TRACKING_CONFIG> curl_tracking_config_ptr;
    std::shared_ptr<AABB_CONFIG> aabb_config_ptr;
    std::shared_ptr<SH_TABLE_CONFIG<BasicType>> SH_table_config_ptr;
    std::shared_ptr<CurlVoxelMapping<BasicType>> curl_voxel_mapping_ptr;
    std::shared_ptr<CurlPoseGraph<BasicType>> curl_pose_graph_ptr;
    std::shared_ptr<DEBUG_CONFIG> debug_config_ptr;
    //    Eigen::Matrix<double, 6, 1> T_w_j_tangent;
    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_w_j;
    Eigen::Isometry3d T_j_1_j;
    Eigen::Isometry3d T_w_j_1;
    std::vector<int> frame_idx_vec;
    std::vector<Eigen::Isometry3d> odometry_vector;
    // optimizatin

    ceres::Manifold *SE3_manifold_x_y_yall;
    ceres::Manifold *SE3_manifold_z_roll_pitch;
    ceres::Manifold *SE3_manifold_z_pitch;

    ceres::LossFunctionWrapper *ground_loss_function;
    ceres::LossFunctionWrapper *seg_loss_function;
    StopCallback stop_callback;
    // ROS settings
    ros::Publisher pub_trajectory, pub_labelled_trajectory;
    nav_msgs::Path trajectory;
    ros::Publisher pub_used_cloud;
    tf2_ros::TransformBroadcaster tfb;
    geometry_msgs::TransformStamped transformStamped;
    ros::Publisher pub_bounding_box;
    int local_window_size = -1;
    std::mutex pose_graph_lock;
    std::mutex trajectory_label_lock;
    enum class LCState : int { IDLE = 0, RUNNING = 1, APPLYING = 2 };
    struct LCResult {
        uint64_t gen = 0;
        bool is_strict = false;
        Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_correction = Eigen::Matrix4d::Identity();
        std::shared_ptr<KeyframeInfo<BasicType>> current_keyframe_ptr;
        std::shared_ptr<KeyframeInfo<BasicType>> history_keyframe_ptr;
        std::shared_ptr<TrajectoryLabel> merged_label_ptr;
        std::shared_ptr<TrajectoryLabel> source_label_ptr;
        std::shared_ptr<const std::unordered_set<unsigned int>> premerge_current_neighbor_label_frame_nums_ptr;
        std::shared_ptr<const std::unordered_set<unsigned int>> premerge_history_neighbor_label_frame_nums_ptr;
        std::shared_ptr<const std::unordered_set<unsigned int>> premerge_current_frame_nums_snapshot_ptr;
        std::shared_ptr<const std::unordered_set<unsigned int>> premerge_history_frame_nums_snapshot_ptr;
        BADeferredPack<BasicType> deferred_pack;
        std::shared_ptr<OverlapRemovalPlan> overlap_plan;
        int ref_frame = -1;
    };
    struct StrictBATask {
        uint64_t gen = 0;
        std::shared_ptr<KeyframeInfo<BasicType>> current_keyframe_ptr;
        std::shared_ptr<KeyframeInfo<BasicType>> history_keyframe_ptr;
        PoseGraph3dErrorTerm *stage1_icp_constrain_ptr = nullptr;
        std::shared_ptr<TrajectoryLabel> source_label_ptr;
        std::shared_ptr<TrajectoryLabel> merged_label_ptr;
        std::shared_ptr<const std::unordered_set<unsigned int>> premerge_current_neighbor_label_frame_nums_ptr;
        std::shared_ptr<const std::unordered_set<unsigned int>> premerge_history_neighbor_label_frame_nums_ptr;
        std::shared_ptr<const std::unordered_set<unsigned int>> premerge_current_frame_nums_snapshot_ptr;
        std::shared_ptr<const std::unordered_set<unsigned int>> premerge_history_frame_nums_snapshot_ptr;
        std::shared_ptr<const LoopPoseSnapshotMap> loop_pose_snapshot_ptr;
        std::shared_ptr<OverlapRemovalPlan> overlap_plan;
        int ref_frame = -1;
    };

    std::atomic<uint64_t> lc_generation{0};
    std::atomic<uint64_t> lc_applied_non_strict{0};
    std::atomic<uint64_t> lc_applied_strict{0};
    std::atomic<int> lc_state{static_cast<int>(LCState::IDLE)};
    std::atomic<bool> strict_lc_running{false};
    std::mutex strict_ba_queue_lock;
    std::deque<StrictBATask> strict_ba_queue;
    std::mutex strict_stage3_mutex;
    std::condition_variable strict_stage3_cv;
    std::atomic<uint64_t> strict_stage3_done_gen{0};
    std::mutex lc_result_lock;
    std::deque<LCResult> pending_lc_results;
    std::atomic<uint64_t> post_strict_debug_gen{0};
    std::atomic<int> post_strict_debug_frames_left{0};
    // time evaluation
    int number_patches;
    int largest_pyramid_depth;
    // for BA
    Eigen::Isometry3d T_lastKeyframe_keyframe; // this is needed because not all frames are stored
    // cost evaluation
    std::vector<BasicType> our_costs;
    // std::vector<double> gt_costs;
    // keyframe adding flag
    bool is_add_keyframe, is_add_trajectory_segment;

    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_w_j_keyframe;
    int T_w_j_keyframe_frame_num = -1;
    int last_keyframe_idx;

    enum BOXSTATUS { INSIDE, ADD, ABANDON };

    void update_and_publish_trajectory(ros::Time current_time, const char *reason = "unspecified", uint64_t gen = 0);
    void apply_pending_lc_if_any();
    void start_strict_lc_task(const StrictBATask &task);
    void strict_lc_worker();
    void trigger_strict_lc_stage1_and_queue_ba(const std::shared_ptr<KeyframeInfo<BasicType>> &current_keyframe_ptr,
                                               const std::shared_ptr<KeyframeInfo<BasicType>> &history_keyframe_ptr,
                                               PoseGraph3dErrorTerm *stage1_icp_constrain_ptr);

    int valid_history_frame_idx;

    int frame_idx_counter;

    unsigned int trajectory_associated_counter;

    int current_label_frame_num;
    std::shared_ptr<TrajectoryLabel> curr_trajectory_label_ptr;

    visualization_msgs::Marker trajectory_segment_marker;
    float trajectory_segment_marker_line_scale[3] = {0.1, 0, 0};

  public:
    CurlTracking() = delete;
    CurlTracking(ros::NodeHandle *nh, std::shared_ptr<CURL_TRACKING_CONFIG> _curl_tracking_config_ptr,
                 std::shared_ptr<AABB_CONFIG> _aabb_config_ptr, std::shared_ptr<SH_TABLE_CONFIG<BasicType>> _SH_table_config_ptr,
                 std::shared_ptr<DEBUG_CONFIG> _debug_config_ptr,
                 std::shared_ptr<CurlVoxelMapping<BasicType>> _curl_voxel_mapping_ptr,
                 std::shared_ptr<CurlPoseGraph<BasicType>> _curl_pose_graph_ptr);
    void run();
    BOXSTATUS new_landmark_add_judge(std::shared_ptr<PatchInfo<BasicType>> patch_info_ptr, const double &IoU,
                                     const std::vector<double> &query_lower_bound,
                                     const std::vector<double> &query_upper_bound);

    void publish_trajectory(const ros::Publisher &publisher, const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_w_j,
                            nav_msgs::Path &path, ros::Time current_time);
    void publish_labelled_trajectory(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_w_j,
                                     const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_w_j_1,
                                     ros::Time current_time);
    void publish_lidar_tf(ros::Time current_time);

    void publish_point_cloud(
        ros::Time current_time,
        const std::vector<std::tuple<Eigen::MatrixX<BasicType>, std::shared_ptr<PatchInfo<BasicType>>, double>>
            &succeed_associations);
    void publish_query_map_bounding_box(
        ros::Time current_time,
        const std::vector<std::pair<std::vector<double>, std::vector<double>>> &query_bounding_box_vec,
        const std::vector<std::pair<std::vector<double>, std::vector<double>>> &map_bounding_box_vec);

    void update_sph_coeff_thread(
        const std::vector<std::tuple<Eigen::MatrixX<BasicType>, std::shared_ptr<PatchInfo<BasicType>>, double>>
            &succeed_associations,
        const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &T_w_lidar);
    void update_sph_coeff_thread(
        const std::vector<std::tuple<Eigen::MatrixX<BasicType>, std::shared_ptr<PatchInfo<BasicType>>, double>>
            &succeed_associations,
        const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &T_w_lidar,
        const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe_ptr, bool _is_add_trajectory_segment);
    bool add_landmark_thread(bool is_new_keyframe, bool _is_add_trajectory_segment,
                             const std::vector<Eigen::MatrixX<BasicType>> &point_cloud_lidar_vec,
                             const std::vector<bool> &is_ground_cloud_vec, const double time,
                             const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &T_w_lidar,
                             const Eigen::Isometry3d &_T_lastKeyframe_keyframe, int _frame_idx,
                             const pcl::PointCloud<PointT>::Ptr &seg_cloud_ptr,
                             const pcl::PointCloud<PointT>::Ptr &ground_cloud_ptr,
                             std::shared_ptr<TrajectoryLabel> &_history_trajectory_label_ptr);

    void log();
};

#endif // SRC_CURLTRACKING_H
