//
// Created by zkc on 15/03/24.
//

#ifndef CURLPOSEGRAPH_H
#define CURLPOSEGRAPH_H
#include "curl_slam/CurlLocalBA.h"
#include "curl_slam/CurlVoxelMapping.h"
#include "curl_slam/nanoflann.hpp"
#include "curl_slam/optimization_types.h"
#include "curl_slam/spatial_hashing.h"
#include "curl_slam/types.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <shared_mutex>
#include <memory>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <unordered_map>
#include <unordered_set>

template <typename BasicType> struct PendingPoseGraphConstrainEntry {
    std::weak_ptr<KeyframeInfo<BasicType>> history_keyframe_ptr;
    std::weak_ptr<KeyframeInfo<BasicType>> current_keyframe_ptr;
    Pose3d T_his_curr;
    bool is_icp_constrain = false;
};

template <typename BasicType> struct PendingLoopConstrainRemovalEntry {
    std::weak_ptr<KeyframeInfo<BasicType>> keyframe_ptr;
    PoseGraph3dErrorTerm *constrain_ptr = nullptr;
};

template <typename BasicType> struct BADeferredPack {
    std::vector<std::weak_ptr<PatchInfo<BasicType>>> pending_ba_patches;
    std::vector<std::pair<std::weak_ptr<KeyframeInfo<BasicType>>, std::weak_ptr<KeyframeInfo<BasicType>>>>
        pending_associated_keyframes;
    std::vector<PendingPoseGraphConstrainEntry<BasicType>> pending_ba_pose_graph_constrains;
    std::vector<PendingLoopConstrainRemovalEntry<BasicType>> pending_ba_remove_loop_constrains;
    PoseGraph3dErrorTerm *pending_ba_remove_icp_constrain_ptr = nullptr;
    std::vector<std::weak_ptr<KeyframeInfo<BasicType>>> ba_used_keyframes;
};

struct OverlapRemovalPlan {
    std::unordered_set<PatchId> remove_patch_ids;
    std::unordered_map<PatchId, PatchId> replacement_map;
    std::unordered_map<PatchId, int> owner_frame_nums;
    std::atomic<bool> pending_removals_active{false};
    int ref_frame_num = -1;
};

using LoopPoseSnapshotMap = std::unordered_map<int, std::array<double, 16>>;

template <typename BasicType> class CurlPoseGraph {
  public:
    CurlPoseGraph(ros::NodeHandle *nh, std::shared_ptr<CURL_LOOP_CLOSURE_CONFIG> _curl_loop_closure_config_ptr,
                  std::shared_ptr<CURL_TRACKING_CONFIG> _curl_tracking_config_ptr,
                  std::shared_ptr<CURL_VOXEL_MAPPING_CONFIG> _curl_voxel_mapping_config_ptr,
                  std::shared_ptr<SH_TABLE_CONFIG<BasicType>> _SH_table_config_ptr,
                  std::shared_ptr<DEBUG_CONFIG> _debug_config_ptr,
                  std::shared_ptr<CurlVoxelMapping<BasicType>> _curl_voxel_mapping_ptr);
    void kdTree_keyframe_poses_emplace_back(const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe_ptr);
    void kdTree_update_keyframe_poses();
    void add_odometry_constrain();
    bool add_odometry_constrain_for_keyframe(const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe_ptr);
    void set_skip_odometry_after_keyframe(const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe_ptr);
    void add_loop_closure_constrain(const std::shared_ptr<KeyframeInfo<BasicType>> &_history_keyframe_ptr,
                                    const std::shared_ptr<KeyframeInfo<BasicType>> &_current_keyframe_ptr,
                                    const Pose3d &T_his_curr, const bool is_icp_constrain);
    /**
     * @input 1: time of the current frame
     * @input 2: position of the current keyframe
     * @input 3: raw point cloud of the neighbour keyframe
     * @return: true if loop closure is detected
     */
    bool detect_loop_for_keyframe(const std::shared_ptr<KeyframeInfo<BasicType>> &query_keyframe_ptr);
    std::shared_ptr<KeyframeInfo<BasicType>> pose_graph_processing_for_pair(
        const std::shared_ptr<KeyframeInfo<BasicType>> &current_keyframe_ptr,
        const std::shared_ptr<KeyframeInfo<BasicType>> &history_keyframe_ptr, bool is_scan_to_map);
    std::shared_ptr<KeyframeInfo<BasicType>> get_history_keyframe_ptr() const { return history_keyframe_ptr; }
    void remove_overlapped_patches_after_loop();
    void remove_overlapped_patches_after_loop_history();
    void apply_pending_ba_updates();
    void apply_pending_ba_updates(BADeferredPack<BasicType> &pack);
    void apply_pending_associated_patches_updates();
    void apply_pending_associated_patches_updates_up_to_frame(int max_current_frame_num);
    void apply_pending_associated_patches_updates(BADeferredPack<BasicType> &pack);
    bool is_scan_to_map_loop_closure_last() const { return is_scan_to_map_loop_closure; }
    void solve_pose_graph();
    void solve_local_BA_pose_graph();
    void set_loop_closure_pair(const std::shared_ptr<KeyframeInfo<BasicType>> &current_keyframe_ptr_in,
                               const std::shared_ptr<KeyframeInfo<BasicType>> &history_keyframe_ptr_in,
                               bool is_scan_to_map);
    void set_loop_closure_pair_with_premerge(
        const std::shared_ptr<KeyframeInfo<BasicType>> &current_keyframe_ptr_in,
        const std::shared_ptr<KeyframeInfo<BasicType>> &history_keyframe_ptr_in, bool is_scan_to_map,
        const std::shared_ptr<TrajectoryLabel> &pre_merge_current_label,
        const std::shared_ptr<TrajectoryLabel> &pre_merge_history_label,
        const std::shared_ptr<const std::unordered_set<unsigned int>>
            &pre_merge_current_label_frame_nums_snapshot_ptr_in =
            std::shared_ptr<const std::unordered_set<unsigned int>>(),
        const std::shared_ptr<const std::unordered_set<unsigned int>>
            &pre_merge_history_label_frame_nums_snapshot_ptr_in =
            std::shared_ptr<const std::unordered_set<unsigned int>>(),
        const std::shared_ptr<const std::unordered_set<unsigned int>>
            &pre_merge_current_frame_nums_snapshot_ptr_in =
            std::shared_ptr<const std::unordered_set<unsigned int>>(),
        const std::shared_ptr<const std::unordered_set<unsigned int>>
            &pre_merge_history_frame_nums_snapshot_ptr_in =
            std::shared_ptr<const std::unordered_set<unsigned int>>());
    void solve_pose_graph_initial_only();
    void run_local_BA_after_pose_update(
        const std::shared_ptr<KeyframeInfo<BasicType>> &ba_current_keyframe_ptr,
        const std::shared_ptr<KeyframeInfo<BasicType>> &ba_history_keyframe_ptr,
        bool defer_pose_graph_updates = false,
        const std::shared_ptr<TrajectoryLabel> &history_label_filter_ptr = std::shared_ptr<TrajectoryLabel>(),
        const std::shared_ptr<const std::unordered_set<unsigned int>> &history_label_frame_num_snapshot_ptr =
            std::shared_ptr<const std::unordered_set<unsigned int>>(),
        int max_frame_num_for_ba = -1,
        BADeferredPack<BasicType> *deferred_pack_out = nullptr,
        const std::shared_ptr<const LoopPoseSnapshotMap> &loop_pose_snapshot_ptr =
            std::shared_ptr<const LoopPoseSnapshotMap>(),
        const std::shared_ptr<const std::unordered_set<unsigned int>> &history_frame_num_snapshot_ptr =
            std::shared_ptr<const std::unordered_set<unsigned int>>());
    void apply_pending_ba_pose_graph_updates();
    void apply_pending_ba_pose_graph_updates(BADeferredPack<BasicType> &pack);
    std::shared_ptr<KeyframeInfo<BasicType>> pose_graph_processing();
    std::shared_ptr<OverlapRemovalPlan> build_overlap_removal_plan_after_loop();
    void apply_overlap_removal_plan(const OverlapRemovalPlan &plan);
    void add_pending_overlap_removals(const std::unordered_set<PatchId> &ids);
    void remove_pending_overlap_removals(const std::unordered_set<PatchId> &ids);

    std::shared_ptr<CURL_LOOP_CLOSURE_CONFIG> curl_loop_closure_config_ptr;
    std::shared_ptr<CURL_TRACKING_CONFIG> curl_tracking_config_ptr;
    std::shared_ptr<CURL_VOXEL_MAPPING_CONFIG> curl_voxel_mapping_config_ptr;
    std::shared_ptr<DEBUG_CONFIG> debug_config_ptr;
    std::shared_ptr<SH_TABLE_CONFIG<BasicType>> SH_table_config_ptr;

    std::vector<LoopClosureConstrain> loop_closure_constrains;

  private:
    void local_BA_data_association_point_clouds(
        ceres::Problem &problem,
        ceres::ParameterBlockOrdering &ordering,
        std::map<std::shared_ptr<KeyframeInfo<BasicType>>,
                 std::vector<std::pair<std::shared_ptr<KeyframeInfo<BasicType>>, int>>,
                 key_frame_info_comparator<BasicType>> &associated_keyframes,
        std::unordered_set<std::shared_ptr<PatchInfo<BasicType>>> &all_associated_history_patches,
        std::set<std::shared_ptr<KeyframeInfo<BasicType>>, key_frame_info_comparator<BasicType>> &history_keyframes_set,
        std::set<std::shared_ptr<KeyframeInfo<BasicType>>, key_frame_info_comparator<BasicType>>
            &opt_history_keyframes_set,
        const std::shared_ptr<TrajectoryLabel> &history_label_filter_ptr,
        const std::shared_ptr<const std::unordered_set<unsigned int>> &history_label_frame_num_snapshot_ptr,
        int max_frame_num_for_ba, BADeferredPack<BasicType> *deferred_pack_out = nullptr);

    void remove_overlapped_patches();

    bool is_patch_pending_overlap_removal(PatchId patch_id) const;

    double get_associations_from_raw_points(
        const int current_frame_idx, const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe_ptr,
        const pcl::PointCloud<PointT> &seg_cloud_transformed_curr,
        const pcl::PointCloud<PointT> &ground_cloud_transformed_curr,
        const std::shared_ptr<TrajectoryLabel> &history_label_filter_ptr,
        const std::shared_ptr<const std::unordered_set<unsigned int>> &history_label_frame_num_snapshot_ptr,
        std::vector<std::tuple<Eigen::MatrixX<BasicType>, std::shared_ptr<PatchInfo<BasicType>>, double>>
            &pose_succeed_associations);

    void solve_pose_graph_without_kdTree_update();

    pcl::PointCloud<PointT>::Ptr get_keyframe_point_clouds(const int shift_idx, const Eigen::Matrix4f &T_lidar_w);

    pcl::PointCloud<PointT>::Ptr get_neighbour_keyframe_point_clouds(const int shift_idx,
                                                                     const Eigen::Matrix4f &T_lidar_w);

    std::shared_ptr<CurlVoxelMapping<BasicType>> curl_voxel_mapping_ptr;
    using my_kd_tree_t =
        curl::nanoflann::KDTreeSingleIndexDynamicAdaptor<nanoflann::L2_Simple_Adaptor<double, curl::PointCloud<double>>,
                                                         curl::PointCloud<double>, 3 /* dim */>;
    std::unique_ptr<my_kd_tree_t> kdTree_pose_ptr;
    curl::PointCloud<double> keyframe_poses;
    std::vector<std::shared_ptr<KeyframeInfo<BasicType>>> kdTree_keyframes;
    std::size_t kdTree_old_idx;
    ceres::Problem pose_graph_problem;
    ceres::Manifold *quat_manifold;
    PoseGraph3dErrorTerm *icp_loop_closure_cost_function = nullptr;
    bool is_first;
    std::shared_ptr<KeyframeInfo<BasicType>> current_keyframe_ptr;
    std::shared_ptr<KeyframeInfo<BasicType>> history_keyframe_ptr;

    bool is_scan_to_map_loop_closure;
    std::atomic<int> skip_odometry_after_frame_num{-1};

    int associated_times = 0;

    std::vector<std::weak_ptr<PatchInfo<BasicType>>> pending_ba_patches;
    std::vector<std::pair<std::weak_ptr<KeyframeInfo<BasicType>>, std::weak_ptr<KeyframeInfo<BasicType>>>>
        pending_associated_keyframes;
    std::vector<PendingPoseGraphConstrainEntry<BasicType>> pending_ba_pose_graph_constrains;
    std::vector<PendingLoopConstrainRemovalEntry<BasicType>> pending_ba_remove_loop_constrains;
    PoseGraph3dErrorTerm *pending_ba_remove_icp_constrain_ptr = nullptr;
    std::shared_ptr<TrajectoryLabel> pre_merge_current_label_ptr;
    std::shared_ptr<TrajectoryLabel> pre_merge_history_label_ptr;
    std::shared_ptr<const std::unordered_set<unsigned int>> pre_merge_current_label_frame_nums_snapshot_ptr;
    std::shared_ptr<const std::unordered_set<unsigned int>> pre_merge_history_label_frame_nums_snapshot_ptr;
    std::shared_ptr<const std::unordered_set<unsigned int>> pre_merge_current_frame_nums_snapshot_ptr;
    std::shared_ptr<const std::unordered_set<unsigned int>> pre_merge_history_frame_nums_snapshot_ptr;
    mutable std::shared_mutex pending_overlap_remove_mutex;
    std::unordered_map<PatchId, int> pending_overlap_remove_refcount;
    std::shared_ptr<const std::unordered_set<PatchId>> pending_overlap_snapshot_ptr;
    std::atomic<uint64_t> pending_overlap_checked{0};
    std::atomic<uint64_t> pending_overlap_skipped{0};
};

#endif // CURLPOSEGRAPH_H
