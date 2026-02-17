//
// Created by zkc on 15/03/24.
//

#include "curl_slam/CurlPoseGraph.h"
#include "curl_slam/Timer.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <ros/ros.h>

template <typename BasicType>
CurlPoseGraph<BasicType>::CurlPoseGraph(ros::NodeHandle *nh,
                                        std::shared_ptr<CURL_LOOP_CLOSURE_CONFIG> _curl_loop_closure_config_ptr,
                                        std::shared_ptr<CURL_TRACKING_CONFIG> _curl_tracking_config_ptr,
                                        std::shared_ptr<CURL_VOXEL_MAPPING_CONFIG> _curl_voxel_mapping_config_ptr,
                                        std::shared_ptr<SH_TABLE_CONFIG<BasicType>> _SH_table_config_ptr,
                                        std::shared_ptr<DEBUG_CONFIG> _debug_config_ptr,
                                        std::shared_ptr<CurlVoxelMapping<BasicType>> _curl_voxel_mapping_ptr)
    : curl_loop_closure_config_ptr(_curl_loop_closure_config_ptr), curl_tracking_config_ptr(_curl_tracking_config_ptr),
      curl_voxel_mapping_config_ptr(_curl_voxel_mapping_config_ptr), SH_table_config_ptr(_SH_table_config_ptr),
      debug_config_ptr(_debug_config_ptr), curl_voxel_mapping_ptr(_curl_voxel_mapping_ptr) {
    is_first = true;
    quat_manifold = new ceres::EigenQuaternionManifold();
    kdTree_pose_ptr = std::make_unique<my_kd_tree_t>(3, keyframe_poses, 10);
    kdTree_old_idx = keyframe_poses.size();
}

template <typename BasicType>
void CurlPoseGraph<BasicType>::kdTree_keyframe_poses_emplace_back(
    const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe_ptr) {
    if (!keyframe_ptr || !keyframe_ptr->is_backend_keyframe) {
        return;
    }
    keyframe_poses.emplace_back(keyframe_ptr->get_T_w_lidar()(Eigen::seq(0, 2), 3));
    kdTree_keyframes.push_back(keyframe_ptr);
    kdTree_pose_ptr->addPoints(kdTree_old_idx, keyframe_poses.size() - 1);
    kdTree_old_idx = keyframe_poses.size();
}

template <typename BasicType> void CurlPoseGraph<BasicType>::kdTree_update_keyframe_poses() {
    keyframe_poses.clear();
    kdTree_keyframes.clear();
    const auto keyframes = curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframes_snapshot();
    keyframe_poses.reserve(keyframes.size());
    kdTree_keyframes.reserve(keyframes.size());
    for (const auto &keyframe : keyframes) {
        if (!keyframe || !keyframe->is_backend_keyframe) {
            continue;
        }
        keyframe_poses.emplace_back(keyframe->get_T_w_lidar()(Eigen::seq(0, 2), 3));
        kdTree_keyframes.push_back(keyframe);
    }
    kdTree_pose_ptr = std::make_unique<my_kd_tree_t>(3, keyframe_poses, 10);
    kdTree_old_idx = keyframe_poses.size();
}

template <typename BasicType> void CurlPoseGraph<BasicType>::add_odometry_constrain() {
    const auto keyframes = curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframes_snapshot();
    if (keyframes.size() < 2) {
        return;
    }
    const auto &keyframe_ptr = keyframes.back();
    const auto &last_keyframe_ptr = keyframes[keyframes.size() - 2];
    // ceres::CostFunction *cost_function = PoseGraph3dErrorTerm::Create(((*keyframe_iter)->get_graph_obs_j_1_j()),
    // false);
    PoseGraph3dErrorTerm *pose_graph3d_error_term =
        new PoseGraph3dErrorTerm((keyframe_ptr->get_graph_obs_j_1_j()), false);
    ceres::CostFunction *cost_function =
        new ceres::AutoDiffCostFunction<PoseGraph3dErrorTerm, 6, 3, 4, 3, 4>(pose_graph3d_error_term);
    pose_graph_problem.AddResidualBlock(cost_function, nullptr, last_keyframe_ptr->get_graph_pose_w_lidar_Pdata(),
                                        last_keyframe_ptr->get_graph_pose_w_lidar_Qdata(),
                                        keyframe_ptr->get_graph_pose_w_lidar_Pdata(),
                                        keyframe_ptr->get_graph_pose_w_lidar_Qdata());
    pose_graph_problem.SetManifold(last_keyframe_ptr->get_graph_pose_w_lidar_Qdata(), quat_manifold);
    pose_graph_problem.SetManifold(keyframe_ptr->get_graph_pose_w_lidar_Qdata(), quat_manifold);
    keyframe_ptr->loop_closure_cost_function = pose_graph3d_error_term;
    if (is_first) {
        pose_graph_problem.SetParameterBlockConstant(last_keyframe_ptr->get_graph_pose_w_lidar_Pdata());
        pose_graph_problem.SetParameterBlockConstant(last_keyframe_ptr->get_graph_pose_w_lidar_Qdata());
        is_first = false;
    }
}

template <typename BasicType>
bool CurlPoseGraph<BasicType>::add_odometry_constrain_for_keyframe(
    const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe_ptr) {
    if (!keyframe_ptr) {
        return false;
    }
    const auto keyframes = curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframes_snapshot();
    auto iter = std::find(keyframes.begin(), keyframes.end(), keyframe_ptr);
    if (iter == keyframes.end()) {
        return false;
    }
    if (iter == keyframes.begin()) {
        // First keyframe has no previous frame for odometry constraint, but should still be backend.
        return true;
    }
    const auto &last_keyframe_ptr = *(iter - 1);
    const auto &curr_keyframe_ptr = *iter;
    const int skip_after = skip_odometry_after_frame_num.load();
    if (skip_after >= 0 && last_keyframe_ptr && last_keyframe_ptr->frame_num == skip_after) {
        skip_odometry_after_frame_num.store(-1);
        return true;
    }
    PoseGraph3dErrorTerm *pose_graph3d_error_term =
        new PoseGraph3dErrorTerm((curr_keyframe_ptr->get_graph_obs_j_1_j()), false);
    ceres::CostFunction *cost_function =
        new ceres::AutoDiffCostFunction<PoseGraph3dErrorTerm, 6, 3, 4, 3, 4>(pose_graph3d_error_term);
    pose_graph_problem.AddResidualBlock(cost_function, nullptr, last_keyframe_ptr->get_graph_pose_w_lidar_Pdata(),
                                        last_keyframe_ptr->get_graph_pose_w_lidar_Qdata(),
                                        curr_keyframe_ptr->get_graph_pose_w_lidar_Pdata(),
                                        curr_keyframe_ptr->get_graph_pose_w_lidar_Qdata());
    pose_graph_problem.SetManifold(last_keyframe_ptr->get_graph_pose_w_lidar_Qdata(), quat_manifold);
    pose_graph_problem.SetManifold(curr_keyframe_ptr->get_graph_pose_w_lidar_Qdata(), quat_manifold);
    curr_keyframe_ptr->loop_closure_cost_function = pose_graph3d_error_term;
    if (is_first) {
        pose_graph_problem.SetParameterBlockConstant(last_keyframe_ptr->get_graph_pose_w_lidar_Pdata());
        pose_graph_problem.SetParameterBlockConstant(last_keyframe_ptr->get_graph_pose_w_lidar_Qdata());
        is_first = false;
    }
    return true;
}

template <typename BasicType>
void CurlPoseGraph<BasicType>::set_skip_odometry_after_keyframe(
    const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe_ptr) {
    if (!keyframe_ptr || keyframe_ptr->frame_num < 0) {
        return;
    }
    skip_odometry_after_frame_num.store(keyframe_ptr->frame_num);
}

template <typename BasicType>
void CurlPoseGraph<BasicType>::add_loop_closure_constrain(
    const std::shared_ptr<KeyframeInfo<BasicType>> &_history_keyframe_ptr,
    const std::shared_ptr<KeyframeInfo<BasicType>> &_current_keyframe_ptr, const Pose3d &T_his_curr,
    const bool is_icp_constrain) {

    // ceres::CostFunction *cost_function = PoseGraph3dErrorTerm::Create(T_his_curr, is_icp_constrain);

    PoseGraph3dErrorTerm *pose_graph3d_error_term = new PoseGraph3dErrorTerm(T_his_curr, is_icp_constrain);
    ceres::CostFunction *cost_function =
        new ceres::AutoDiffCostFunction<PoseGraph3dErrorTerm, 6, 3, 4, 3, 4>(pose_graph3d_error_term);

    if (is_icp_constrain) {
        icp_loop_closure_cost_function = pose_graph3d_error_term;
    }
    pose_graph_problem.AddResidualBlock(cost_function, nullptr, _history_keyframe_ptr->get_graph_pose_w_lidar_Pdata(),
                                        _history_keyframe_ptr->get_graph_pose_w_lidar_Qdata(),
                                        _current_keyframe_ptr->get_graph_pose_w_lidar_Pdata(),
                                        _current_keyframe_ptr->get_graph_pose_w_lidar_Qdata());
    pose_graph_problem.SetManifold(_history_keyframe_ptr->get_graph_pose_w_lidar_Qdata(), quat_manifold);
    pose_graph_problem.SetManifold(_current_keyframe_ptr->get_graph_pose_w_lidar_Qdata(), quat_manifold);
}

template <typename BasicType>
void CurlPoseGraph<BasicType>::local_BA_data_association_point_clouds(
    ceres::Problem &problem, ceres::ParameterBlockOrdering &ordering,
    std::map<std::shared_ptr<KeyframeInfo<BasicType>>,
             std::vector<std::pair<std::shared_ptr<KeyframeInfo<BasicType>>, int>>,
             key_frame_info_comparator<BasicType>> &associated_keyframes,
    std::unordered_set<std::shared_ptr<PatchInfo<BasicType>>> &all_associated_history_patches,
    std::set<std::shared_ptr<KeyframeInfo<BasicType>>, key_frame_info_comparator<BasicType>> &history_keyframes_set,
    std::set<std::shared_ptr<KeyframeInfo<BasicType>>, key_frame_info_comparator<BasicType>>
        &opt_history_keyframes_set,
    const std::shared_ptr<TrajectoryLabel> &history_label_filter_ptr,
    const std::shared_ptr<const std::unordered_set<unsigned int>> &history_label_frame_num_snapshot_ptr,
    int max_frame_num_for_ba, BADeferredPack<BasicType> *deferred_pack_out) {
    std::shared_ptr<KeyframeInfo<BasicType>> oldest_keyframe_ptr = nullptr;
    const auto keyframes = curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframes_snapshot_reverse();
    if (keyframes.empty()) {
        return;
    }
    int current_frame_idx = -1;
    bool is_intersected = false;
    ceres::Manifold *_SE3_manifold = SE3Manifold::Create();
    ceres::LossFunctionWrapper *_loss_function = new ceres::LossFunctionWrapper(
        new ceres::HuberLoss(curl_tracking_config_ptr->seg_kernel_threshold), ceres::DO_NOT_TAKE_OWNERSHIP);
    double max_associate_patch_num = 0;
    bool is_initial = true;
    int invalid_counter = 0;
    for (const auto &keyframe_ptr : keyframes) {
        if (max_frame_num_for_ba >= 0 && keyframe_ptr && keyframe_ptr->frame_num > max_frame_num_for_ba) {
            continue;
        }
        if (!keyframe_ptr) {
            continue;
        }
        if (!keyframe_ptr->is_backend_keyframe) {
            continue;
        }
        if (current_frame_idx < 0) {
            current_frame_idx = keyframe_ptr->frame_idx;
        }
        // Double-check pattern: set flag first, then check pointer
        keyframe_ptr->is_being_used_by_ba = true;
        if (deferred_pack_out) {
            deferred_pack_out->ba_used_keyframes.push_back(keyframe_ptr);
        }
        if (!keyframe_ptr->seg_cloud_ptr || !keyframe_ptr->ground_cloud_ptr) {
            keyframe_ptr->is_being_used_by_ba = false;
            break;
        }
        pcl::PointCloud<PointT> seg_cloud_transformed_curr, ground_cloud_transformed_curr;
        Timer transform_timer;
        transform_timer.start();
        pcl::transformPointCloud(*keyframe_ptr->seg_cloud_ptr, seg_cloud_transformed_curr,
                                 Eigen::Matrix4f(keyframe_ptr->get_T_w_lidar().template cast<float>()));
        pcl::transformPointCloud(*keyframe_ptr->ground_cloud_ptr, ground_cloud_transformed_curr,
                                 Eigen::Matrix4f(keyframe_ptr->get_T_w_lidar().template cast<float>()));

        // get data associations from the history map
        std::vector<std::tuple<Eigen::MatrixX<BasicType>, std::shared_ptr<PatchInfo<BasicType>>, double>>
            pose_succeed_associations;
        Timer assoc_timer;
        assoc_timer.start();
        double association_num =
            get_associations_from_raw_points(current_frame_idx, keyframe_ptr, seg_cloud_transformed_curr,
                                             ground_cloud_transformed_curr, history_label_filter_ptr,
                                             history_label_frame_num_snapshot_ptr,
                                             pose_succeed_associations);
        // std::cout << "association_percentage: " << association_percentage << std::endl;
        if (is_initial || association_num > max_associate_patch_num) {
            max_associate_patch_num = association_num;
            is_initial = false;
            std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << std::endl;
            std::cout << max_associate_patch_num << std::endl;
            std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << std::endl;
            invalid_counter = 0;
        } else if (association_num < 0.9 * max_associate_patch_num) {
            // ++invalid_counter;
            // if (invalid_counter > 3) {
            // }
            // continue;
            break;
        }
        // TODO: the following should be remove out of this loop. history_keyframes_set needs to be preprocessed
        Timer load_residuals_timer;
        load_residuals_timer.start();
        CurlLocalBA<BasicType>::load_residuals_point_clouds(
            problem, _SE3_manifold, _loss_function, &ordering, keyframe_ptr, pose_succeed_associations,
            curl_tracking_config_ptr, curl_voxel_mapping_config_ptr, SH_table_config_ptr, associated_keyframes,
            all_associated_history_patches, history_keyframes_set);
    }
    if (current_frame_idx < 0) {
        return;
    }

    // selected the oldest keyframe here
    for (auto &keyframe_ptr : history_keyframes_set) {
        double asso_rate =
            double(keyframe_ptr->get_BA_asso_local_patches_size() / double(keyframe_ptr->get_local_patches_size()));
        // keyframe_ptr->clear_BA_asso_local_patches();
        if (asso_rate < curl_loop_closure_config_ptr->association_rate_for_opt_history_keyframe) {
            CurlLocalBA<BasicType>::set_last_keyframe_constant(problem, keyframe_ptr);
        } else {
            opt_history_keyframes_set.insert(keyframe_ptr);
        }
    }
}

template <typename BasicType>
double CurlPoseGraph<BasicType>::get_associations_from_raw_points(
    const int current_frame_idx, const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe_ptr,
    const pcl::PointCloud<PointT> &seg_cloud_transformed_curr,
    const pcl::PointCloud<PointT> &ground_cloud_transformed_curr,
    const std::shared_ptr<TrajectoryLabel> &history_label_filter_ptr,
    const std::shared_ptr<const std::unordered_set<unsigned int>> &history_label_frame_num_snapshot_ptr,
    std::vector<std::tuple<Eigen::MatrixX<BasicType>, std::shared_ptr<PatchInfo<BasicType>>, double>>
        &pose_succeed_associations) {
    Timer assoc_total_timer;
    assoc_total_timer.start();
    double seg_leaf_array[3] = {curl_voxel_mapping_ptr->curl_voxel_mapping_config_ptr->cut_threshold,
                                curl_voxel_mapping_ptr->curl_voxel_mapping_config_ptr->cut_threshold,
                                curl_voxel_mapping_ptr->curl_voxel_mapping_config_ptr->cut_threshold};
    std::pair<std::vector<double>, std::vector<double>> scan_bounds_w =
        curl::get_bounding_box_w<PointT>(seg_cloud_transformed_curr, ground_cloud_transformed_curr);
    std::vector<std::vector<Eigen::MatrixX<BasicType>>> seg_cloud_vec;
    Timer divide_timer;
    divide_timer.start();
    seg_cloud_vec = curl::divide_patches<PointT, BasicType>(
        seg_cloud_transformed_curr, *keyframe_ptr->seg_cloud_ptr, seg_leaf_array,
        curl_voxel_mapping_ptr->curl_voxel_mapping_config_ptr->patch_minimun_pts_num_after_filter, scan_bounds_w.first,
        scan_bounds_w.second, curl_loop_closure_config_ptr->region_width_elements);
    std::vector<std::vector<bool>> seg_is_ground_cloud_vec(seg_cloud_vec.size());
    for (int i = 0; i < seg_cloud_vec.size(); ++i) {
        seg_is_ground_cloud_vec[i].resize(seg_cloud_vec[i].size(), false);
    }
    double ground_leaf_array[3] = {curl_voxel_mapping_ptr->curl_voxel_mapping_config_ptr->cut_threshold,
                                   curl_voxel_mapping_ptr->curl_voxel_mapping_config_ptr->cut_threshold,
                                   curl_voxel_mapping_ptr->curl_voxel_mapping_config_ptr->cut_threshold};
    std::vector<std::vector<Eigen::MatrixX<BasicType>>> ground_cloud_vec;
    ground_cloud_vec = curl::divide_patches<PointT, BasicType>(
        ground_cloud_transformed_curr, *keyframe_ptr->ground_cloud_ptr, ground_leaf_array,
        curl_voxel_mapping_ptr->curl_voxel_mapping_config_ptr->patch_minimun_pts_num_after_filter, scan_bounds_w.first,
        scan_bounds_w.second, curl_loop_closure_config_ptr->region_width_elements);
    std::vector<std::vector<bool>> ground_is_ground_cloud_vec(ground_cloud_vec.size());
    for (int i = 0; i < ground_cloud_vec.size(); ++i) {
        ground_is_ground_cloud_vec[i].resize(ground_cloud_vec[i].size(), true);
    }

    std::vector<std::vector<Eigen::MatrixX<BasicType>>> point_cloud_vec;
    PointCloudInfo<BasicType> point_cloud_info;
    point_cloud_vec.reserve(seg_cloud_vec.size() + ground_cloud_vec.size());
    point_cloud_vec.insert(point_cloud_vec.end(), seg_cloud_vec.begin(), seg_cloud_vec.end());
    point_cloud_info.is_ground_cloud_vec.reserve(seg_cloud_vec.size() + ground_cloud_vec.size());
    point_cloud_info.is_ground_cloud_vec.insert(point_cloud_info.is_ground_cloud_vec.end(),
                                                seg_is_ground_cloud_vec.begin(), seg_is_ground_cloud_vec.end());
    // get ground cloud information
    for (int i = 0; i < ground_cloud_vec.size(); ++i) {
        point_cloud_vec[i].insert(point_cloud_vec[i].end(), ground_cloud_vec[i].begin(), ground_cloud_vec[i].end());
        point_cloud_info.is_ground_cloud_vec[i].insert(point_cloud_info.is_ground_cloud_vec[i].end(),
                                                       ground_is_ground_cloud_vec[i].begin(),
                                                       ground_is_ground_cloud_vec[i].end());
    }
    const double divide_ms = divide_timer.elapsedMilliseconds();
    point_cloud_info.intersected_score_pairs_vec.resize(point_cloud_vec.size());
    point_cloud_info.bounding_box_pair_vec.resize(point_cloud_vec.size());
    int intersect_counter = 0;
    int totoal_raw_patches = 0;
    uint64_t pending_overlap_checked_local = 0;
    uint64_t pending_overlap_skipped_local = 0;
    const Eigen::Matrix4d T_w_lidar = keyframe_ptr->get_T_w_lidar();
    const Eigen::Matrix<BasicType, 3, 3> R_w_lidar =
        T_w_lidar(Eigen::seq(0, 2), Eigen::seq(0, 2)).template cast<BasicType>();
    const Eigen::Matrix<BasicType, 3, 1> t_w_lidar = T_w_lidar(Eigen::seq(0, 2), 3).template cast<BasicType>();
    for (int i = 0; i < point_cloud_vec.size(); ++i) {
        point_cloud_info.intersected_score_pairs_vec[i].resize(point_cloud_vec[i].size());
        point_cloud_info.bounding_box_pair_vec[i].resize(point_cloud_vec[i].size());
    }

    Timer aabb_precompute_timer;
    aabb_precompute_timer.start();
    #pragma omp parallel for num_threads(curl_voxel_mapping_config_ptr->data_association_thread_num) \
        default(none) shared(point_cloud_vec, point_cloud_info, R_w_lidar, t_w_lidar) schedule(dynamic)
    for (int i = 0; i < point_cloud_vec.size(); ++i) {
        for (int j = 0; j < point_cloud_vec[i].size(); ++j) {
            const Eigen::MatrixX<BasicType> &patch_cloud = point_cloud_vec[i][j];
            Eigen::Matrix<BasicType, 3, 1> min_pt, max_pt;
            min_pt.setConstant(std::numeric_limits<BasicType>::max());
            max_pt.setConstant(std::numeric_limits<BasicType>::lowest());
            for (int c = 0; c < patch_cloud.cols(); ++c) {
                const Eigen::Matrix<BasicType, 3, 1> p = R_w_lidar * patch_cloud.col(c) + t_w_lidar;
                min_pt = min_pt.cwiseMin(p);
                max_pt = max_pt.cwiseMax(p);
            }
            std::vector<double> lower_bound_w(3), upper_bound_w(3);
            lower_bound_w[0] = static_cast<double>(min_pt(0));
            lower_bound_w[1] = static_cast<double>(min_pt(1));
            lower_bound_w[2] = static_cast<double>(min_pt(2));
            upper_bound_w[0] = static_cast<double>(max_pt(0));
            upper_bound_w[1] = static_cast<double>(max_pt(1));
            upper_bound_w[2] = static_cast<double>(max_pt(2));
            point_cloud_info.bounding_box_pair_vec[i][j] = std::make_pair(lower_bound_w, upper_bound_w);
        }
    }
    const double aabb_precompute_ms = aabb_precompute_timer.elapsedMilliseconds();

    std::vector<std::pair<std::shared_ptr<PatchInfo<BasicType>>, double>> intersected_pairs;
    intersected_pairs.reserve(64);
    std::vector<std::pair<std::shared_ptr<PatchInfo<BasicType>>, double>> local_intersected_pairs;
    std::vector<double> lower_bound_w(3);
    std::vector<double> upper_bound_w(3);
    Timer query_timer;
    query_timer.start();
    #pragma omp parallel num_threads(curl_voxel_mapping_config_ptr->data_association_thread_num) \
        default(none) shared(point_cloud_vec, point_cloud_info, keyframe_ptr, R_w_lidar, t_w_lidar, \
                            pending_overlap_snapshot_ptr, history_label_filter_ptr, history_label_frame_num_snapshot_ptr) \
            reduction(+:intersect_counter, totoal_raw_patches, pending_overlap_checked_local, pending_overlap_skipped_local) \
            firstprivate(local_intersected_pairs, lower_bound_w, upper_bound_w)
    {
        local_intersected_pairs.clear();
        local_intersected_pairs.reserve(64);
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < point_cloud_vec.size(); ++i) {        // NOTE: each region
            local_intersected_pairs.clear();
            for (int j = 0; j < point_cloud_vec[i].size(); ++j) { // NOTE: elements of region
                ++totoal_raw_patches;
                const Eigen::MatrixX<BasicType> &patch_cloud = point_cloud_vec[i][j];
                const auto &bounding_box = point_cloud_info.bounding_box_pair_vec[i][j];
                const std::vector<double> &lower_bound_w = bounding_box.first;
                const std::vector<double> &upper_bound_w = bounding_box.second;
                curl_voxel_mapping_ptr->spatial_hashing_ptr->query_data_association_overlap_excludeCurrKeyframe(
                    keyframe_ptr, lower_bound_w, upper_bound_w, local_intersected_pairs);
                auto pending_snapshot = std::atomic_load(&pending_overlap_snapshot_ptr);
                for (const auto &pair : local_intersected_pairs) {
                    if (!pair.first) {
                        continue;
                    }
                    ++pending_overlap_checked_local;
                    if (pending_snapshot && pending_snapshot->find(pair.first->key) != pending_snapshot->end()) {
                        ++pending_overlap_skipped_local;
                        continue;
                    }
                    std::shared_ptr<KeyframeInfo<BasicType>> patch_keyframe_ptr;
                    std::shared_ptr<TrajectoryLabel> patch_label_ptr;
                    {
                        std::lock_guard<std::mutex> patch_lock(pair.first->patch_update_lock);
                        patch_keyframe_ptr = pair.first->keyframe_ptr;
                        if (patch_keyframe_ptr) {
                            patch_label_ptr = patch_keyframe_ptr->trajectory_label_ptr;
                        }
                    }
                    bool is_label_accepted = false;
                    if (history_label_frame_num_snapshot_ptr) {
                        if (patch_keyframe_ptr && patch_keyframe_ptr->frame_num >= 0) {
                            is_label_accepted =
                                history_label_frame_num_snapshot_ptr->find(
                                    static_cast<unsigned int>(patch_keyframe_ptr->frame_num)) !=
                                history_label_frame_num_snapshot_ptr->end();
                        }
                    } else if (!history_label_filter_ptr) {
                        is_label_accepted = true;
                    } else if (patch_label_ptr) {
                        is_label_accepted = TrajectoryLabel::is_connected(history_label_filter_ptr, patch_label_ptr);
                    }
                    if ((point_cloud_info.is_ground_cloud_vec[i][j] == pair.first->is_ground) && is_label_accepted) {
                        point_cloud_info.intersected_score_pairs_vec[i][j].emplace_back(pair.first, pair.second);
                        ++intersect_counter;
                    }
                }
            }
        }
    }
    pending_overlap_checked.store(pending_overlap_checked_local);
    pending_overlap_skipped.store(pending_overlap_skipped_local);
    const double query_ms = query_timer.elapsedMilliseconds();

    Timer match_timer;
    match_timer.start();
    std::unordered_map<const PatchInfo<BasicType> *, MapDataAsso> local_data_asso;
    local_data_asso.reserve(point_cloud_vec.size() * 4);
    std::vector<std::vector<ScanDataAsso<BasicType>>> initial_scan_data_asso_vec(point_cloud_vec.size());
    for (int i = 0; i < point_cloud_vec.size(); ++i) {
        initial_scan_data_asso_vec[i].resize(point_cloud_vec[i].size(), std::array<int, 2>{-1, -1});
        for (int j = 0; j < point_cloud_vec[i].size(); ++j) {
            if (!point_cloud_info.intersected_score_pairs_vec[i][j].empty()) {
                initial_scan_data_asso_vec[i][j].point_idx = {i, j};
                for (auto &pair : point_cloud_info.intersected_score_pairs_vec[i][j]) {
                    initial_scan_data_asso_vec[i][j].patch_info_ptr_vec.push_back(pair.first);
                    auto &asso = local_data_asso[pair.first.get()];
                    asso.points_idx_vec.push_back(std::array<int, 2>{i, j});
                    asso.IoU_vec.push_back(pair.second);
                    break; // because we need the largest IoU is enough
                }
            }
        }
    }

    std::unordered_map<const PatchInfo<BasicType> *, size_t> patch_index;
    patch_index.reserve(point_cloud_vec.size() * 8);
    std::vector<std::shared_ptr<PatchInfo<BasicType>>> associated_patches;
    associated_patches.reserve(point_cloud_vec.size() * 8);
    for (int i = 0; i < initial_scan_data_asso_vec.size(); ++i) {
        for (int j = 0; j < initial_scan_data_asso_vec[i].size(); ++j) {
            if (!initial_scan_data_asso_vec[i][j].patch_info_ptr_vec.empty()) {
                for (const auto &patch_ptr : initial_scan_data_asso_vec[i][j].patch_info_ptr_vec) {
                    auto inserted = patch_index.emplace(patch_ptr.get(), associated_patches.size());
                    if (inserted.second) {
                        associated_patches.emplace_back(patch_ptr);
                    }
                }
            }
        }
    }
    std::vector<std::array<int, 2>> best_point_idx_vec(associated_patches.size(), std::array<int, 2>{-1, -1});
    std::vector<double> best_iou_vec(associated_patches.size(), 0.0);
    for (int p = 0; p < static_cast<int>(associated_patches.size()); ++p) {
        auto it = local_data_asso.find(associated_patches[p].get());
        if (it == local_data_asso.end() || it->second.IoU_vec.empty()) {
            continue;
        }
        const int best_idx = it->second.best_index();
        if (best_idx < 0) {
            continue;
        }
        best_point_idx_vec[p] = it->second.points_idx_vec[best_idx];
        best_iou_vec[p] = it->second.IoU_vec[best_idx];
    }

    std::vector<std::vector<std::tuple<std::array<int, 2>, double, std::shared_ptr<PatchInfo<BasicType>>>>> ground_sets(
        point_cloud_vec.size());
    std::vector<std::vector<std::tuple<std::array<int, 2>, double, std::shared_ptr<PatchInfo<BasicType>>>>> seg_sets(
        point_cloud_vec.size());

    for (int i = 0; i < initial_scan_data_asso_vec.size(); ++i) {
        for (int j = 0; j < initial_scan_data_asso_vec[i].size(); ++j) {
            if (!initial_scan_data_asso_vec[i][j].patch_info_ptr_vec.empty()) {
                std::shared_ptr<PatchInfo<BasicType>> highest_score_patch_info_ptr;
                double highest_score = -1.0;
                const auto point_idx = initial_scan_data_asso_vec[i][j].point_idx;
                for (const auto &patch_ptr : initial_scan_data_asso_vec[i][j].patch_info_ptr_vec) {
                    auto it = local_data_asso.find(patch_ptr.get());
                    if (it == local_data_asso.end()) {
                        continue;
                    }
                    double score = -1.0;
                    for (std::size_t k = 0; k < it->second.points_idx_vec.size(); ++k) {
                        if (it->second.points_idx_vec[k] == point_idx) {
                            score = it->second.IoU_vec[k];
                            break;
                        }
                    }
                    if (score > highest_score) {
                        highest_score = score;
                        highest_score_patch_info_ptr = patch_ptr;
                    }
                }
                if (!highest_score_patch_info_ptr) {
                    continue;
                }
                auto it = patch_index.find(highest_score_patch_info_ptr.get());
                if (it == patch_index.end()) {
                    continue;
                }
                const size_t patch_idx = it->second;
                // if both are the best selection for each other then data-association successed
                if (initial_scan_data_asso_vec[i][j].point_idx == best_point_idx_vec[patch_idx]) {
                    // update data-association information for update sph coeff and visualization
                    if (point_cloud_info.is_ground_cloud_vec[i][j]) {
                        ground_sets[i].emplace_back(std::array<int, 2>{i, j}, best_iou_vec[patch_idx],
                                                    highest_score_patch_info_ptr);
                    } else {
                        seg_sets[i].emplace_back(std::array<int, 2>{i, j}, best_iou_vec[patch_idx],
                                                 highest_score_patch_info_ptr);
                    }
                }
            }
        }
    }
    const double match_ms = match_timer.elapsedMilliseconds();
    // sort seg_sets and ground_sets
    Timer sort_timer;
    sort_timer.start();
    for (auto &set : seg_sets) {
        std::sort(set.begin(), set.end(),
                  [](std::tuple<std::array<int, 2>, double, std::shared_ptr<PatchInfo<BasicType>>> &a,
                     std::tuple<std::array<int, 2>, double, std::shared_ptr<PatchInfo<BasicType>>> &b) {
                      return std::get<1>(a) > std::get<1>(b);
                  });
    }
    for (auto &set : ground_sets) {
        std::sort(set.begin(), set.end(),
                  [](std::tuple<std::array<int, 2>, double, std::shared_ptr<PatchInfo<BasicType>>> &a,
                     std::tuple<std::array<int, 2>, double, std::shared_ptr<PatchInfo<BasicType>>> &b) {
                      return std::get<1>(a) > std::get<1>(b);
                  });
    }
    const double sort_ms = sort_timer.elapsedMilliseconds();

    int true_ground_num = 0;
    int true_seg_num = 0;
    Timer collect_timer;
    collect_timer.start();
    for (const auto &region : ground_sets) {
        int region_counter = 0;
        for (const auto &ground_set : region) {

            pose_succeed_associations.emplace_back(
                point_cloud_vec[std::get<0>(ground_set)[0]][std::get<0>(ground_set)[1]], std::get<2>(ground_set),
                std::get<1>(ground_set));
            ++true_ground_num;
            ++region_counter;
        }
    }

    for (const auto &region : seg_sets) {
        int region_counter = 0;
        for (const auto seg_set : region) {
            pose_succeed_associations.emplace_back(point_cloud_vec[std::get<0>(seg_set)[0]][std::get<0>(seg_set)[1]],
                                                   std::get<2>(seg_set), std::get<1>(seg_set));
            ++true_seg_num;
            ++region_counter;
        }
    }
    const double collect_ms = collect_timer.elapsedMilliseconds();

    return (true_seg_num + true_ground_num);
}

template <typename BasicType> void CurlPoseGraph<BasicType>::solve_pose_graph_without_kdTree_update() {
    // ceres::LossFunction *loss_function = new ceres::HuberLoss(10.0);
    {
        std::unique_lock<std::shared_mutex> ul_graph_pose(KeyframeInfo<BasicType>::T_w_lidar_lock);
        // read lock for the keyframe set
        std::shared_lock<std::shared_mutex> sl_keyframe_set(
            curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframe_set_lock);
        // to do the optimization
        ceres::Solver::Options options;
        options.max_num_iterations = 200;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &pose_graph_problem, &summary);
        std::cout << summary.FullReport() << '\n';
    }
    {
        // Update keyframe poses (and patch bounds) before BA association, same as 060b07f behavior.
        curl_voxel_mapping_ptr->spatial_hashing_ptr->update_keyframe_pose_by_poseGraph();
    }
}

template <typename BasicType>
bool CurlPoseGraph<BasicType>::detect_loop_for_keyframe(
    const std::shared_ptr<KeyframeInfo<BasicType>> &query_keyframe_ptr) {
    if (!query_keyframe_ptr) {
        return false;
    }
    std::vector<std::pair<size_t, double>> indices_dists;
    nanoflann::RadiusResultSet<double, size_t> resultSet(curl_loop_closure_config_ptr->search_radius_squared,
                                                         indices_dists);
    curl::nanoflann::SearchParams params;

    Eigen::Vector3d query_pt = query_keyframe_ptr->get_T_w_lidar()(Eigen::seq(0, 2), 3);
    const size_t nMatches = kdTree_pose_ptr->findNeighbors(resultSet, query_pt.data(), params);
    (void)nMatches;
    if (indices_dists.empty()) {
        return false;
    }
    std::sort(
        indices_dists.begin(), indices_dists.end(),
        [](const std::pair<size_t, double> &a, const std::pair<size_t, double> &b) { return a.second < b.second; });
    std::set<int> valid_keyframe_indices;
    current_keyframe_ptr = query_keyframe_ptr;
    int curr_frame_idx = current_keyframe_ptr->frame_idx;
    bool simple_loop_closure_detector = false;
    is_scan_to_map_loop_closure = false;
    std::pair<int, float> sc_matches;
    for (auto &index_dist : indices_dists) {
        const auto history_idx = static_cast<std::size_t>(index_dist.first);
        if (history_idx >= kdTree_keyframes.size()) {
            continue;
        }
        auto history_keyframe_ptr_tmp = kdTree_keyframes[history_idx];
        if (!history_keyframe_ptr_tmp) {
            continue;
        }
        const int history_frame_num = history_keyframe_ptr_tmp->frame_num;
        if ((index_dist.second < curl_loop_closure_config_ptr->simple_loop_closure_search_radius_squared) &&
            (curl_voxel_mapping_ptr->spatial_hashing_ptr->loop_closure_valid_check(
                history_frame_num, current_keyframe_ptr->frame_num,
                curl_loop_closure_config_ptr->minimum_middle_segment_valid_dis_squared))) {
            if (TrajectoryLabel::is_connected(current_keyframe_ptr->trajectory_label_ptr,
                                              history_keyframe_ptr_tmp->trajectory_label_ptr)) {
                history_keyframe_ptr = history_keyframe_ptr_tmp;
                Pose3d T_his_curr;
                T_his_curr.set_pose(Eigen::Isometry3d(history_keyframe_ptr->get_T_w_lidar()).inverse().matrix() *
                                    current_keyframe_ptr->get_T_w_lidar());
                add_loop_closure_constrain(history_keyframe_ptr, current_keyframe_ptr, T_his_curr, false);
                is_scan_to_map_loop_closure = true;
                std::cout << "Scan to Map Loop Closure detected" << std::endl;
                return true;
            } else {
                sc_matches = std::make_pair(history_frame_num, index_dist.second);
                simple_loop_closure_detector = true;
                std::cout << "Simple Loop Closure detected" << std::endl;
                break;
            }
        }

        if (!TrajectoryLabel::is_connected(current_keyframe_ptr->trajectory_label_ptr,
                                           history_keyframe_ptr_tmp->trajectory_label_ptr)) {
            valid_keyframe_indices.insert(history_frame_num);
        }
    }
    if (!simple_loop_closure_detector) {
        sc_matches = curl_voxel_mapping_ptr->detectLoopClosureID(current_keyframe_ptr->frame_num);
        if (sc_matches.first == -1 || valid_keyframe_indices.find(sc_matches.first) == valid_keyframe_indices.end()) {
            return false;
        }
        std::cout << "Scan Context detected" << std::endl;
    }

    std::cout << "Successfully detect loop closure" << std::endl;
    history_keyframe_ptr = curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframe_by_frame_num(sc_matches.first);
    if (!history_keyframe_ptr) {
        return false;
    }
    Eigen::Matrix4f T_hisLidar_w =
        Eigen::Isometry3d(history_keyframe_ptr->get_T_w_lidar()).inverse().matrix().cast<float>();
    pcl::PointCloud<PointT>::Ptr history_integrate_cloud_ptr =
        get_keyframe_point_clouds(sc_matches.first, T_hisLidar_w);
    Eigen::Matrix4f T_his_curr_init = T_hisLidar_w * current_keyframe_ptr->get_T_w_lidar().template cast<float>();
    T_his_curr_init(Eigen::seq(0, 2), 3).setZero();
    std::cout << "T_his_curr_init: " << T_his_curr_init << std::endl;

    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setMaxCorrespondenceDistance(100);
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);
    assert(current_keyframe_ptr->seg_cloud_ptr != nullptr);
    assert(current_keyframe_ptr->ground_cloud_ptr != nullptr);
    pcl::PointCloud<PointT>::Ptr current_cloud_ptr =
        ((*current_keyframe_ptr->seg_cloud_ptr) + (*current_keyframe_ptr->ground_cloud_ptr)).makeShared();
    std::cout << "history_integrate_cloud_ptr size: " << history_integrate_cloud_ptr->size() << std::endl;
    std::cout << "current_cloud_ptr size: " << current_cloud_ptr->size() << std::endl;
    pcl::PointCloud<PointT>::Ptr transformed_current_cloud_ptr = std::make_shared<pcl::PointCloud<PointT>>();
    pcl::transformPointCloud(*current_cloud_ptr, *transformed_current_cloud_ptr, T_his_curr_init);
    icp.setInputSource(transformed_current_cloud_ptr);
    icp.setInputTarget(history_integrate_cloud_ptr);

    pcl::PointCloud<PointT>::Ptr unused_result(new pcl::PointCloud<PointT>());
    icp.align(*unused_result);
    bool isValidSCloopFactor = false;
    double historyKeyframeFitnessScore = 1.5;
    std::cout << "[SC] ICP fit score: " << icp.getFitnessScore() << std::endl;
    if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore) {
        std::cout << "[SC] Reject this loop (bad icp fit score, > " << historyKeyframeFitnessScore << ")" << std::endl;
        isValidSCloopFactor = false;
    } else {
        std::cout << "[SC] The detected loop factor is added between Current [ " << current_keyframe_ptr->frame_idx
                  << " ] and SC nearest [ " << history_keyframe_ptr->frame_idx << " ]" << std::endl;
        isValidSCloopFactor = true;
    }
    if (isValidSCloopFactor) {
        Eigen::Matrix4d T_his_curr_matrix = (icp.getFinalTransformation() * T_his_curr_init).cast<double>();
        Eigen::Matrix3d R = T_his_curr_matrix.block<3, 3>(0, 0);
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        R = svd.matrixU() * svd.matrixV().transpose();
        T_his_curr_matrix.block<3, 3>(0, 0) = R;

        Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_w_curr =
            history_keyframe_ptr->get_T_w_lidar() * T_his_curr_matrix;

        curl_voxel_mapping_ptr->curl_registration_method_minimum(
            current_keyframe_ptr->seg_cloud_ptr, current_keyframe_ptr->ground_cloud_ptr,
            curl_voxel_mapping_config_ptr->is_voxel_grid_filter, curl_voxel_mapping_config_ptr->leaf_size,
            history_keyframe_ptr->associated_patches, T_w_curr);
        std::cout << "Before optimization: \n" << T_his_curr_matrix << std::endl;
        T_his_curr_matrix = Eigen::Isometry3d(history_keyframe_ptr->get_T_w_lidar()).inverse().matrix() * T_w_curr;
        std::cout << "After optimization: \n" << T_his_curr_matrix << std::endl;
        Pose3d T_his_curr;
        T_his_curr.set_pose(T_his_curr_matrix);
        add_loop_closure_constrain(history_keyframe_ptr, current_keyframe_ptr, T_his_curr, true);
        pending_associated_keyframes.emplace_back(current_keyframe_ptr, history_keyframe_ptr);
        return true;
    }
    return false;
}

template <typename BasicType> void CurlPoseGraph<BasicType>::solve_pose_graph() {
    // ceres::LossFunction *loss_function = new ceres::HuberLoss(10.0);
    {
        // read lock for the keyframe set
        std::shared_lock<std::shared_mutex> sl_keyframe_set(
            curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframe_set_lock);
        // to do the optimization
        ceres::Solver::Options options;
        options.max_num_iterations = 200;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &pose_graph_problem, &summary);
        std::cout << summary.FullReport() << '\n';
    }
    {
        // TODO: update the poses of the keyframes
        // Pose updates are applied by the tracking thread after strict LC completes.
    }
    {
        // KD-tree update is applied by the tracking thread after strict LC completes.
    }
}

template <typename BasicType> void CurlPoseGraph<BasicType>::solve_local_BA_pose_graph() {

    // TODO: find the data-association between the current and the history keyframes then doing local BA to make
    // map more consistent

    // TODO: 1. find associated patches has intersected with the history keyframe started from the current
    solve_pose_graph_without_kdTree_update();
    run_local_BA_after_pose_update(current_keyframe_ptr, history_keyframe_ptr);

    // add these two keyframes' trajectory segment as neighbor
}

template <typename BasicType>
void CurlPoseGraph<BasicType>::set_loop_closure_pair(
    const std::shared_ptr<KeyframeInfo<BasicType>> &current_keyframe_ptr_in,
    const std::shared_ptr<KeyframeInfo<BasicType>> &history_keyframe_ptr_in, bool is_scan_to_map) {
    set_loop_closure_pair_with_premerge(current_keyframe_ptr_in, history_keyframe_ptr_in, is_scan_to_map,
                                        current_keyframe_ptr_in ? current_keyframe_ptr_in->trajectory_label_ptr : nullptr,
                                        history_keyframe_ptr_in ? history_keyframe_ptr_in->trajectory_label_ptr : nullptr);
}

template <typename BasicType>
void CurlPoseGraph<BasicType>::set_loop_closure_pair_with_premerge(
    const std::shared_ptr<KeyframeInfo<BasicType>> &current_keyframe_ptr_in,
    const std::shared_ptr<KeyframeInfo<BasicType>> &history_keyframe_ptr_in, bool is_scan_to_map,
    const std::shared_ptr<TrajectoryLabel> &pre_merge_current_label,
    const std::shared_ptr<TrajectoryLabel> &pre_merge_history_label,
    const std::shared_ptr<const std::unordered_set<unsigned int>> &pre_merge_current_label_frame_nums_snapshot_ptr_in,
    const std::shared_ptr<const std::unordered_set<unsigned int>> &pre_merge_history_label_frame_nums_snapshot_ptr_in,
    const std::shared_ptr<const std::unordered_set<unsigned int>> &pre_merge_current_frame_nums_snapshot_ptr_in,
    const std::shared_ptr<const std::unordered_set<unsigned int>> &pre_merge_history_frame_nums_snapshot_ptr_in) {
    current_keyframe_ptr = current_keyframe_ptr_in;
    history_keyframe_ptr = history_keyframe_ptr_in;
    is_scan_to_map_loop_closure = is_scan_to_map;
    pre_merge_current_label_ptr = pre_merge_current_label;
    pre_merge_history_label_ptr = pre_merge_history_label;
    if (pre_merge_current_label_frame_nums_snapshot_ptr_in) {
        this->pre_merge_current_label_frame_nums_snapshot_ptr = pre_merge_current_label_frame_nums_snapshot_ptr_in;
    } else if (pre_merge_current_label_ptr) {
        auto snapshot_ptr = std::make_shared<std::unordered_set<unsigned int>>();
        {
            std::shared_lock<std::shared_mutex> lock(TrajectoryLabel::label_mutex);
            snapshot_ptr->insert(pre_merge_current_label_ptr->neighbor_label_frame_num.begin(),
                                 pre_merge_current_label_ptr->neighbor_label_frame_num.end());
        }
        this->pre_merge_current_label_frame_nums_snapshot_ptr = snapshot_ptr;
    } else {
        this->pre_merge_current_label_frame_nums_snapshot_ptr.reset();
    }
    if (pre_merge_history_label_frame_nums_snapshot_ptr_in) {
        this->pre_merge_history_label_frame_nums_snapshot_ptr = pre_merge_history_label_frame_nums_snapshot_ptr_in;
    } else if (pre_merge_history_label_ptr) {
        auto snapshot_ptr = std::make_shared<std::unordered_set<unsigned int>>();
        {
            std::shared_lock<std::shared_mutex> lock(TrajectoryLabel::label_mutex);
            snapshot_ptr->insert(pre_merge_history_label_ptr->neighbor_label_frame_num.begin(),
                                 pre_merge_history_label_ptr->neighbor_label_frame_num.end());
        }
        this->pre_merge_history_label_frame_nums_snapshot_ptr = snapshot_ptr;
    } else {
        this->pre_merge_history_label_frame_nums_snapshot_ptr.reset();
    }
    if (pre_merge_current_frame_nums_snapshot_ptr_in) {
        this->pre_merge_current_frame_nums_snapshot_ptr = pre_merge_current_frame_nums_snapshot_ptr_in;
    } else {
        this->pre_merge_current_frame_nums_snapshot_ptr.reset();
    }
    if (pre_merge_history_frame_nums_snapshot_ptr_in) {
        this->pre_merge_history_frame_nums_snapshot_ptr = pre_merge_history_frame_nums_snapshot_ptr_in;
    } else {
        this->pre_merge_history_frame_nums_snapshot_ptr.reset();
    }
}

template <typename BasicType> void CurlPoseGraph<BasicType>::solve_pose_graph_initial_only() {
    solve_pose_graph_without_kdTree_update();
}

template <typename BasicType>
std::shared_ptr<OverlapRemovalPlan> CurlPoseGraph<BasicType>::build_overlap_removal_plan_after_loop() {
    if (!current_keyframe_ptr || !history_keyframe_ptr) {
        return nullptr;
    }
    const int strict_ref_frame_num = current_keyframe_ptr->frame_num;
    if (strict_ref_frame_num < 0) {
        return nullptr;
    }
    auto plan = std::make_shared<OverlapRemovalPlan>();
    plan->ref_frame_num = strict_ref_frame_num;
    auto keyframe_in_group =
        [&](const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe,
            const std::shared_ptr<const std::unordered_set<unsigned int>> &frame_nums_snapshot_ptr,
            const std::shared_ptr<const std::unordered_set<unsigned int>> &label_frame_nums_snapshot_ptr,
            const std::shared_ptr<TrajectoryLabel> &fallback_label_ptr) {
            if (!keyframe) {
                return false;
            }
            if (frame_nums_snapshot_ptr && keyframe->frame_num >= 0) {
                return frame_nums_snapshot_ptr->find(static_cast<unsigned int>(keyframe->frame_num)) !=
                       frame_nums_snapshot_ptr->end();
            }
            if (!keyframe->trajectory_label_ptr) {
                return false;
            }
            if (label_frame_nums_snapshot_ptr) {
                return label_frame_nums_snapshot_ptr->find(keyframe->trajectory_label_ptr->label_frame_num) !=
                       label_frame_nums_snapshot_ptr->end();
            }
            return fallback_label_ptr && (keyframe->trajectory_label_ptr == fallback_label_ptr);
        };
    auto is_current_group_keyframe = [&](const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe) {
        return (keyframe == current_keyframe_ptr) ||
               keyframe_in_group(keyframe, pre_merge_current_frame_nums_snapshot_ptr,
                                 pre_merge_current_label_frame_nums_snapshot_ptr, pre_merge_current_label_ptr);
    };
    auto is_history_group_keyframe = [&](const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe) {
        return (keyframe == history_keyframe_ptr) ||
               keyframe_in_group(keyframe, pre_merge_history_frame_nums_snapshot_ptr,
                                 pre_merge_history_label_frame_nums_snapshot_ptr, pre_merge_history_label_ptr);
    };
    const auto keyframes = curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframes_snapshot_reverse();
    struct KeyframeData {
        std::shared_ptr<KeyframeInfo<BasicType>> keyframe;
        std::vector<PatchId> local_patch_ids;
    };
    std::vector<KeyframeData> keyframe_datas;
    keyframe_datas.reserve(64);
    for (const auto &keyframe : keyframes) {
        if (!keyframe || keyframe->frame_num > strict_ref_frame_num) {
            continue;
        }
        const bool is_current_traj = is_current_group_keyframe(keyframe);
        if (!is_current_traj) {
            continue;
        }
        const bool is_history_traj = is_history_group_keyframe(keyframe);
        if (is_history_traj) {
            continue;
        }
        if (!keyframe->is_backend_keyframe) {
            continue;
        }
        KeyframeData kf_data;
        kf_data.keyframe = keyframe;
        {
            std::unique_lock<std::mutex> local_lock(keyframe->local_patches_lock);
            kf_data.local_patch_ids.reserve(keyframe->local_patches.size());
            for (const auto patch_id : keyframe->local_patches) {
                kf_data.local_patch_ids.push_back(patch_id);
            }
        }
        keyframe_datas.push_back(std::move(kf_data));
    }
    const int num_threads = curl_voxel_mapping_config_ptr->overlap_plan_thread_num;
    const size_t total_keyframes = keyframe_datas.size();
    std::vector<std::unordered_set<PatchId>> thread_remove_patch_ids(num_threads);
    std::vector<std::unordered_map<PatchId, int>> thread_owner_frame_nums(num_threads);
    std::vector<std::unordered_map<PatchId, PatchId>> thread_replacement_map(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        thread_remove_patch_ids[t].reserve(256);
    }
    std::size_t scanned_local_patch_num = 0;
    std::size_t candidate_remove_num = 0;
    std::size_t stale_missing_patch_num = 0;
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic) reduction(+ : scanned_local_patch_num, candidate_remove_num, stale_missing_patch_num)
    for (size_t i = 0; i < total_keyframes; ++i) {
        const auto &kf_data = keyframe_datas[i];
        const auto &keyframe = kf_data.keyframe;
        const int thread_id = omp_get_thread_num();
        for (const auto patch_id : kf_data.local_patch_ids) {
            ++scanned_local_patch_num;
            auto patch_ptr = curl_voxel_mapping_ptr->spatial_hashing_ptr->get_patch_by_id(patch_id);
            if (!patch_ptr) {
                thread_remove_patch_ids[thread_id].insert(patch_id);
                thread_owner_frame_nums[thread_id][patch_id] = keyframe->frame_num;
                ++stale_missing_patch_num;
                continue;
            }
            std::vector<std::pair<std::shared_ptr<PatchInfo<BasicType>>, double>> intersected_pairs =
                curl_voxel_mapping_ptr->spatial_hashing_ptr->query_data_association_overlap_excludeCurrKeyframe(
                    keyframe, patch_ptr->get_lower_bound_w(), patch_ptr->get_upper_bound_w());
            std::shared_ptr<PatchInfo<BasicType>> best_patch_ptr;
            double best_iou = -1.0;
            for (auto &pairs : intersected_pairs) {
                if (pairs.second <= curl_loop_closure_config_ptr->remove_overlap_iou_thres) {
                    continue;
                }
                if (!pairs.first) {
                    continue;
                }
                std::shared_ptr<KeyframeInfo<BasicType>> pair_owner_keyframe;
                {
                    std::lock_guard<std::mutex> patch_lock(pairs.first->patch_update_lock);
                    pair_owner_keyframe = pairs.first->keyframe_ptr;
                }
                if (!pair_owner_keyframe) {
                    continue;
                }
                if (pair_owner_keyframe->frame_num > strict_ref_frame_num) {
                    continue;
                }
                if (!is_history_group_keyframe(pair_owner_keyframe)) {
                    continue;
                }
                if (is_current_group_keyframe(pair_owner_keyframe)) {
                    continue;
                }
                if (pairs.second > best_iou) {
                    best_iou = pairs.second;
                    best_patch_ptr = pairs.first;
                }
            }
            if (!best_patch_ptr) {
                continue;
            }
            bool owner_matches = false;
            {
                std::lock_guard<std::mutex> patch_lock(patch_ptr->patch_update_lock);
                owner_matches = (patch_ptr->keyframe_ptr == keyframe);
            }
            if (!owner_matches) {
                std::cerr << "[Error] Patch keyframe mismatch during removal plan (Current)! Skipping." << std::endl;
                continue;
            }
            const PatchId removed_patch_id = patch_ptr->key;
            const PatchId replacement_patch_id = best_patch_ptr->key;
            thread_remove_patch_ids[thread_id].insert(removed_patch_id);
            thread_owner_frame_nums[thread_id][removed_patch_id] = keyframe->frame_num;
            if (replacement_patch_id != removed_patch_id) {
                thread_replacement_map[thread_id][removed_patch_id] = replacement_patch_id;
            }
            ++candidate_remove_num;
        }
    }
    for (int t = 0; t < num_threads; ++t) {
        plan->remove_patch_ids.insert(thread_remove_patch_ids[t].begin(), thread_remove_patch_ids[t].end());
        plan->owner_frame_nums.insert(thread_owner_frame_nums[t].begin(), thread_owner_frame_nums[t].end());
        plan->replacement_map.insert(thread_replacement_map[t].begin(), thread_replacement_map[t].end());
    }
    std::cout << "[StrictLC][PatchOverlapPlan] ref_frame=" << strict_ref_frame_num
              << " backend_keyframes=" << keyframe_datas.size()
              << " scanned_local_patches=" << scanned_local_patch_num
              << " candidates=" << candidate_remove_num
              << " stale_missing=" << stale_missing_patch_num << std::endl;
    return plan;
}

template <typename BasicType>
void CurlPoseGraph<BasicType>::apply_overlap_removal_plan(const OverlapRemovalPlan &plan) {
    if (plan.remove_patch_ids.empty()) {
        std::cout << "[StrictLC][PatchOverlap] skip_empty_plan" << std::endl;
        return;
    }
    auto merge_label_frame_nums = [](const std::shared_ptr<PatchInfo<BasicType>> &from,
                                     const std::shared_ptr<PatchInfo<BasicType>> &to) {
        if (!from || !to || from == to) {
            return;
        }
        std::scoped_lock lock(from->patch_update_lock, to->patch_update_lock);
        to->label_frame_num_set.insert(from->label_frame_num_set.begin(), from->label_frame_num_set.end());
    };
    std::unordered_map<PatchId, PatchId> associated_replacements;
    auto apply_associated_patch_replacements = [&](const std::unordered_map<PatchId, PatchId> &replacements) {
        if (replacements.empty()) {
            return;
        }
        std::unordered_map<PatchId, PatchId> valid_replacements;
        valid_replacements.reserve(replacements.size());
        for (const auto &entry : replacements) {
            if (entry.first == entry.second) {
                continue;
            }
            if (curl_voxel_mapping_ptr->spatial_hashing_ptr->get_patch_by_id(entry.second)) {
                valid_replacements.emplace(entry);
            }
        }
        if (valid_replacements.empty()) {
            return;
        }
        const auto keyframes_all = curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframes_snapshot();
        for (const auto &keyframe : keyframes_all) {
            if (!keyframe) {
                continue;
            }
            std::vector<std::pair<PatchId, PatchId>> updates;
            std::unique_lock<std::shared_mutex> assoc_lock(keyframe->associated_patches_lock);
            updates.reserve(keyframe->associated_patches.size());
            for (const auto &patch_id : keyframe->associated_patches) {
                auto it = valid_replacements.find(patch_id);
                if (it != valid_replacements.end()) {
                    updates.emplace_back(patch_id, it->second);
                }
            }
            for (const auto &update : updates) {
                keyframe->associated_patches.erase(update.first);
                keyframe->associated_patches.insert(update.second);
            }
        }
    };
    auto erase_patch_on_keyframe = [&](const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe, PatchId patch_id) {
        if (!keyframe) {
            return;
        }
        std::unique_lock<std::mutex> local_lock(keyframe->local_patches_lock);
        auto patch_id_iter = keyframe->local_patches.find(patch_id);
        if (patch_id_iter == keyframe->local_patches.end()) {
            return;
        }
        curl_voxel_mapping_ptr->spatial_hashing_ptr->erase_patch_iterator(keyframe, patch_id_iter);
    };
    std::size_t removed_patch_num = 0;
    std::size_t replaced_patch_num = 0;
    std::size_t stale_missing_patch_num = 0;
    for (const auto &patch_id : plan.remove_patch_ids) {
        int owner_frame_num = -1;
        auto owner_it = plan.owner_frame_nums.find(patch_id);
        if (owner_it != plan.owner_frame_nums.end()) {
            owner_frame_num = owner_it->second;
        }
        std::shared_ptr<KeyframeInfo<BasicType>> keyframe;
        if (owner_frame_num >= 0) {
            keyframe = curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframe_by_frame_num(owner_frame_num);
        }
        if (!keyframe) {
            continue;
        }
        auto patch_ptr = curl_voxel_mapping_ptr->spatial_hashing_ptr->get_patch_by_id(patch_id);
        if (!patch_ptr) {
            erase_patch_on_keyframe(keyframe, patch_id);
            ++stale_missing_patch_num;
            continue;
        }
        bool owner_matches = false;
        {
            std::lock_guard<std::mutex> patch_lock(patch_ptr->patch_update_lock);
            owner_matches = (patch_ptr->keyframe_ptr == keyframe);
        }
        if (!owner_matches) {
            std::cerr << "[Error] Patch keyframe mismatch during removal apply! Skipping." << std::endl;
            continue;
        }
        PatchId replacement_patch_id = patch_id;
        auto replacement_it = plan.replacement_map.find(patch_id);
        if (replacement_it != plan.replacement_map.end()) {
            replacement_patch_id = replacement_it->second;
        }
        if (replacement_patch_id != patch_id) {
            auto replacement_ptr =
                curl_voxel_mapping_ptr->spatial_hashing_ptr->get_patch_by_id(replacement_patch_id);
            if (replacement_ptr) {
                merge_label_frame_nums(patch_ptr, replacement_ptr);
                associated_replacements[patch_id] = replacement_patch_id;
                std::unique_lock<std::shared_mutex> assoc_lock(keyframe->associated_patches_lock);
                keyframe->associated_patches.insert(replacement_patch_id);
                ++replaced_patch_num;
            }
        }
        erase_patch_on_keyframe(keyframe, patch_id);
        ++removed_patch_num;
    }
    apply_associated_patch_replacements(associated_replacements);
    std::cout << "[StrictLC][PatchOverlap] plan_ref_frame=" << plan.ref_frame_num
              << " removed=" << removed_patch_num
              << " replaced=" << replaced_patch_num
              << " stale_missing=" << stale_missing_patch_num
              << " candidates=" << plan.remove_patch_ids.size() << std::endl;
}

template <typename BasicType>
void CurlPoseGraph<BasicType>::add_pending_overlap_removals(const std::unordered_set<PatchId> &ids) {
    if (ids.empty()) {
        return;
    }
    std::unique_lock<std::shared_mutex> lock(pending_overlap_remove_mutex);
    for (const auto patch_id : ids) {
        ++pending_overlap_remove_refcount[patch_id];
    }
    auto snapshot = std::make_shared<std::unordered_set<PatchId>>();
    snapshot->reserve(pending_overlap_remove_refcount.size() * 1.5);
    for (const auto &entry : pending_overlap_remove_refcount) {
        snapshot->insert(entry.first);
    }
    std::shared_ptr<const std::unordered_set<PatchId>> snapshot_const = snapshot;
    std::atomic_store(&pending_overlap_snapshot_ptr, snapshot_const);
}

template <typename BasicType>
void CurlPoseGraph<BasicType>::remove_pending_overlap_removals(const std::unordered_set<PatchId> &ids) {
    if (ids.empty()) {
        return;
    }
    std::unique_lock<std::shared_mutex> lock(pending_overlap_remove_mutex);
    for (const auto patch_id : ids) {
        auto it = pending_overlap_remove_refcount.find(patch_id);
        if (it == pending_overlap_remove_refcount.end()) {
            continue;
        }
        if (it->second <= 1) {
            pending_overlap_remove_refcount.erase(it);
        } else {
            --it->second;
        }
    }
    auto snapshot = std::make_shared<std::unordered_set<PatchId>>();
    snapshot->reserve(pending_overlap_remove_refcount.size() * 1.5);
    for (const auto &entry : pending_overlap_remove_refcount) {
        snapshot->insert(entry.first);
    }
    std::shared_ptr<const std::unordered_set<PatchId>> snapshot_const = snapshot;
    std::atomic_store(&pending_overlap_snapshot_ptr, snapshot_const);
}

template <typename BasicType>
bool CurlPoseGraph<BasicType>::is_patch_pending_overlap_removal(PatchId patch_id) const {
    std::shared_lock<std::shared_mutex> lock(pending_overlap_remove_mutex);
    return pending_overlap_remove_refcount.find(patch_id) != pending_overlap_remove_refcount.end();
}

template <typename BasicType>
void CurlPoseGraph<BasicType>::run_local_BA_after_pose_update(
    const std::shared_ptr<KeyframeInfo<BasicType>> &ba_current_keyframe_ptr,
    const std::shared_ptr<KeyframeInfo<BasicType>> &ba_history_keyframe_ptr,
    bool defer_pose_graph_updates, const std::shared_ptr<TrajectoryLabel> &history_label_filter_ptr,
    const std::shared_ptr<const std::unordered_set<unsigned int>> &history_label_frame_num_snapshot_ptr,
    int max_frame_num_for_ba, BADeferredPack<BasicType> *deferred_pack_out,
    const std::shared_ptr<const LoopPoseSnapshotMap> &loop_pose_snapshot_ptr,
    const std::shared_ptr<const std::unordered_set<unsigned int>> &history_frame_num_snapshot_ptr) {
    if (!ba_current_keyframe_ptr || !ba_history_keyframe_ptr) {
        return;
    }
    pending_overlap_checked.store(0);
    pending_overlap_skipped.store(0);
    {
        std::shared_lock<std::shared_mutex> lock(pending_overlap_remove_mutex);
        auto snapshot = std::make_shared<std::unordered_set<PatchId>>();
        snapshot->reserve(pending_overlap_remove_refcount.size() * 1.5);
        for (const auto &entry : pending_overlap_remove_refcount) {
            snapshot->insert(entry.first);
        }
        std::shared_ptr<const std::unordered_set<PatchId>> snapshot_const = snapshot;
        std::atomic_store(&pending_overlap_snapshot_ptr, snapshot_const);
    }
    struct ScopedPatchRebindFreeze {
        SpatialHashing<BasicType> *spatial_hashing_ptr = nullptr;
        std::unordered_set<int> frozen_frame_nums;
        ~ScopedPatchRebindFreeze() {
            if (spatial_hashing_ptr && !frozen_frame_nums.empty()) {
                spatial_hashing_ptr->end_skip_patch_rebind_for_keyframes(frozen_frame_nums);
            }
        }
    };
    std::map<std::shared_ptr<KeyframeInfo<BasicType>>,
             std::vector<std::pair<std::shared_ptr<KeyframeInfo<BasicType>>, int>>,
             key_frame_info_comparator<BasicType>>
        associated_keyframes; // left is current keyframe, right is history keyframe

    std::set<std::shared_ptr<KeyframeInfo<BasicType>>, key_frame_info_comparator<BasicType>> current_keyframe_sets;
    std::set<std::shared_ptr<KeyframeInfo<BasicType>>, key_frame_info_comparator<BasicType>> history_keyframes_set;
    std::set<std::shared_ptr<KeyframeInfo<BasicType>>, key_frame_info_comparator<BasicType>>
        opt_history_keyframes_set;
    std::unordered_set<std::shared_ptr<PatchInfo<BasicType>>> all_associated_history_patches; // for visualization

    if (deferred_pack_out) {
        deferred_pack_out->pending_ba_patches.clear();
        deferred_pack_out->pending_associated_keyframes.clear();
        deferred_pack_out->pending_ba_pose_graph_constrains.clear();
        deferred_pack_out->pending_ba_remove_loop_constrains.clear();
        deferred_pack_out->pending_ba_remove_icp_constrain_ptr = nullptr;
        deferred_pack_out->ba_used_keyframes.clear();
    } else if (defer_pose_graph_updates) {
        pending_ba_pose_graph_constrains.clear();
        pending_ba_remove_loop_constrains.clear();
        pending_ba_remove_icp_constrain_ptr = nullptr;
    }

    auto &target_pending_ba_patches =
        deferred_pack_out ? deferred_pack_out->pending_ba_patches : pending_ba_patches;
    auto &target_pending_associated_keyframes =
        deferred_pack_out ? deferred_pack_out->pending_associated_keyframes : pending_associated_keyframes;
    auto &target_pending_ba_pose_graph_constrains =
        deferred_pack_out ? deferred_pack_out->pending_ba_pose_graph_constrains : pending_ba_pose_graph_constrains;
    auto &target_pending_ba_remove_loop_constrains =
        deferred_pack_out ? deferred_pack_out->pending_ba_remove_loop_constrains : pending_ba_remove_loop_constrains;
    auto &target_pending_ba_remove_icp_constrain_ptr =
        deferred_pack_out ? deferred_pack_out->pending_ba_remove_icp_constrain_ptr
                          : pending_ba_remove_icp_constrain_ptr;
    // Re-initialize loop-closure pose parameters once per BA run
    const auto keyframes = curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframes_snapshot_reverse();
    auto in_ba_window = [&](const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe_ptr) {
        return keyframe_ptr && (max_frame_num_for_ba < 0 || keyframe_ptr->frame_num <= max_frame_num_for_ba);
    };
    for (const auto &keyframe_ptr : keyframes) {
        if (!in_ba_window(keyframe_ptr)) {
            continue;
        }
        if (loop_pose_snapshot_ptr) {
            auto iter = loop_pose_snapshot_ptr->find(keyframe_ptr->frame_num);
            if (iter != loop_pose_snapshot_ptr->end()) {
                const Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> snapshot_pose(
                    iter->second.data());
                keyframe_ptr->set_loopClosure_T_w_lidar_from_snapshot(snapshot_pose);
                continue;
            }
        }
        keyframe_ptr->clear_loopClosure_T_w_lidar_cache();
    }
    // TODO: 2. insert the associated patches into the local BA optimization object
    ceres::Problem problem;
    auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();
    std::shared_ptr<TrajectoryLabel> ba_history_label_filter_ptr = history_label_filter_ptr;
    if (!ba_history_label_filter_ptr && ba_history_keyframe_ptr) {
        ba_history_label_filter_ptr = ba_history_keyframe_ptr->trajectory_label_ptr;
    }
    (void)history_label_frame_num_snapshot_ptr;
    std::shared_ptr<const std::unordered_set<unsigned int>> ba_history_keyframe_frame_nums_snapshot_ptr;
    if (!history_frame_num_snapshot_ptr) {
        std::cerr << "[FATAL] run_local_BA_after_pose_update requires history_frame_num_snapshot_ptr. "
                  << "Live-label fallback has been disabled." << std::endl;
        std::abort();
    }
    {
        auto frame_nums_ptr = std::make_shared<std::unordered_set<unsigned int>>();
        frame_nums_ptr->reserve(history_frame_num_snapshot_ptr->size() + 8);
        for (const auto frame_num : *history_frame_num_snapshot_ptr) {
            if (max_frame_num_for_ba >= 0 && static_cast<int>(frame_num) > max_frame_num_for_ba) {
                continue;
            }
            frame_nums_ptr->insert(frame_num);
        }
        if (ba_history_keyframe_ptr && ba_history_keyframe_ptr->frame_num >= 0) {
            if (max_frame_num_for_ba < 0 || ba_history_keyframe_ptr->frame_num <= max_frame_num_for_ba) {
                frame_nums_ptr->insert(static_cast<unsigned int>(ba_history_keyframe_ptr->frame_num));
            }
        }
        if (!frame_nums_ptr->empty()) {
            ba_history_keyframe_frame_nums_snapshot_ptr = frame_nums_ptr;
        }
    }
    if (!ba_history_keyframe_frame_nums_snapshot_ptr) {
        std::cerr << "[FATAL] run_local_BA_after_pose_update got empty history frame snapshot after BA window "
                  << "filtering. Live-label fallback has been disabled." << std::endl;
        std::abort();
    }
    std::unordered_set<int> frozen_rebind_frame_nums;
    frozen_rebind_frame_nums.reserve(keyframes.size());
    for (const auto &keyframe_ptr : keyframes) {
        if (!in_ba_window(keyframe_ptr) || keyframe_ptr->frame_num < 0) {
            continue;
        }
        frozen_rebind_frame_nums.insert(keyframe_ptr->frame_num);
    }
    ScopedPatchRebindFreeze freeze_guard;
    if (!frozen_rebind_frame_nums.empty()) {
        curl_voxel_mapping_ptr->spatial_hashing_ptr->begin_skip_patch_rebind_for_keyframes(frozen_rebind_frame_nums);
        freeze_guard.spatial_hashing_ptr = curl_voxel_mapping_ptr->spatial_hashing_ptr.get();
        freeze_guard.frozen_frame_nums = std::move(frozen_rebind_frame_nums);
    }
    local_BA_data_association_point_clouds(problem, *ordering, associated_keyframes, all_associated_history_patches,
                                           history_keyframes_set, opt_history_keyframes_set,
                                           ba_history_label_filter_ptr, ba_history_keyframe_frame_nums_snapshot_ptr,
                                           max_frame_num_for_ba, deferred_pack_out);
    // TODO: 3. solve the local BA optimization problem
    std::cout << "********************************************" << std::endl;
    std::cout << "********************************************" << std::endl;
    std::cout << "********************************************" << std::endl;
    std::cout << associated_keyframes.size() << std::endl;
    std::cout << "********************************************" << std::endl;
    std::cout << "********************************************" << std::endl;
    std::cout << "********************************************" << std::endl;

    CurlLocalBA<BasicType>::solve_problem(problem, curl_voxel_mapping_config_ptr, ordering);
    const uint64_t overlap_checked = pending_overlap_checked.load();
    const uint64_t overlap_skipped = pending_overlap_skipped.load();
    std::cout << "[StrictLC][BA][PendingOverlap] current_frame=" << ba_current_keyframe_ptr->frame_num
              << " history_frame=" << ba_history_keyframe_ptr->frame_num << " checked=" << overlap_checked
              << " skipped=" << overlap_skipped << std::endl;
    target_pending_ba_patches.clear();
    target_pending_ba_patches.reserve(all_associated_history_patches.size());
    for (const auto &patch_info_ptr : all_associated_history_patches) {
        if (patch_info_ptr) {
            target_pending_ba_patches.emplace_back(patch_info_ptr);
        }
    }
    // give BA info into pose graph
    for (auto &keyframe_pair : associated_keyframes) {
        // add association between current keyframe and historical keyframe
        std::sort(
            keyframe_pair.second.begin(), keyframe_pair.second.end(),
            [](const std::pair<std::shared_ptr<KeyframeInfo<BasicType>>, int> &a,
               const std::pair<std::shared_ptr<KeyframeInfo<BasicType>>, int> &b) { return a.second > b.second; });
        Pose3d T_second_first;
        Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_second_first_matrix =
            Eigen::Isometry3d(keyframe_pair.second[0].first->get_loopClosure_T_w_lidar()).inverse().matrix() *
            keyframe_pair.first->get_loopClosure_T_w_lidar();
        T_second_first.set_pose(T_second_first_matrix);
        if (defer_pose_graph_updates) {
            PendingPoseGraphConstrainEntry<BasicType> pending_constrain;
            pending_constrain.history_keyframe_ptr = keyframe_pair.second[0].first;
            pending_constrain.current_keyframe_ptr = keyframe_pair.first;
            pending_constrain.T_his_curr = T_second_first;
            pending_constrain.is_icp_constrain = false;
            target_pending_ba_pose_graph_constrains.emplace_back(std::move(pending_constrain));
        } else {
            add_loop_closure_constrain(keyframe_pair.second[0].first, keyframe_pair.first, T_second_first, false);
        }
        std::cout << "add associated constrains: " << keyframe_pair.second[0].first->frame_idx << " "
                  << keyframe_pair.first->frame_idx << std::endl;
        // begin for visualization
        current_keyframe_sets.insert(keyframe_pair.first);
        // update the associated patches
        target_pending_associated_keyframes.emplace_back(keyframe_pair.first, keyframe_pair.second[0].first);
        // publish visualization
    }
    // add current keyframe
    bool is_first = true;
    std::shared_ptr<KeyframeInfo<BasicType>> last_keyframe_ptr;
    for (const auto keyframe_ptr : current_keyframe_sets) {
        // add constrains for current keyframes
        if (is_first) {
            is_first = false;
        } else {
            Pose3d T_his_curr;
            Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_his_curr_matrix =
                Eigen::Isometry3d(last_keyframe_ptr->get_loopClosure_T_w_lidar()).inverse().matrix() *
                keyframe_ptr->get_loopClosure_T_w_lidar();
            T_his_curr.set_pose(T_his_curr_matrix);
            if (defer_pose_graph_updates) {
                PendingPoseGraphConstrainEntry<BasicType> pending_constrain;
                pending_constrain.history_keyframe_ptr = last_keyframe_ptr;
                pending_constrain.current_keyframe_ptr = keyframe_ptr;
                pending_constrain.T_his_curr = T_his_curr;
                pending_constrain.is_icp_constrain = false;
                target_pending_ba_pose_graph_constrains.emplace_back(std::move(pending_constrain));
            } else {
                add_loop_closure_constrain(last_keyframe_ptr, keyframe_ptr, T_his_curr, false);
            }
            // TODO: remove old constrain
            if (keyframe_ptr->loop_closure_cost_function) {
                PoseGraph3dErrorTerm *old_loop_constrain_ptr = keyframe_ptr->loop_closure_cost_function;
                if (defer_pose_graph_updates) {
                    PendingLoopConstrainRemovalEntry<BasicType> pending_remove_entry;
                    pending_remove_entry.keyframe_ptr = keyframe_ptr;
                    pending_remove_entry.constrain_ptr = old_loop_constrain_ptr;
                    target_pending_ba_remove_loop_constrains.emplace_back(std::move(pending_remove_entry));
                } else {
                    old_loop_constrain_ptr->remove_this_constrain();
                    if (keyframe_ptr->loop_closure_cost_function == old_loop_constrain_ptr) {
                        keyframe_ptr->loop_closure_cost_function = nullptr;
                    }
                }
                std::cout << "remove current keyframe num: " << last_keyframe_ptr->frame_idx << " "
                          << keyframe_ptr->frame_idx << std::endl;
            }
        }
        last_keyframe_ptr = keyframe_ptr;
    }
    // add historical keyframe
    is_first = true;
    for (const auto keyframe_ptr : history_keyframes_set) {
        if (is_first) {
            is_first = false;
        } else {
            auto curr_keyframe_iter = opt_history_keyframes_set.find(keyframe_ptr);
            auto last_keyframe_iter = opt_history_keyframes_set.find(last_keyframe_ptr);
            if (curr_keyframe_iter != opt_history_keyframes_set.end() ||
                last_keyframe_iter != opt_history_keyframes_set.end()) {
                Pose3d T_his_curr;
                Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_his_curr_matrix =
                    Eigen::Isometry3d(last_keyframe_ptr->get_loopClosure_T_w_lidar()).inverse().matrix() *
                    keyframe_ptr->get_loopClosure_T_w_lidar();
                T_his_curr.set_pose(T_his_curr_matrix);
                if (defer_pose_graph_updates) {
                    PendingPoseGraphConstrainEntry<BasicType> pending_constrain;
                    pending_constrain.history_keyframe_ptr = last_keyframe_ptr;
                    pending_constrain.current_keyframe_ptr = keyframe_ptr;
                    pending_constrain.T_his_curr = T_his_curr;
                    pending_constrain.is_icp_constrain = false;
                    target_pending_ba_pose_graph_constrains.emplace_back(std::move(pending_constrain));
                } else {
                    add_loop_closure_constrain(last_keyframe_ptr, keyframe_ptr, T_his_curr, false);
                }
                // TODO: remove old constrain
                if (last_keyframe_ptr->frame_num == (keyframe_ptr->frame_num - 1)) {
                    if (keyframe_ptr->loop_closure_cost_function) {
                        PoseGraph3dErrorTerm *old_loop_constrain_ptr = keyframe_ptr->loop_closure_cost_function;
                        if (defer_pose_graph_updates) {
                            PendingLoopConstrainRemovalEntry<BasicType> pending_remove_entry;
                            pending_remove_entry.keyframe_ptr = keyframe_ptr;
                            pending_remove_entry.constrain_ptr = old_loop_constrain_ptr;
                            target_pending_ba_remove_loop_constrains.emplace_back(std::move(pending_remove_entry));
                        } else {
                            old_loop_constrain_ptr->remove_this_constrain();
                            if (keyframe_ptr->loop_closure_cost_function == old_loop_constrain_ptr) {
                                keyframe_ptr->loop_closure_cost_function = nullptr;
                            }
                        }
                        std::cout << "remove history keyframe num: " << last_keyframe_ptr->frame_idx << " "
                                  << keyframe_ptr->frame_idx << std::endl;
                    }
                }
            }
        }
        last_keyframe_ptr = keyframe_ptr;
    }
    // remove icp constrain
    if (icp_loop_closure_cost_function) {
        PoseGraph3dErrorTerm *old_icp_constrain_ptr = icp_loop_closure_cost_function;
        if (defer_pose_graph_updates) {
            target_pending_ba_remove_icp_constrain_ptr = old_icp_constrain_ptr;
        } else {
            old_icp_constrain_ptr->remove_icp_constrain();
            if (icp_loop_closure_cost_function == old_icp_constrain_ptr) {
                icp_loop_closure_cost_function = nullptr;
            }
        }
    }
    // TODO: Update the odometry constrain according to the BA results

    if (!defer_pose_graph_updates) {
        solve_pose_graph();
    }
    // NOTE: overlapped patches removal is deferred to the tracking thread.
}

template <typename BasicType> void CurlPoseGraph<BasicType>::remove_overlapped_patches_after_loop() {
    if (!current_keyframe_ptr || !history_keyframe_ptr) {
        return;
    }
    const int strict_ref_frame_num = current_keyframe_ptr->frame_num;
    if (strict_ref_frame_num < 0) {
        return;
    }
    auto keyframe_in_group =
        [&](const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe,
            const std::shared_ptr<const std::unordered_set<unsigned int>> &frame_nums_snapshot_ptr,
            const std::shared_ptr<const std::unordered_set<unsigned int>> &label_frame_nums_snapshot_ptr,
            const std::shared_ptr<TrajectoryLabel> &fallback_label_ptr) {
            if (!keyframe) {
                return false;
            }
            if (frame_nums_snapshot_ptr && keyframe->frame_num >= 0) {
                return frame_nums_snapshot_ptr->find(static_cast<unsigned int>(keyframe->frame_num)) !=
                       frame_nums_snapshot_ptr->end();
            }
            if (!keyframe->trajectory_label_ptr) {
                return false;
            }
            if (label_frame_nums_snapshot_ptr) {
                return label_frame_nums_snapshot_ptr->find(keyframe->trajectory_label_ptr->label_frame_num) !=
                       label_frame_nums_snapshot_ptr->end();
            }
            return fallback_label_ptr && (keyframe->trajectory_label_ptr == fallback_label_ptr);
        };
    auto is_current_group_keyframe = [&](const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe) {
        return (keyframe == current_keyframe_ptr) ||
               keyframe_in_group(keyframe, pre_merge_current_frame_nums_snapshot_ptr,
                                 pre_merge_current_label_frame_nums_snapshot_ptr, pre_merge_current_label_ptr);
    };
    auto is_history_group_keyframe = [&](const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe) {
        return (keyframe == history_keyframe_ptr) ||
               keyframe_in_group(keyframe, pre_merge_history_frame_nums_snapshot_ptr,
                                 pre_merge_history_label_frame_nums_snapshot_ptr, pre_merge_history_label_ptr);
    };
    auto merge_label_frame_nums = [](const std::shared_ptr<PatchInfo<BasicType>> &from,
                                     const std::shared_ptr<PatchInfo<BasicType>> &to) {
        if (!from || !to || from == to) {
            return;
        }
        std::scoped_lock lock(from->patch_update_lock, to->patch_update_lock);
        to->label_frame_num_set.insert(from->label_frame_num_set.begin(), from->label_frame_num_set.end());
    };
    std::unordered_map<PatchId, PatchId> associated_replacements;
    auto apply_associated_patch_replacements = [&](const std::unordered_map<PatchId, PatchId> &replacements) {
        if (replacements.empty()) {
            return;
        }
        std::unordered_map<PatchId, PatchId> valid_replacements;
        valid_replacements.reserve(replacements.size());
        for (const auto &entry : replacements) {
            if (entry.first == entry.second) {
                continue;
            }
            if (curl_voxel_mapping_ptr->spatial_hashing_ptr->get_patch_by_id(entry.second)) {
                valid_replacements.emplace(entry);
            }
        }
        if (valid_replacements.empty()) {
            return;
        }
        const auto keyframes_all = curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframes_snapshot();
        for (const auto &keyframe : keyframes_all) {
            if (!keyframe) {
                continue;
            }
            std::vector<std::pair<PatchId, PatchId>> updates;
            std::unique_lock<std::shared_mutex> assoc_lock(keyframe->associated_patches_lock);
            updates.reserve(keyframe->associated_patches.size());
            for (const auto &patch_id : keyframe->associated_patches) {
                auto it = valid_replacements.find(patch_id);
                if (it != valid_replacements.end()) {
                    updates.emplace_back(patch_id, it->second);
                }
            }
            for (const auto &update : updates) {
                keyframe->associated_patches.erase(update.first);
                keyframe->associated_patches.insert(update.second);
            }
        }
    };
    auto erase_patch_on_keyframe = [&](const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe, PatchId patch_id) {
        if (!keyframe) {
            return;
        }
        std::unique_lock<std::mutex> local_lock(keyframe->local_patches_lock);
        auto patch_id_iter = keyframe->local_patches.find(patch_id);
        if (patch_id_iter == keyframe->local_patches.end()) {
            return;
        }
        curl_voxel_mapping_ptr->spatial_hashing_ptr->erase_patch_iterator(keyframe, patch_id_iter);
    };
    std::size_t backend_current_group_keyframe_num = 0;
    std::size_t scanned_local_patch_num = 0;
    std::size_t removed_patch_num = 0;
    std::size_t replaced_patch_num = 0;
    std::size_t stale_missing_patch_num = 0;
    const auto keyframes = curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframes_snapshot_reverse();
    for (const auto &keyframe : keyframes) {
        if (!keyframe || keyframe->frame_num > strict_ref_frame_num) {
            continue;
        }
        const bool is_current_traj = is_current_group_keyframe(keyframe);
        if (!is_current_traj) {
            continue;
        }
        const bool is_history_traj = is_history_group_keyframe(keyframe);
        if (is_history_traj) {
            continue;
        }
        if (!keyframe->is_backend_keyframe) {
            continue;
        }
        ++backend_current_group_keyframe_num;
        std::vector<PatchId> local_patch_ids;
        {
            std::unique_lock<std::mutex> local_lock(keyframe->local_patches_lock);
            local_patch_ids.reserve(keyframe->local_patches.size());
            for (const auto patch_id : keyframe->local_patches) {
                local_patch_ids.push_back(patch_id);
            }
        }
        for (const auto patch_id : local_patch_ids) {
            ++scanned_local_patch_num;
            auto patch_ptr = curl_voxel_mapping_ptr->spatial_hashing_ptr->get_patch_by_id(patch_id);
            if (!patch_ptr) {
                erase_patch_on_keyframe(keyframe, patch_id);
                ++stale_missing_patch_num;
                continue;
            }
            std::vector<std::pair<std::shared_ptr<PatchInfo<BasicType>>, double>> intersected_pairs =
                curl_voxel_mapping_ptr->spatial_hashing_ptr->query_data_association_overlap_excludeCurrKeyframe(
                    keyframe, patch_ptr->get_lower_bound_w(), patch_ptr->get_upper_bound_w());
            std::shared_ptr<PatchInfo<BasicType>> best_patch_ptr;
            double best_iou = -1.0;
            for (auto &pairs : intersected_pairs) {
                if (pairs.second <= curl_loop_closure_config_ptr->remove_overlap_iou_thres) {
                    continue;
                }
                if (!pairs.first) {
                    continue;
                }
                std::shared_ptr<KeyframeInfo<BasicType>> pair_owner_keyframe;
                {
                    std::lock_guard<std::mutex> patch_lock(pairs.first->patch_update_lock);
                    pair_owner_keyframe = pairs.first->keyframe_ptr;
                }
                if (!pair_owner_keyframe) {
                    continue;
                }
                if (pair_owner_keyframe->frame_num > strict_ref_frame_num) {
                    continue;
                }
                if (!is_history_group_keyframe(pair_owner_keyframe)) {
                    continue;
                }
                if (is_current_group_keyframe(pair_owner_keyframe)) {
                    continue;
                }
                if (pairs.second > best_iou) {
                    best_iou = pairs.second;
                    best_patch_ptr = pairs.first;
                }
            }
            if (!best_patch_ptr) {
                continue;
            }
            bool owner_matches = false;
            {
                std::lock_guard<std::mutex> patch_lock(patch_ptr->patch_update_lock);
                owner_matches = (patch_ptr->keyframe_ptr == keyframe);
            }
            if (!owner_matches) {
                std::cerr << "[Error] Patch keyframe mismatch during removal (Current)! Skipping." << std::endl;
                continue;
            }
            const PatchId removed_patch_id = patch_ptr->key;
            const PatchId replacement_patch_id = best_patch_ptr->key;
            merge_label_frame_nums(patch_ptr, best_patch_ptr);
            erase_patch_on_keyframe(keyframe, patch_id);
            ++removed_patch_num;
            if (replacement_patch_id != removed_patch_id) {
                associated_replacements[removed_patch_id] = replacement_patch_id;
                std::unique_lock<std::shared_mutex> assoc_lock(keyframe->associated_patches_lock);
                keyframe->associated_patches.insert(replacement_patch_id);
                ++replaced_patch_num;
            }
        }
    }
    apply_associated_patch_replacements(associated_replacements);
    std::cout << "[StrictLC][PatchOverlap] current_group_ref_frame=" << strict_ref_frame_num
              << " backend_keyframes=" << backend_current_group_keyframe_num
              << " scanned_local_patches=" << scanned_local_patch_num
              << " removed=" << removed_patch_num
              << " replaced=" << replaced_patch_num
              << " stale_missing=" << stale_missing_patch_num << std::endl;
}

template <typename BasicType> void CurlPoseGraph<BasicType>::remove_overlapped_patches_after_loop_history() {
    if (!current_keyframe_ptr || !history_keyframe_ptr) {
        return;
    }
    const int strict_ref_frame_num = current_keyframe_ptr->frame_num;
    if (strict_ref_frame_num < 0) {
        return;
    }
    auto keyframe_in_group =
        [&](const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe,
            const std::shared_ptr<const std::unordered_set<unsigned int>> &frame_nums_snapshot_ptr,
            const std::shared_ptr<const std::unordered_set<unsigned int>> &label_frame_nums_snapshot_ptr,
            const std::shared_ptr<TrajectoryLabel> &fallback_label_ptr) {
            if (!keyframe) {
                return false;
            }
            if (frame_nums_snapshot_ptr && keyframe->frame_num >= 0) {
                return frame_nums_snapshot_ptr->find(static_cast<unsigned int>(keyframe->frame_num)) !=
                       frame_nums_snapshot_ptr->end();
            }
            if (!keyframe->trajectory_label_ptr) {
                return false;
            }
            if (label_frame_nums_snapshot_ptr) {
                return label_frame_nums_snapshot_ptr->find(keyframe->trajectory_label_ptr->label_frame_num) !=
                       label_frame_nums_snapshot_ptr->end();
            }
            return fallback_label_ptr && (keyframe->trajectory_label_ptr == fallback_label_ptr);
        };
    auto is_current_group_keyframe = [&](const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe) {
        return (keyframe == current_keyframe_ptr) ||
               keyframe_in_group(keyframe, pre_merge_current_frame_nums_snapshot_ptr,
                                 pre_merge_current_label_frame_nums_snapshot_ptr, pre_merge_current_label_ptr);
    };
    auto is_history_group_keyframe = [&](const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe) {
        return (keyframe == history_keyframe_ptr) ||
               keyframe_in_group(keyframe, pre_merge_history_frame_nums_snapshot_ptr,
                                 pre_merge_history_label_frame_nums_snapshot_ptr, pre_merge_history_label_ptr);
    };
    auto merge_label_frame_nums = [](const std::shared_ptr<PatchInfo<BasicType>> &from,
                                     const std::shared_ptr<PatchInfo<BasicType>> &to) {
        if (!from || !to || from == to) {
            return;
        }
        std::scoped_lock lock(from->patch_update_lock, to->patch_update_lock);
        to->label_frame_num_set.insert(from->label_frame_num_set.begin(), from->label_frame_num_set.end());
    };
    std::unordered_map<PatchId, PatchId> associated_replacements;
    auto apply_associated_patch_replacements = [&](const std::unordered_map<PatchId, PatchId> &replacements) {
        if (replacements.empty()) {
            return;
        }
        std::unordered_map<PatchId, PatchId> valid_replacements;
        valid_replacements.reserve(replacements.size());
        for (const auto &entry : replacements) {
            if (entry.first == entry.second) {
                continue;
            }
            if (curl_voxel_mapping_ptr->spatial_hashing_ptr->get_patch_by_id(entry.second)) {
                valid_replacements.emplace(entry);
            }
        }
        if (valid_replacements.empty()) {
            return;
        }
        const auto keyframes_all = curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframes_snapshot();
        for (const auto &keyframe : keyframes_all) {
            if (!keyframe) {
                continue;
            }
            std::vector<std::pair<PatchId, PatchId>> updates;
            std::unique_lock<std::shared_mutex> assoc_lock(keyframe->associated_patches_lock);
            updates.reserve(keyframe->associated_patches.size());
            for (const auto &patch_id : keyframe->associated_patches) {
                auto it = valid_replacements.find(patch_id);
                if (it != valid_replacements.end()) {
                    updates.emplace_back(patch_id, it->second);
                }
            }
            for (const auto &update : updates) {
                keyframe->associated_patches.erase(update.first);
                keyframe->associated_patches.insert(update.second);
            }
        }
    };
    auto erase_patch_on_keyframe = [&](const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe, PatchId patch_id) {
        if (!keyframe) {
            return;
        }
        std::unique_lock<std::mutex> local_lock(keyframe->local_patches_lock);
        auto patch_id_iter = keyframe->local_patches.find(patch_id);
        if (patch_id_iter == keyframe->local_patches.end()) {
            return;
        }
        curl_voxel_mapping_ptr->spatial_hashing_ptr->erase_patch_iterator(keyframe, patch_id_iter);
    };
    const auto keyframes = curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframes_snapshot();
    for (const auto &keyframe : keyframes) {
        if (!keyframe || keyframe->frame_num > strict_ref_frame_num) {
            continue;
        }
        const bool is_history_traj = is_history_group_keyframe(keyframe);
        if (!is_history_traj) {
            continue;
        }
        const bool is_current_traj = is_current_group_keyframe(keyframe);
        if (is_current_traj) {
            continue;
        }
        if (!keyframe->is_backend_keyframe) {
            continue;
        }
        std::vector<std::pair<size_t, double>> indices_dists;
        nanoflann::RadiusResultSet<double, size_t> resultSet(
            curl_loop_closure_config_ptr->remove_overlap_keyframe_patches_region_squared, indices_dists);
        curl::nanoflann::SearchParams params;
        Eigen::Vector3d query_pt = keyframe->get_T_w_lidar()(Eigen::seq(0, 2), 3);
        const size_t nMatches = kdTree_pose_ptr->findNeighbors(resultSet, query_pt.data(), params);
        if (indices_dists.empty()) {
            break;
        }
        bool has_nearby_history_frame = false;
        for (auto &index_dist : indices_dists) {
            const auto history_idx = static_cast<std::size_t>(index_dist.first);
            if (history_idx >= kdTree_keyframes.size()) {
                continue;
            }
            auto history_keyframe_ptr_tmp = kdTree_keyframes[history_idx];
            if (!history_keyframe_ptr_tmp) {
                continue;
            }
            if (is_history_group_keyframe(history_keyframe_ptr_tmp)) {
                has_nearby_history_frame = true;
                break;
            }
        }
        if (!has_nearby_history_frame) {
            break;
        }
        std::vector<PatchId> local_patch_ids;
        {
            std::unique_lock<std::mutex> local_lock(keyframe->local_patches_lock);
            local_patch_ids.reserve(keyframe->local_patches.size());
            for (const auto patch_id : keyframe->local_patches) {
                local_patch_ids.push_back(patch_id);
            }
        }
        for (const auto patch_id : local_patch_ids) {
            auto patch_ptr = curl_voxel_mapping_ptr->spatial_hashing_ptr->get_patch_by_id(patch_id);
            if (!patch_ptr) {
                erase_patch_on_keyframe(keyframe, patch_id);
                continue;
            }
            std::vector<std::pair<std::shared_ptr<PatchInfo<BasicType>>, double>> intersected_pairs =
                curl_voxel_mapping_ptr->spatial_hashing_ptr->query_data_association_overlap_excludeCurrKeyframe(
                    keyframe, patch_ptr->get_lower_bound_w(), patch_ptr->get_upper_bound_w());
            std::shared_ptr<PatchInfo<BasicType>> best_patch_ptr;
            double best_iou = -1.0;
            for (auto &pairs : intersected_pairs) {
                if (pairs.second <= curl_loop_closure_config_ptr->remove_overlap_iou_thres) {
                    continue;
                }
                if (!pairs.first) {
                    continue;
                }
                std::shared_ptr<KeyframeInfo<BasicType>> pair_owner_keyframe;
                {
                    std::lock_guard<std::mutex> patch_lock(pairs.first->patch_update_lock);
                    pair_owner_keyframe = pairs.first->keyframe_ptr;
                }
                if (!pair_owner_keyframe) {
                    continue;
                }
                if (pair_owner_keyframe->frame_num > strict_ref_frame_num) {
                    continue;
                }
                if (!is_current_group_keyframe(pair_owner_keyframe)) {
                    continue;
                }
                if (is_history_group_keyframe(pair_owner_keyframe)) {
                    continue;
                }
                if (pairs.second > best_iou) {
                    best_iou = pairs.second;
                    best_patch_ptr = pairs.first;
                }
            }
            if (!best_patch_ptr) {
                continue;
            }
            bool owner_matches = false;
            {
                std::lock_guard<std::mutex> patch_lock(patch_ptr->patch_update_lock);
                owner_matches = (patch_ptr->keyframe_ptr == keyframe);
            }
            if (!owner_matches) {
                std::cerr << "[Error] Patch keyframe mismatch during removal (History)! Skipping." << std::endl;
                continue;
            }
            const PatchId removed_patch_id = patch_ptr->key;
            const PatchId replacement_patch_id = best_patch_ptr->key;
            merge_label_frame_nums(patch_ptr, best_patch_ptr);
            erase_patch_on_keyframe(keyframe, patch_id);
            if (replacement_patch_id != removed_patch_id) {
                associated_replacements[removed_patch_id] = replacement_patch_id;
                std::unique_lock<std::shared_mutex> assoc_lock(keyframe->associated_patches_lock);
                keyframe->associated_patches.insert(replacement_patch_id);
            }
        }
    }
    apply_associated_patch_replacements(associated_replacements);
}

template <typename BasicType> void CurlPoseGraph<BasicType>::apply_pending_ba_pose_graph_updates() {
    if (pending_ba_pose_graph_constrains.empty() && pending_ba_remove_loop_constrains.empty() &&
        !pending_ba_remove_icp_constrain_ptr) {
        std::cout << "[StrictLC][PGO] skip_empty_pending_updates" << std::endl;
        return;
    }
    const std::size_t pgo_constraints_num = pending_ba_pose_graph_constrains.size();
    const std::size_t remove_loop_num = pending_ba_remove_loop_constrains.size();
    const bool has_remove_icp = (pending_ba_remove_icp_constrain_ptr != nullptr);
    std::cout << "[StrictLC][PGO] apply_pending_updates constraints=" << pgo_constraints_num
              << " remove_loop_constrains=" << remove_loop_num
              << " remove_icp=" << (has_remove_icp ? 1 : 0) << std::endl;
    for (const auto &pending_constrain : pending_ba_pose_graph_constrains) {
        auto history_keyframe_ptr_local = pending_constrain.history_keyframe_ptr.lock();
        auto current_keyframe_ptr_local = pending_constrain.current_keyframe_ptr.lock();
        if (!history_keyframe_ptr_local || !current_keyframe_ptr_local) {
            continue;
        }
        add_loop_closure_constrain(history_keyframe_ptr_local, current_keyframe_ptr_local, pending_constrain.T_his_curr,
                                   pending_constrain.is_icp_constrain);
    }
    for (const auto &pending_remove_entry : pending_ba_remove_loop_constrains) {
        if (!pending_remove_entry.constrain_ptr) {
            continue;
        }
        pending_remove_entry.constrain_ptr->remove_this_constrain();
        if (auto keyframe_ptr = pending_remove_entry.keyframe_ptr.lock()) {
            if (keyframe_ptr->loop_closure_cost_function == pending_remove_entry.constrain_ptr) {
                keyframe_ptr->loop_closure_cost_function = nullptr;
            }
        }
    }
    if (pending_ba_remove_icp_constrain_ptr) {
        pending_ba_remove_icp_constrain_ptr->remove_icp_constrain();
        if (icp_loop_closure_cost_function == pending_ba_remove_icp_constrain_ptr) {
            icp_loop_closure_cost_function = nullptr;
        }
    }
    pending_ba_pose_graph_constrains.clear();
    pending_ba_remove_loop_constrains.clear();
    pending_ba_remove_icp_constrain_ptr = nullptr;
    std::cout << "[StrictLC][PGO] solve_pose_graph_start (pending)" << std::endl;
    solve_pose_graph();
    std::cout << "[StrictLC][PGO] solve_pose_graph_done (pending)" << std::endl;
}

template <typename BasicType>
void CurlPoseGraph<BasicType>::apply_pending_ba_pose_graph_updates(BADeferredPack<BasicType> &pack) {
    if (pack.pending_ba_pose_graph_constrains.empty() && pack.pending_ba_remove_loop_constrains.empty() &&
        !pack.pending_ba_remove_icp_constrain_ptr) {
        std::cout << "[StrictLC][PGO] skip_empty_pack_updates" << std::endl;
        return;
    }
    const std::size_t pgo_constraints_num = pack.pending_ba_pose_graph_constrains.size();
    const std::size_t remove_loop_num = pack.pending_ba_remove_loop_constrains.size();
    const bool has_remove_icp = (pack.pending_ba_remove_icp_constrain_ptr != nullptr);
    std::cout << "[StrictLC][PGO] apply_pack_updates constraints=" << pgo_constraints_num
              << " remove_loop_constrains=" << remove_loop_num
              << " remove_icp=" << (has_remove_icp ? 1 : 0) << std::endl;
    for (const auto &pending_constrain : pack.pending_ba_pose_graph_constrains) {
        auto history_keyframe_ptr_local = pending_constrain.history_keyframe_ptr.lock();
        auto current_keyframe_ptr_local = pending_constrain.current_keyframe_ptr.lock();
        if (!history_keyframe_ptr_local || !current_keyframe_ptr_local) {
            continue;
        }
        add_loop_closure_constrain(history_keyframe_ptr_local, current_keyframe_ptr_local, pending_constrain.T_his_curr,
                                   pending_constrain.is_icp_constrain);
    }
    for (const auto &pending_remove_entry : pack.pending_ba_remove_loop_constrains) {
        if (!pending_remove_entry.constrain_ptr) {
            continue;
        }
        pending_remove_entry.constrain_ptr->remove_this_constrain();
        if (auto keyframe_ptr = pending_remove_entry.keyframe_ptr.lock()) {
            if (keyframe_ptr->loop_closure_cost_function == pending_remove_entry.constrain_ptr) {
                keyframe_ptr->loop_closure_cost_function = nullptr;
            }
        }
    }
    if (pack.pending_ba_remove_icp_constrain_ptr) {
        pack.pending_ba_remove_icp_constrain_ptr->remove_icp_constrain();
        if (icp_loop_closure_cost_function == pack.pending_ba_remove_icp_constrain_ptr) {
            icp_loop_closure_cost_function = nullptr;
        }
    }
    pack.pending_ba_pose_graph_constrains.clear();
    pack.pending_ba_remove_loop_constrains.clear();
    pack.pending_ba_remove_icp_constrain_ptr = nullptr;
    std::cout << "[StrictLC][PGO] solve_pose_graph_start (pack)" << std::endl;
    solve_pose_graph();
    std::cout << "[StrictLC][PGO] solve_pose_graph_done (pack)" << std::endl;
}

template <typename BasicType> void CurlPoseGraph<BasicType>::apply_pending_ba_updates() {
    if (pending_ba_patches.empty()) {
        std::cout << "[StrictLC][BACommit] skip_empty_pending_patches" << std::endl;
        return;
    }
    std::size_t committed_patch_num = 0;
    for (const auto &patch_weak_ptr : pending_ba_patches) {
        if (auto patch_info_ptr = patch_weak_ptr.lock()) {
            if (patch_info_ptr->patch_procession_ptr) {
                patch_info_ptr->patch_procession_ptr->commit_BA_sph_coeff();
                patch_info_ptr->patch_procession_ptr->mark_recons_v_local_dirty();
                ++committed_patch_num;
            }
        }
    }
    std::cout << "[StrictLC][BACommit] pending committed_patches=" << committed_patch_num
              << " requested=" << pending_ba_patches.size() << std::endl;
    pending_ba_patches.clear();
}

template <typename BasicType> void CurlPoseGraph<BasicType>::apply_pending_ba_updates(BADeferredPack<BasicType> &pack) {
    if (pack.pending_ba_patches.empty()) {
        std::cout << "[StrictLC][BACommit] skip_empty_pack_patches" << std::endl;
        return;
    }
    std::size_t committed_patch_num = 0;
    for (const auto &patch_weak_ptr : pack.pending_ba_patches) {
        if (auto patch_info_ptr = patch_weak_ptr.lock()) {
            if (patch_info_ptr->patch_procession_ptr) {
                patch_info_ptr->patch_procession_ptr->commit_BA_sph_coeff();
                patch_info_ptr->patch_procession_ptr->mark_recons_v_local_dirty();
                ++committed_patch_num;
            }
        }
    }
    std::cout << "[StrictLC][BACommit] pack committed_patches=" << committed_patch_num
              << " requested=" << pack.pending_ba_patches.size() << std::endl;
    pack.pending_ba_patches.clear();
}

template <typename BasicType> void CurlPoseGraph<BasicType>::apply_pending_associated_patches_updates() {
    apply_pending_associated_patches_updates_up_to_frame(-1);
}

template <typename BasicType>
void CurlPoseGraph<BasicType>::apply_pending_associated_patches_updates_up_to_frame(int max_current_frame_num) {
    if (pending_associated_keyframes.empty()) {
        std::cout << "[StrictLC][AssocPatchSync] skip_empty_pending_pairs" << std::endl;
        return;
    }
    std::size_t pair_num = 0;
    std::size_t shrink_pair_num = 0;
    std::size_t grow_pair_num = 0;
    std::size_t deferred_pair_num = 0;
    long long total_delta = 0;
    std::vector<std::pair<std::weak_ptr<KeyframeInfo<BasicType>>, std::weak_ptr<KeyframeInfo<BasicType>>>> deferred_pairs;
    deferred_pairs.reserve(pending_associated_keyframes.size());
    for (const auto &pair : pending_associated_keyframes) {
        auto current_keyframe = pair.first.lock();
        auto history_keyframe = pair.second.lock();
        if (!current_keyframe || !history_keyframe) {
            std::cout << "[StrictLC][AssocPatchSync][Pair] scope=pending status=expired_weak_ptr" << std::endl;
            continue;
        }
        if (max_current_frame_num >= 0 && current_keyframe->frame_num > max_current_frame_num) {
            deferred_pairs.emplace_back(pair);
            ++deferred_pair_num;
            std::cout << "[StrictLC][AssocPatchSync][Pair] scope=pending defer_newer_pair current_frame="
                      << current_keyframe->frame_num << " history_frame=" << history_keyframe->frame_num
                      << " max_current_frame=" << max_current_frame_num << std::endl;
            continue;
        }
        std::shared_lock<std::shared_mutex> history_lock(history_keyframe->associated_patches_lock);
        const std::size_t history_size = history_keyframe->associated_patches.size();
        std::unique_lock<std::shared_mutex> current_lock(current_keyframe->associated_patches_lock);
        const std::size_t current_before = current_keyframe->associated_patches.size();
        current_keyframe->associated_patches.insert(history_keyframe->associated_patches.begin(),
                                                    history_keyframe->associated_patches.end());
        const std::size_t current_after = current_keyframe->associated_patches.size();
        ++pair_num;
        const long long delta = static_cast<long long>(current_after) - static_cast<long long>(current_before);
        total_delta += delta;
        std::cout << "[StrictLC][AssocPatchSync][Pair] scope=pending current_frame=" << current_keyframe->frame_num
                  << " history_frame=" << history_keyframe->frame_num << " current_before=" << current_before
                  << " history_size=" << history_size << " current_after=" << current_after
                  << " delta=" << delta << std::endl;
        if (delta < 0) {
            ++shrink_pair_num;
        } else if (delta > 0) {
            ++grow_pair_num;
        }
        if (current_before > 0 && current_after * 4 < current_before) {
            std::cout << "[StrictLC][AssocPatchSync] pending severe_shrink_after_union current_frame="
                      << current_keyframe->frame_num << " history_frame=" << history_keyframe->frame_num
                      << " current_before=" << current_before << " current_after=" << current_after
                      << " history_size=" << history_size << std::endl;
        }
    }
    pending_associated_keyframes = std::move(deferred_pairs);
    std::cout << "[StrictLC][AssocPatchSync] pending pairs=" << pair_num << " shrink_pairs=" << shrink_pair_num
              << " grow_pairs=" << grow_pair_num << " total_delta=" << total_delta
              << " deferred_newer_pairs=" << deferred_pair_num
              << " remaining_pending=" << pending_associated_keyframes.size() << std::endl;
}

template <typename BasicType>
void CurlPoseGraph<BasicType>::apply_pending_associated_patches_updates(BADeferredPack<BasicType> &pack) {
    if (pack.pending_associated_keyframes.empty()) {
        std::cout << "[StrictLC][AssocPatchSync] skip_empty_pack_pairs" << std::endl;
        return;
    }
    std::size_t pair_num = 0;
    std::size_t shrink_pair_num = 0;
    std::size_t grow_pair_num = 0;
    long long total_delta = 0;
    for (const auto &pair : pack.pending_associated_keyframes) {
        auto current_keyframe = pair.first.lock();
        auto history_keyframe = pair.second.lock();
        if (!current_keyframe || !history_keyframe) {
            std::cout << "[StrictLC][AssocPatchSync][Pair] scope=pack status=expired_weak_ptr" << std::endl;
            continue;
        }
        std::shared_lock<std::shared_mutex> history_lock(history_keyframe->associated_patches_lock);
        const std::size_t history_size = history_keyframe->associated_patches.size();
        std::unique_lock<std::shared_mutex> current_lock(current_keyframe->associated_patches_lock);
        const std::size_t current_before = current_keyframe->associated_patches.size();
        current_keyframe->associated_patches.insert(history_keyframe->associated_patches.begin(),
                                                    history_keyframe->associated_patches.end());
        const std::size_t current_after = current_keyframe->associated_patches.size();
        ++pair_num;
        const long long delta = static_cast<long long>(current_after) - static_cast<long long>(current_before);
        total_delta += delta;
        std::cout << "[StrictLC][AssocPatchSync][Pair] scope=pack current_frame=" << current_keyframe->frame_num
                  << " history_frame=" << history_keyframe->frame_num << " current_before=" << current_before
                  << " history_size=" << history_size << " current_after=" << current_after
                  << " delta=" << delta << std::endl;
        if (delta < 0) {
            ++shrink_pair_num;
        } else if (delta > 0) {
            ++grow_pair_num;
        }
        if (current_before > 0 && current_after * 4 < current_before) {
            std::cout << "[StrictLC][AssocPatchSync] pack severe_shrink_after_union current_frame="
                      << current_keyframe->frame_num << " history_frame=" << history_keyframe->frame_num
                      << " current_before=" << current_before << " current_after=" << current_after
                      << " history_size=" << history_size << std::endl;
        }
    }
    std::cout << "[StrictLC][AssocPatchSync] pack pairs=" << pair_num << " shrink_pairs=" << shrink_pair_num
              << " grow_pairs=" << grow_pair_num << " total_delta=" << total_delta << std::endl;
    pack.pending_associated_keyframes.clear();
}

template <typename BasicType> void CurlPoseGraph<BasicType>::remove_overlapped_patches() {
    auto merge_label_frame_nums = [](const std::shared_ptr<PatchInfo<BasicType>> &from,
                                     const std::shared_ptr<PatchInfo<BasicType>> &to) {
        if (!from || !to || from == to) {
            return;
        }
        std::scoped_lock lock(from->patch_update_lock, to->patch_update_lock);
        to->label_frame_num_set.insert(from->label_frame_num_set.begin(), from->label_frame_num_set.end());
    };
    std::unordered_map<PatchId, PatchId> associated_replacements;
    auto apply_associated_patch_replacements = [&](const std::unordered_map<PatchId, PatchId> &replacements) {
        if (replacements.empty()) {
            return;
        }
        std::unordered_map<PatchId, PatchId> valid_replacements;
        valid_replacements.reserve(replacements.size());
        for (const auto &entry : replacements) {
            if (entry.first == entry.second) {
                continue;
            }
            if (curl_voxel_mapping_ptr->spatial_hashing_ptr->get_patch_by_id(entry.second)) {
                valid_replacements.emplace(entry);
            }
        }
        if (valid_replacements.empty()) {
            return;
        }
        const auto keyframes_all = curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframes_snapshot();
        for (const auto &keyframe : keyframes_all) {
            if (!keyframe) {
                continue;
            }
            std::vector<std::pair<PatchId, PatchId>> updates;
            std::unique_lock<std::shared_mutex> assoc_lock(keyframe->associated_patches_lock);
            updates.reserve(keyframe->associated_patches.size());
            for (const auto &patch_id : keyframe->associated_patches) {
                auto it = valid_replacements.find(patch_id);
                if (it != valid_replacements.end()) {
                    updates.emplace_back(patch_id, it->second);
                }
            }
            for (const auto &update : updates) {
                keyframe->associated_patches.erase(update.first);
                keyframe->associated_patches.insert(update.second);
            }
        }
    };
    auto erase_patch_on_keyframe = [&](const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe, PatchId patch_id) {
        if (!keyframe) {
            return;
        }
        std::unique_lock<std::mutex> local_lock(keyframe->local_patches_lock);
        auto patch_id_iter = keyframe->local_patches.find(patch_id);
        if (patch_id_iter == keyframe->local_patches.end()) {
            return;
        }
        curl_voxel_mapping_ptr->spatial_hashing_ptr->erase_patch_iterator(keyframe, patch_id_iter);
    };
    const auto keyframes = curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframes_snapshot_reverse();
    for (const auto &keyframe : keyframes) {
        std::vector<std::pair<size_t, double>> indices_dists;
        nanoflann::RadiusResultSet<double, size_t> resultSet(
            curl_loop_closure_config_ptr->remove_overlap_keyframe_patches_region_squared,
            indices_dists); // 5*5=25, squared radius
        curl::nanoflann::SearchParams params;
        int curr_frame_idx = keyframe->frame_idx;
        Eigen::Vector3d query_pt = keyframe->get_T_w_lidar()(Eigen::seq(0, 2), 3);
        const size_t nMatches = kdTree_pose_ptr->findNeighbors(resultSet, query_pt.data(), params);
        if (indices_dists.size() == 0) {
            break;
        }
        bool has_nearby_history_frame = false;
        // check the time difference between the current keyframe and the neighbour keyframes
        for (auto &index_dist : indices_dists) {
            const auto history_idx = static_cast<std::size_t>(index_dist.first);
            if (history_idx >= kdTree_keyframes.size()) {
                continue;
            }
            auto history_keyframe_ptr_tmp = kdTree_keyframes[history_idx];
            if (!history_keyframe_ptr_tmp) {
                continue;
            }
            // if the time difference is larger than a threshold, use scan-context to compare the similarity
            // between the current keyframe and the neighbour keyframes
            if (TrajectoryLabel::is_connected(history_keyframe_ptr->trajectory_label_ptr,
                                              history_keyframe_ptr_tmp->trajectory_label_ptr)) {
                has_nearby_history_frame = true;
                break;
            }
        }
        if (has_nearby_history_frame) {
            std::vector<PatchId> local_patch_ids;
            {
                std::unique_lock<std::mutex> local_lock(keyframe->local_patches_lock);
                local_patch_ids.reserve(keyframe->local_patches.size());
                for (const auto patch_id : keyframe->local_patches) {
                    local_patch_ids.push_back(patch_id);
                }
            }
            for (const auto patch_id : local_patch_ids) {
                auto patch_ptr = curl_voxel_mapping_ptr->spatial_hashing_ptr->get_patch_by_id(patch_id);
                if (!patch_ptr) {
                    erase_patch_on_keyframe(keyframe, patch_id);
                    continue;
                }
                std::vector<std::pair<std::shared_ptr<PatchInfo<BasicType>>, double>> intersected_pairs =
                    curl_voxel_mapping_ptr->spatial_hashing_ptr->query_data_association_overlap_excludeCurrKeyframe(
                        keyframe, patch_ptr->get_lower_bound_w(), patch_ptr->get_upper_bound_w());
                std::shared_ptr<PatchInfo<BasicType>> best_patch_ptr;
                double best_iou = -1.0;
                for (auto &pairs : intersected_pairs) {
                    if (pairs.second <= curl_voxel_mapping_config_ptr->highest_IoU_new_landmark_thres) {
                        continue;
                    }
                    if (!pairs.first) {
                        continue;
                    }
                    std::shared_ptr<KeyframeInfo<BasicType>> pair_owner_keyframe;
                    {
                        std::lock_guard<std::mutex> patch_lock(pairs.first->patch_update_lock);
                        pair_owner_keyframe = pairs.first->keyframe_ptr;
                    }
                    if (!(pair_owner_keyframe && pair_owner_keyframe->trajectory_label_ptr &&
                          keyframe->trajectory_label_ptr) ||
                        TrajectoryLabel::is_connected(keyframe->trajectory_label_ptr,
                                                      pair_owner_keyframe->trajectory_label_ptr)) {
                        continue;
                    }
                    if (pairs.second > best_iou) {
                        best_iou = pairs.second;
                        best_patch_ptr = pairs.first;
                    }
                }
                if (!best_patch_ptr) {
                    continue;
                }
                bool owner_matches = false;
                {
                    std::lock_guard<std::mutex> patch_lock(patch_ptr->patch_update_lock);
                    owner_matches = (patch_ptr->keyframe_ptr == keyframe);
                }
                if (!owner_matches) {
                    continue;
                }
                const PatchId removed_patch_id = patch_ptr->key;
                const PatchId replacement_patch_id = best_patch_ptr->key;
                merge_label_frame_nums(patch_ptr, best_patch_ptr);
                erase_patch_on_keyframe(keyframe, patch_id);
                if (replacement_patch_id != removed_patch_id) {
                    associated_replacements[removed_patch_id] = replacement_patch_id;
                    std::unique_lock<std::shared_mutex> assoc_lock(keyframe->associated_patches_lock);
                    keyframe->associated_patches.insert(replacement_patch_id);
                }
            }
        } else {
            break;
        }
    }
    apply_associated_patch_replacements(associated_replacements);
}

template <typename BasicType>
pcl::PointCloud<PointT>::Ptr CurlPoseGraph<BasicType>::get_keyframe_point_clouds(const int shift_idx,
                                                                                 const Eigen::Matrix4f &T_lidar_w) {
    pcl::PointCloud<PointT>::Ptr history_integrate_cloud_ptr = std::make_shared<pcl::PointCloud<PointT>>();
    auto history_keyframe_ptr = curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframe_by_frame_num(shift_idx);
    if (!history_keyframe_ptr) {
        return history_integrate_cloud_ptr;
    }
    {
        std::unique_lock<std::shared_mutex> assoc_lock(history_keyframe_ptr->associated_patches_lock);
        for (auto it = history_keyframe_ptr->associated_patches.begin();
             it != history_keyframe_ptr->associated_patches.end();) {
            auto patch_info_ptr = curl_voxel_mapping_ptr->spatial_hashing_ptr->get_patch_by_id(*it);
            if (!patch_info_ptr) {
                it = history_keyframe_ptr->associated_patches.erase(it);
                continue;
            }
            Eigen::Isometry3d T_w_obj = Eigen::Isometry3d(patch_info_ptr->keyframe_ptr->get_T_w_lidar()) *
                                        patch_info_ptr->T_obj_lidar.inverse();
            Eigen::MatrixX<BasicType> recons_v_obj =
                patch_info_ptr->patch_procession_ptr->get_recons_v_obj().transpose();
            // filter invalid points
            std::vector<BasicType> recons_v_obj_valid_vec;
            recons_v_obj_valid_vec.reserve(recons_v_obj_valid_vec.size());
            for (int i = 0; i < recons_v_obj.cols(); ++i) {
                if (std::abs(recons_v_obj(2, i)) <
                    curl_voxel_mapping_ptr->curl_voxel_mapping_config_ptr->half_diag_cut_threshold) {
                    recons_v_obj_valid_vec.push_back(recons_v_obj(0, i));
                    recons_v_obj_valid_vec.push_back(recons_v_obj(1, i));
                    recons_v_obj_valid_vec.push_back(recons_v_obj(2, i));
                }
            }
            Eigen::Map<Eigen::MatrixX<BasicType>> recons_v_obj_valid(recons_v_obj_valid_vec.data(), 3,
                                                                     recons_v_obj_valid_vec.size() / 3);
            Eigen::MatrixX<BasicType> recons_v_world =
                (T_w_obj.rotation().cast<BasicType>() * recons_v_obj_valid).colwise() +
                T_w_obj.translation().cast<BasicType>();
            for (int j = 0; j < recons_v_world.cols(); ++j) {
                PointT pt;
                pt.x = recons_v_world(0, j);
                pt.y = recons_v_world(1, j);
                pt.z = recons_v_world(2, j);
                history_integrate_cloud_ptr->push_back(pt);
            }
            ++it;
        }
    }
    pcl::transformPointCloud(*history_integrate_cloud_ptr, *history_integrate_cloud_ptr, T_lidar_w);
    curl::subSampleFrame(*history_integrate_cloud_ptr, 0.2);
    return history_integrate_cloud_ptr;
}

template <typename BasicType>
pcl::PointCloud<PointT>::Ptr
CurlPoseGraph<BasicType>::get_neighbour_keyframe_point_clouds(const int shift_idx, const Eigen::Matrix4f &T_lidar_w) {
    pcl::PointCloud<PointT>::Ptr history_integrate_cloud_ptr = std::make_shared<pcl::PointCloud<PointT>>();
    for (int i = -6; i <= 6; ++i) {
        int shift = shift_idx + i;
        if (shift < 0 || shift >= curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframe_size()) {
            continue;
        }
        auto history_keyframe_ptr = curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframe_by_frame_num(shift);
        if (!history_keyframe_ptr) {
            continue;
        }
        for (auto it = history_keyframe_ptr->local_patches.begin(); it != history_keyframe_ptr->local_patches.end();) {
            auto patch_info_ptr = curl_voxel_mapping_ptr->spatial_hashing_ptr->get_patch_by_id(*it);
            if (!patch_info_ptr) {
                it = history_keyframe_ptr->local_patches.erase(it);
                continue;
            }
            Eigen::Isometry3d T_w_obj = Eigen::Isometry3d(patch_info_ptr->keyframe_ptr->get_T_w_lidar()) *
                                        patch_info_ptr->T_obj_lidar.inverse();
            Eigen::MatrixX<BasicType> recons_v_obj =
                patch_info_ptr->patch_procession_ptr->get_recons_v_obj().transpose();
            // filter invalid points
            std::vector<BasicType> recons_v_obj_valid_vec;
            recons_v_obj_valid_vec.reserve(recons_v_obj_valid_vec.size());
            for (int i = 0; i < recons_v_obj.cols(); ++i) {
                if (std::abs(recons_v_obj(2, i)) <
                    curl_voxel_mapping_ptr->curl_voxel_mapping_config_ptr->half_diag_cut_threshold) {
                    recons_v_obj_valid_vec.push_back(recons_v_obj(0, i));
                    recons_v_obj_valid_vec.push_back(recons_v_obj(1, i));
                    recons_v_obj_valid_vec.push_back(recons_v_obj(2, i));
                }
            }
            Eigen::Map<Eigen::MatrixX<BasicType>> recons_v_obj_valid(recons_v_obj_valid_vec.data(), 3,
                                                                     recons_v_obj_valid_vec.size() / 3);
            Eigen::MatrixX<BasicType> recons_v_world =
                (T_w_obj.rotation().cast<BasicType>() * recons_v_obj_valid).colwise() +
                T_w_obj.translation().cast<BasicType>();
            for (int j = 0; j < recons_v_world.cols(); ++j) {
                PointT pt;
                pt.x = recons_v_world(0, j);
                pt.y = recons_v_world(1, j);
                pt.z = recons_v_world(2, j);
                history_integrate_cloud_ptr->push_back(pt);
            }
            ++it;
        }
    }
    pcl::transformPointCloud(*history_integrate_cloud_ptr, *history_integrate_cloud_ptr, T_lidar_w);
    curl::subSampleFrame(*history_integrate_cloud_ptr, 0.2);
    return history_integrate_cloud_ptr;
}

template <typename BasicType>
std::shared_ptr<KeyframeInfo<BasicType>> CurlPoseGraph<BasicType>::pose_graph_processing() {
    if (!is_scan_to_map_loop_closure) {
        solve_local_BA_pose_graph();
        curl_voxel_mapping_ptr->is_strict_loop_closure_detected = true;
        curl_voxel_mapping_ptr->refresh_dense_reconstruction_markers();
    } else {
        kdTree_keyframe_poses_emplace_back(current_keyframe_ptr);
    }
    curl_voxel_mapping_ptr->is_loop_closure_detected = true;
    pre_merge_current_label_ptr = current_keyframe_ptr ? current_keyframe_ptr->trajectory_label_ptr : nullptr;
    pre_merge_history_label_ptr = history_keyframe_ptr ? history_keyframe_ptr->trajectory_label_ptr : nullptr;
    // std::cout << "After merge" << std::endl;
    // std::cout << "True Current: " << current_keyframe_ptr->trajectory_label_ptr->label_frame_num << "\n";
    // for (auto &neighbor_frame_num : current_keyframe_ptr->trajectory_label_ptr->neighbor_label_frame_num) {
    //     std::cout << neighbor_frame_num << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "History: " << history_keyframe_ptr->trajectory_label_ptr->label_frame_num << "\n";
    // for (auto &neighbor_frame_num : history_keyframe_ptr->trajectory_label_ptr->neighbor_label_frame_num) {
    //     std::cout << neighbor_frame_num << " ";
    // }
    // std::cout << std::endl;
    return history_keyframe_ptr;
}

template <typename BasicType>
std::shared_ptr<KeyframeInfo<BasicType>> CurlPoseGraph<BasicType>::pose_graph_processing_for_pair(
    const std::shared_ptr<KeyframeInfo<BasicType>> &current_keyframe_ptr_in,
    const std::shared_ptr<KeyframeInfo<BasicType>> &history_keyframe_ptr_in, bool is_scan_to_map) {
    if (!current_keyframe_ptr_in || !history_keyframe_ptr_in) {
        return nullptr;
    }
    current_keyframe_ptr = current_keyframe_ptr_in;
    history_keyframe_ptr = history_keyframe_ptr_in;
    is_scan_to_map_loop_closure = is_scan_to_map;
    return pose_graph_processing();
}

template class CurlPoseGraph<BT>; // this is very important
