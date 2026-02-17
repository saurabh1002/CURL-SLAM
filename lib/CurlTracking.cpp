//
// Created by zkc on 06/06/23.
//

#include "curl_slam/CurlTracking.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <ros/ros.h>

namespace {
// Debug switch: keep Stage1 + BA queueing, but skip strict Stage3 apply on tracking thread.
constexpr bool kDisableStrictStage3ApplyForDebug = false;
// Debug switch: keep Stage3 bookkeeping/BA-commit flow, but skip Stage3 PGO solve.
constexpr bool kDisableStrictStage3PGOForDebug = false;
// Debug switch: run Stage3 PGO solve, but skip map pose application from pose graph.
constexpr bool kDisableStrictStage3PoseGraphMapUpdateForDebug = false;
// Debug switch: skip frontend correction after Stage3.
constexpr bool kDisableStrictStage3CorrectionForDebug = false;
// Debug switch: when Stage3 correction is enabled, apply frontend pose re-anchoring (T_w_j/T_w_j_1/T_j_1_j).
constexpr bool kStrictStage3CorrectionApplyFrontendPoseForDebug = true;
// Debug switch: when Stage3 correction is enabled, apply correction delta to non-backend keyframes.
constexpr bool kStrictStage3CorrectionApplyNonBackendDeltaForDebug = true;
// Debug switch: when Stage3 correction is enabled, refresh frontend keyframe anchor from backend/latest keyframe.
constexpr bool kStrictStage3CorrectionApplyKeyframeAnchorForDebug = true;
}

template <typename BasicType>
CurlTracking<BasicType>::CurlTracking(ros::NodeHandle *nh,
                                      std::shared_ptr<CURL_TRACKING_CONFIG> _curl_tracking_config_ptr,
                                      std::shared_ptr<AABB_CONFIG> _aabb_config_ptr,
                                      std::shared_ptr<SH_TABLE_CONFIG<BasicType>> _SH_table_config_ptr,
                                      std::shared_ptr<DEBUG_CONFIG> _debug_config_ptr,
                                      std::shared_ptr<CurlVoxelMapping<BasicType>> _curl_voxel_mapping_ptr,
                                      std::shared_ptr<CurlPoseGraph<BasicType>> _curl_pose_graph_ptr)
    : curl_tracking_config_ptr(_curl_tracking_config_ptr), aabb_config_ptr(_aabb_config_ptr),
      SH_table_config_ptr(_SH_table_config_ptr), debug_config_ptr(_debug_config_ptr),
      curl_voxel_mapping_ptr(_curl_voxel_mapping_ptr), curl_pose_graph_ptr(_curl_pose_graph_ptr) {
    pub_trajectory = nh->advertise<nav_msgs::Path>("/curl/trajectory", 1, true);
    pub_labelled_trajectory = nh->advertise<visualization_msgs::Marker>("curl/labelled_trajectory", 10);
    pub_used_cloud = nh->advertise<sensor_msgs::PointCloud2>("/curl/used_cloud", 10);
    pub_bounding_box = nh->advertise<visualization_msgs::Marker>("/curl/bounding_box", 10);
    trajectory.header.frame_id = "map";
    T_w_j.setIdentity();
    //    T_w_j_tangent = Sophus::SE3<double>(T_w_j).log().matrix();
    T_w_j_1.setIdentity();
    T_j_1_j.setIdentity();
    T_lastKeyframe_keyframe.setIdentity();
    valid_history_frame_idx = -1;
    frame_idx_counter = 0;
    is_initial = true;
    number_patches = 0;
    largest_pyramid_depth = 0;
    local_window_size = curl_tracking_config_ptr->local_window_size;
}

template <typename BasicType> void CurlTracking<BasicType>::run() {
    ros::Rate rate(curl_tracking_config_ptr->frequency);
    std::cout << "CurlTracking thread started" << std::endl;
    double last_cloud_time = -1.0;
    int consecutive_failure_count = 0;
    while (ros::ok()) {
        apply_pending_lc_if_any();
        // initialize a keyframe before entering following optimization
        if (!curl_voxel_mapping_ptr->pcl_ptr_pair_queue.empty() && is_initial) {
            std::vector<std::vector<Eigen::MatrixX<BasicType>>>
                point_cloud_vec; // First region_width_elements*region_width_elements is non-ground points and the last
                                 // region_width_elements*region_width_elements is ground points
            PointCloudInfo<BasicType> point_cloud_info;
            if (debug_config_ptr->is_evaluate_time) {
                preprocessing_timer.start();
            }
            pcl::PointCloud<PointT>::Ptr seg_cloud_ptr =
                std::get<1>(curl_voxel_mapping_ptr->pcl_ptr_pair_queue.front());
            pcl::PointCloud<PointT>::Ptr ground_cloud_ptr =
                std::get<2>(curl_voxel_mapping_ptr->pcl_ptr_pair_queue.front());
            double cloud_time = std::get<0>(curl_voxel_mapping_ptr->pcl_ptr_pair_queue.front());
            last_cloud_time = cloud_time;
            frame_idx_vec.push_back(frame_idx_counter++);
            curl_voxel_mapping_ptr->preprocessing_queue_lock.lock();
            curl_voxel_mapping_ptr->pcl_ptr_pair_queue.pop();
            curl_voxel_mapping_ptr->preprocessing_queue_lock.unlock();

            curl_voxel_mapping_ptr->preprocessing(cloud_time, seg_cloud_ptr, ground_cloud_ptr, T_j_1_j.matrix(),
                                                  Eigen::Matrix4f::Identity(), point_cloud_vec, point_cloud_info);
            if (debug_config_ptr->is_evaluate_time) {
                preprocessing_time_vec.push_back(preprocessing_timer.elapsedMilliseconds());
            }
            // do conformal mapping generate a keyframe
            Timer keyframe_init_timer;
            keyframe_init_timer.start();
            // NOTE: merge point_cloud_vec into a one dimensional vector
            std::vector<Eigen::MatrixX<BasicType>> point_cloud_vec_merged;
            for (const auto &vec : point_cloud_vec) {
                point_cloud_vec_merged.insert(point_cloud_vec_merged.end(), vec.begin(), vec.end());
            }
            std::vector<bool> is_ground_cloud_vec_merged;
            for (const auto &vec : point_cloud_info.is_ground_cloud_vec) {
                is_ground_cloud_vec_merged.insert(is_ground_cloud_vec_merged.end(), vec.begin(), vec.end());
            }
            std::shared_ptr<TrajectoryLabel> init_label_ptr;
            {
                std::lock_guard<std::mutex> label_lock(trajectory_label_lock);
                init_label_ptr = curr_trajectory_label_ptr;
            }
            if (curl_voxel_mapping_ptr->rt_fix_voxel_initialization(
                    true, true, point_cloud_vec_merged, is_ground_cloud_vec_merged, point_cloud_info.time, T_w_j,
                    Eigen::Isometry3d::Identity(), false, frame_idx_vec.back(), point_cloud_info.seg_cloud_ptr,
                    point_cloud_info.ground_cloud_ptr, init_label_ptr)) {
                std::shared_ptr<TrajectoryLabel> local_label_ptr;
                if (auto last_keyframe = curl_voxel_mapping_ptr->spatial_hashing_ptr->latest_keyframe()) {
                    std::lock_guard<std::mutex> label_lock(trajectory_label_lock);
                    current_label_frame_num = last_keyframe->frame_num;
                    curr_trajectory_label_ptr = last_keyframe->trajectory_label_ptr;
                    local_label_ptr = curr_trajectory_label_ptr;
                }
                if (local_label_ptr) {
                    trajectory_segment_marker = curl_voxel_mapping_ptr->set_default_marker(
                        "map", "trajectory_segment", local_label_ptr->label_frame_num, local_label_ptr->color,
                        visualization_msgs::Marker::LINE_LIST, trajectory_segment_marker_line_scale);
                }
                if (auto last_keyframe = curl_voxel_mapping_ptr->spatial_hashing_ptr->latest_keyframe()) {
                    T_w_j_keyframe = last_keyframe->get_T_w_lidar();
                    T_w_j_keyframe_frame_num = last_keyframe->frame_num;
                } else {
                    T_w_j_keyframe = T_w_j;
                    T_w_j_keyframe_frame_num = -1;
                }
                last_keyframe_idx = frame_idx_vec.back();
            }
            is_initial = false;
            std::cout << "frame id: " << frame_idx_vec.back() << std::endl;
            ros::Time current_time = ros::Time::now();
            publish_lidar_tf(current_time);
            publish_trajectory(pub_trajectory, T_w_j, trajectory, current_time);
            ++curl_voxel_mapping_ptr->frame_counter;
        }
        // after the first frame
        if (!curl_voxel_mapping_ptr->pcl_ptr_pair_queue.empty() && !is_initial) {
            double cloud_time = std::get<0>(curl_voxel_mapping_ptr->pcl_ptr_pair_queue.front());
            if (last_cloud_time > 0 && cloud_time < last_cloud_time) {
                ROS_WARN("Time reset detected (%.3f -> %.3f). Resetting system to initialization state.",
                         last_cloud_time, cloud_time);
                is_initial = true;
                T_w_j.setIdentity();
                T_w_j_1.setIdentity();
                T_j_1_j.setIdentity();
                T_lastKeyframe_keyframe.setIdentity();
                T_w_j_keyframe.setIdentity();
                T_w_j_keyframe_frame_num = -1;
                last_cloud_time = cloud_time;
                continue;
            }
            last_cloud_time = cloud_time;

            //            Timer tracking_timer;
            if (debug_config_ptr->is_evaluate_time) {
                total_timer.start();
            }
            T_w_j = (T_w_j_1 * T_j_1_j).matrix();

            pcl::PointCloud<PointT>::Ptr seg_cloud_ptr =
                std::get<1>(curl_voxel_mapping_ptr->pcl_ptr_pair_queue.front());
            pcl::PointCloud<PointT>::Ptr ground_cloud_ptr =
                std::get<2>(curl_voxel_mapping_ptr->pcl_ptr_pair_queue.front());
            // double cloud_time = std::get<0>(curl_voxel_mapping_ptr->pcl_ptr_pair_queue.front());
            frame_idx_vec.push_back(frame_idx_counter++);
            curl_voxel_mapping_ptr->preprocessing_queue_lock.lock();
            curl_voxel_mapping_ptr->pcl_ptr_pair_queue.pop();
            curl_voxel_mapping_ptr->preprocessing_queue_lock.unlock();

            // NOTE: BEGIN variables
            // Outputs
            std::vector<std::vector<Eigen::MatrixX<BasicType>>> point_cloud_vec;
            PointCloudInfo<BasicType> point_cloud_info;
            std::vector<std::tuple<Eigen::MatrixX<BasicType>, std::shared_ptr<PatchInfo<BasicType>>, double>>
                succeed_associations;
            std::pair<std::vector<double>, std::vector<double>> bounds_w;
            std::unordered_set<std::array<int, 2>, Voxel2DHashFuncPrimeArray> new_box_map;
            double final_cost;
            // for visualization
            std::vector<std::pair<std::vector<double>, std::vector<double>>> query_bounding_box_vec;
            std::vector<std::pair<std::vector<double>, std::vector<double>>> map_bounding_box_vec;
            // NOTE: END variables
            int local_label_frame_num = -1;
            std::shared_ptr<TrajectoryLabel> local_label_ptr;
            {
                std::lock_guard<std::mutex> label_lock(trajectory_label_lock);
                local_label_frame_num = current_label_frame_num;
                local_label_ptr = curr_trajectory_label_ptr;
            }
            bool is_registration_succeed = curl_voxel_mapping_ptr->curl_registration_method(
                cloud_time, last_keyframe_idx, seg_cloud_ptr, ground_cloud_ptr, local_label_frame_num, local_label_ptr,
                curl_voxel_mapping_ptr->curl_voxel_mapping_config_ptr->is_use_all_associated_patches,
                curl_voxel_mapping_ptr->curl_voxel_mapping_config_ptr->highest_IoU_new_landmark_thres,
                curl_voxel_mapping_ptr->curl_voxel_mapping_config_ptr->minimum_observation_num,
                curl_voxel_mapping_ptr->curl_voxel_mapping_config_ptr->is_voxel_grid_filter,
                curl_voxel_mapping_ptr->curl_voxel_mapping_config_ptr->leaf_size, T_w_j_keyframe,
                Eigen::Matrix4d(T_j_1_j.matrix()), T_w_j, point_cloud_vec, point_cloud_info, succeed_associations,
                bounds_w, new_box_map, number_patches, is_add_keyframe, is_add_trajectory_segment, our_costs,
                preprocessing_time_vec, data_association_time_vec, opt_time_vec, query_bounding_box_vec,
                map_bounding_box_vec);

            if (!is_registration_succeed) {
                ROS_WARN("Registration failed! Skipping frame.");
                T_j_1_j.setIdentity();
                consecutive_failure_count++;
                if (consecutive_failure_count > 5) {
                    ROS_WARN("Consecutive registration failures detected (%d). Resetting system.",
                             consecutive_failure_count);
                    is_initial = true;
                    T_w_j.setIdentity();
                    T_w_j_1.setIdentity();
                    T_j_1_j.setIdentity();
                    T_lastKeyframe_keyframe.setIdentity();
                    T_w_j_keyframe.setIdentity();
                    T_w_j_keyframe_frame_num = -1;
                    consecutive_failure_count = 0;
                }
                continue;
            }
            consecutive_failure_count = 0;

            // for visualization
            //            tracking_timer.start();
            ros::Time current_time = ros::Time::now();
            publish_lidar_tf(current_time);
            publish_trajectory(pub_trajectory, T_w_j, trajectory, current_time);
            publish_labelled_trajectory(T_w_j, T_w_j_1.matrix(), current_time);
            publish_point_cloud(current_time, succeed_associations);
            publish_query_map_bounding_box(current_time, query_bounding_box_vec, map_bounding_box_vec);
            //            std::cout << "Publishing time: " << tracking_timer.elapsedMilliseconds() << " ms" <<
            //            std::endl;
            // get odometry
            T_j_1_j = T_w_j_1.inverse() * Eigen::Isometry3d(T_w_j);
            // for debugging
            if (debug_config_ptr->is_save_final_results) {
                odometry_vector.emplace_back(T_j_1_j);
            }
            // use constant speed model to update the next initial pose
            T_w_j_1.matrix() = T_w_j;
            // add new landmarks
            //            std::set<std::pair<int, double>, PairCompare> new_box_pair_set(new_box_map.begin(),
            //            new_box_map.end());

            std::vector<Eigen::MatrixX<BasicType>> point_cloud_lidar_vec;
            point_cloud_lidar_vec.reserve(new_box_map.size());
            std::vector<bool> is_ground_cloud_vec;
            is_ground_cloud_vec.reserve(new_box_map.size());
            double time = point_cloud_info.time; // FIXME: Time is important
            for (auto &new_box_idx : new_box_map) {
                point_cloud_lidar_vec.push_back(point_cloud_vec[new_box_idx[0]][new_box_idx[1]]);
                is_ground_cloud_vec.push_back(point_cloud_info.is_ground_cloud_vec[new_box_idx[0]][new_box_idx[1]]);
            }
            if (curl_tracking_config_ptr->is_frame_to_frame) {
                omp_set_lock(&curl_voxel_mapping_ptr->writelock);
                curl_voxel_mapping_ptr->spatial_hashing_ptr->clear();
                omp_unset_lock(&curl_voxel_mapping_ptr->writelock);
            }
            T_lastKeyframe_keyframe = T_lastKeyframe_keyframe * T_j_1_j;
            std::shared_ptr<TrajectoryLabel> label_ptr_for_landmark;
            {
                std::lock_guard<std::mutex> label_lock(trajectory_label_lock);
                label_ptr_for_landmark = curr_trajectory_label_ptr;
            }
            if (is_add_keyframe) {
                add_landmark_thread(is_add_keyframe, is_add_trajectory_segment, point_cloud_lidar_vec,
                                    is_ground_cloud_vec, time, T_w_j, T_lastKeyframe_keyframe, frame_idx_vec.back(),
                                    point_cloud_info.seg_cloud_ptr, point_cloud_info.ground_cloud_ptr,
                                    label_ptr_for_landmark);
                if (is_add_trajectory_segment) {
                    std::shared_ptr<TrajectoryLabel> segment_label_ptr;
                    if (auto last_keyframe = curl_voxel_mapping_ptr->spatial_hashing_ptr->latest_keyframe()) {
                        std::lock_guard<std::mutex> label_lock(trajectory_label_lock);
                        current_label_frame_num = last_keyframe->frame_num;
                        curr_trajectory_label_ptr = last_keyframe->trajectory_label_ptr;
                        segment_label_ptr = curr_trajectory_label_ptr;
                    }
                    if (segment_label_ptr) {
                        trajectory_segment_marker = curl_voxel_mapping_ptr->set_default_marker(
                            "map", "trajectory_segment", segment_label_ptr->label_frame_num, segment_label_ptr->color,
                            visualization_msgs::Marker::LINE_LIST, trajectory_segment_marker_line_scale);
                    }
                }
                if (auto last_keyframe = curl_voxel_mapping_ptr->spatial_hashing_ptr->latest_keyframe()) {
                    T_w_j_keyframe = last_keyframe->get_T_w_lidar();
                    T_w_j_keyframe_frame_num = last_keyframe->frame_num;
                }
                T_lastKeyframe_keyframe.setIdentity();
                last_keyframe_idx = frame_idx_vec.back();
            } else {
                add_landmark_thread(is_add_keyframe, is_add_trajectory_segment, point_cloud_lidar_vec,
                                    is_ground_cloud_vec, time, T_w_j, T_lastKeyframe_keyframe, frame_idx_vec.back(),
                                    point_cloud_info.seg_cloud_ptr, point_cloud_info.ground_cloud_ptr,
                                    label_ptr_for_landmark);
                // NOTE: add current scan to patchinfo in its keyframe coordinate
                // for (const auto &original_patch_points : original_patch_points_vec) {
                //     original_patch_points.second->original_points_lidar_vec_emplace_back(
                //         (T_lastKeyframe_keyframe.matrix()(Eigen::seq(0, 2), Eigen::seq(0, 2)) *
                //          original_patch_points.first)
                //             .colwise() +
                //         T_lastKeyframe_keyframe.matrix()(Eigen::seq(0, 2), 3));
                // }
            }

            if (is_add_keyframe) {
                auto new_keyframe_ptr = curl_voxel_mapping_ptr->spatial_hashing_ptr->latest_keyframe();
                if (new_keyframe_ptr && curl_pose_graph_ptr->curl_loop_closure_config_ptr->is_enable) {
                    if (new_keyframe_ptr->frame_num == 0 && !new_keyframe_ptr->is_backend_keyframe) {
                        std::lock_guard<std::mutex> pose_lock(pose_graph_lock);
                        const bool added = curl_pose_graph_ptr->add_odometry_constrain_for_keyframe(new_keyframe_ptr);
                        new_keyframe_ptr->is_backend_keyframe = added;
                    }
                    bool loop_detected = false;
                    bool scan_to_map = false;
                    std::shared_ptr<KeyframeInfo<BasicType>> history_keyframe_ptr;
                    std::shared_ptr<TrajectoryLabel> merged_label_ptr;
                    std::shared_ptr<TrajectoryLabel> source_label_ptr = new_keyframe_ptr->trajectory_label_ptr;
                    {
                        std::lock_guard<std::mutex> pose_lock(pose_graph_lock);
                        const bool added =
                            curl_pose_graph_ptr->add_odometry_constrain_for_keyframe(new_keyframe_ptr);
                        new_keyframe_ptr->is_backend_keyframe = added;
                        loop_detected = curl_pose_graph_ptr->detect_loop_for_keyframe(new_keyframe_ptr);
                        scan_to_map = curl_pose_graph_ptr->is_scan_to_map_loop_closure_last();
                        if (loop_detected) {
                            history_keyframe_ptr = curl_pose_graph_ptr->get_history_keyframe_ptr();
                        }
                        if (loop_detected && scan_to_map) {
                            auto history_keyframe_ptr_local = curl_pose_graph_ptr->pose_graph_processing_for_pair(
                                new_keyframe_ptr, history_keyframe_ptr, true);
                            if (history_keyframe_ptr_local) {
                                merged_label_ptr = history_keyframe_ptr_local->trajectory_label_ptr;
                            }
                        }
                    }
                    if (loop_detected) {
                        std::cout << "Loop closure detected: current_frame=" << new_keyframe_ptr->frame_num 
                                  << " history_frame=" << (history_keyframe_ptr ? history_keyframe_ptr->frame_num : -1)
                                  << " scan_to_map=" << (scan_to_map ? 1 : 0) << std::endl;
                        if (scan_to_map) {
                            LCResult result;
                            result.gen = ++lc_generation;
                            result.is_strict = false;
                            result.current_keyframe_ptr = new_keyframe_ptr;
                            result.history_keyframe_ptr = history_keyframe_ptr;
                            result.merged_label_ptr = merged_label_ptr;
                            result.source_label_ptr = source_label_ptr;
                            result.ref_frame = new_keyframe_ptr->frame_num;
                            {
                                std::lock_guard<std::mutex> lock(lc_result_lock);
                                pending_lc_results.push_back(result);
                            }
                        } else {
                            if (lc_state.load() == static_cast<int>(LCState::APPLYING)) {
                                apply_pending_lc_if_any();
                            }
                            curl_pose_graph_ptr->set_skip_odometry_after_keyframe(new_keyframe_ptr);
                            trigger_strict_lc_stage1_and_queue_ba(new_keyframe_ptr, history_keyframe_ptr);
                        }
                    } else {
                        std::lock_guard<std::mutex> pose_lock(pose_graph_lock);
                        curl_pose_graph_ptr->kdTree_keyframe_poses_emplace_back(new_keyframe_ptr);
                    }
                }
            }

            // Update CURL-MAP
            if (!succeed_associations.empty()) {
                if (is_add_keyframe) {
                    if (auto last_keyframe = curl_voxel_mapping_ptr->spatial_hashing_ptr->latest_keyframe()) {
                        update_sph_coeff_thread(succeed_associations, T_w_j, last_keyframe, is_add_trajectory_segment);
                    }
                } else {
                    update_sph_coeff_thread(succeed_associations, T_w_j);
                }
            }

            // 3. save corresponding points and its information into vectors
            ++curl_voxel_mapping_ptr->frame_counter;
            double frame_total_time_ms = 0.0;
            if (debug_config_ptr->is_evaluate_time) {
                frame_total_time_ms = total_timer.elapsedMilliseconds();
                total_time_vec.push_back(frame_total_time_ms);
                const int frame_id = frame_idx_vec.back();
                std::cout << "[Timing] Frame " << frame_id << " | Total: " << frame_total_time_ms << " ms | Preprocessing: " 
                         << (preprocessing_time_vec.empty() ? 0.0 : preprocessing_time_vec.back()) << " ms | Data Association: "
                         << (data_association_time_vec.empty() ? 0.0 : data_association_time_vec.back()) << " ms | Optimization: "
                         << (opt_time_vec.empty() ? 0.0 : opt_time_vec.back()) << " ms | SPH Update: "
                         << (update_sph_time_vec.empty() ? 0.0 : update_sph_time_vec.back()) << " ms" << std::endl;
            }

            // wait for BA
            // if (curl_voxel_mapping_ptr->is_tracking_wait_BA) {
            //     std::unique_lock<std::mutex> ul(curl_voxel_mapping_ptr->BA_wait_lock);
            //     curl_voxel_mapping_ptr->tracking_wait_BA_cv.wait(ul);
            // }
            // manage memory
            //            if (curl_voxel_mapping_ptr->curl_voxel_mapping_config_ptr->largest_frame_idx_difference != -1
            //            &&
            //                curl_voxel_mapping_ptr->spatial_hashing_ptr->erase_patch_caches_of_old_keyframes(
            //                    frame_idx_vec.back(),
            //                    curl_voxel_mapping_ptr->curl_voxel_mapping_config_ptr->largest_frame_idx_difference);
            //            }

            // manage memory
            // if (!SH_table_config_ptr->is_SH_analytic_jacobian) {
            //     if (curl_voxel_mapping_ptr->curl_voxel_mapping_config_ptr->max_patch_search_box_times != -1) {
            //         std::pair<std::vector<double>, std::vector<double>> active_box_region =
            //             curl::enlarge_bounding_box_w_times(
            //                 bounds_w,
            //                 curl_voxel_mapping_ptr->curl_voxel_mapping_config_ptr->max_patch_search_box_times);
            //         curl_voxel_mapping_ptr->spatial_hashing_ptr->clear_patch_caches_out_sight(T_w_j,
            //         active_box_region);
            //     }
            // }
        }

        rate.sleep();
    }
    log();
}

template <typename BasicType> void CurlTracking<BasicType>::start_strict_lc_task(const StrictBATask &task) {
    if (!task.current_keyframe_ptr || !task.history_keyframe_ptr) {
        return;
    }
    std::size_t queued_task_num = 0;
    {
        std::lock_guard<std::mutex> lock(strict_ba_queue_lock);
        strict_ba_queue.push_back(task);
        queued_task_num = strict_ba_queue.size();
    }
    bool expected = false;
    if (strict_lc_running.compare_exchange_strong(expected, true)) {
        std::thread([this]() { strict_lc_worker(); }).detach();
    }
}

template <typename BasicType> void CurlTracking<BasicType>::strict_lc_worker() {
    while (true) {
        StrictBATask task;
        std::size_t queue_left = 0;
        {
            std::lock_guard<std::mutex> lock(strict_ba_queue_lock);
            if (strict_ba_queue.empty()) {
                break;
            }
            task = strict_ba_queue.front();
            strict_ba_queue.pop_front();
            queue_left = strict_ba_queue.size();
        }
        if (!task.current_keyframe_ptr || !task.history_keyframe_ptr) {
            if (task.overlap_plan && task.overlap_plan->pending_removals_active.exchange(false)) {
                curl_pose_graph_ptr->remove_pending_overlap_removals(task.overlap_plan->remove_patch_ids);
            }
            {
                std::lock_guard<std::mutex> done_lock(strict_stage3_mutex);
                strict_stage3_done_gen.store(task.gen);
            }
            strict_stage3_cv.notify_all();
            continue;
        }
        const int current_frame_num = task.current_keyframe_ptr->frame_num;
        const int history_frame_num = task.history_keyframe_ptr->frame_num;
        Timer ba_timer;
        ba_timer.start();
        std::atomic<bool> ba_running{true};
        BADeferredPack<BasicType> deferred_pack;
        curl_pose_graph_ptr->run_local_BA_after_pose_update(
            task.current_keyframe_ptr, task.history_keyframe_ptr, true,
            task.merged_label_ptr ? task.merged_label_ptr : task.history_keyframe_ptr->trajectory_label_ptr,
            task.premerge_history_neighbor_label_frame_nums_ptr, task.ref_frame, &deferred_pack,
            task.loop_pose_snapshot_ptr, task.premerge_history_frame_nums_snapshot_ptr);
        ba_running.store(false);
        // Reset BA usage flag for all keyframes used in this BA
        for (const auto &kf_weak : deferred_pack.ba_used_keyframes) {
            if (auto kf_ptr = kf_weak.lock()) {
                kf_ptr->is_being_used_by_ba = false;
            }
        }
        curl_voxel_mapping_ptr->is_strict_loop_closure_detected = true;
        curl_voxel_mapping_ptr->refresh_dense_reconstruction_markers();
        curl_voxel_mapping_ptr->is_loop_closure_detected = true;

        LCResult result;
        result.gen = task.gen;
        result.is_strict = true;
        result.current_keyframe_ptr = task.current_keyframe_ptr;
        result.history_keyframe_ptr = task.history_keyframe_ptr;
        result.merged_label_ptr =
            task.merged_label_ptr ? task.merged_label_ptr : task.history_keyframe_ptr->trajectory_label_ptr;
        result.source_label_ptr = task.source_label_ptr;
        result.premerge_current_neighbor_label_frame_nums_ptr = task.premerge_current_neighbor_label_frame_nums_ptr;
        result.premerge_history_neighbor_label_frame_nums_ptr = task.premerge_history_neighbor_label_frame_nums_ptr;
        result.premerge_current_frame_nums_snapshot_ptr = task.premerge_current_frame_nums_snapshot_ptr;
        result.premerge_history_frame_nums_snapshot_ptr = task.premerge_history_frame_nums_snapshot_ptr;
        result.deferred_pack = std::move(deferred_pack);
        result.overlap_plan = task.overlap_plan;
        result.ref_frame = task.ref_frame;
        {
            std::lock_guard<std::mutex> lock(lc_result_lock);
            pending_lc_results.push_back(std::move(result));
        }
        lc_state.store(static_cast<int>(LCState::APPLYING));
        {
            std::unique_lock<std::mutex> lk(strict_stage3_mutex);
            strict_stage3_cv.wait(lk, [&]() { return strict_stage3_done_gen.load() >= task.gen; });
        }
    }
    strict_lc_running.store(false);
    bool need_restart = false;
    {
        std::lock_guard<std::mutex> lock(strict_ba_queue_lock);
        need_restart = !strict_ba_queue.empty();
    }
    if (need_restart) {
        bool expected = false;
        if (strict_lc_running.compare_exchange_strong(expected, true)) {
            std::thread([this]() { strict_lc_worker(); }).detach();
        }
    }
}

template <typename BasicType>
void CurlTracking<BasicType>::trigger_strict_lc_stage1_and_queue_ba(
    const std::shared_ptr<KeyframeInfo<BasicType>> &current_keyframe_ptr,
    const std::shared_ptr<KeyframeInfo<BasicType>> &history_keyframe_ptr) {
    if (!current_keyframe_ptr || !history_keyframe_ptr) {
        return;
    }
    const uint64_t lc_gen = ++lc_generation;
    std::shared_ptr<TrajectoryLabel> source_label_ptr = current_keyframe_ptr->trajectory_label_ptr;
    std::shared_ptr<TrajectoryLabel> merged_label_ptr = history_keyframe_ptr->trajectory_label_ptr;
    std::shared_ptr<const std::unordered_set<unsigned int>> premerge_current_neighbor_label_frame_nums_ptr;
    if (source_label_ptr) {
        auto snapshot_ptr = std::make_shared<std::unordered_set<unsigned int>>();
        {
            std::shared_lock<std::shared_mutex> lock(TrajectoryLabel::label_mutex);
            snapshot_ptr->insert(source_label_ptr->neighbor_label_frame_num.begin(),
                                 source_label_ptr->neighbor_label_frame_num.end());
        }
        premerge_current_neighbor_label_frame_nums_ptr = snapshot_ptr;
    }
    std::shared_ptr<const std::unordered_set<unsigned int>> premerge_history_neighbor_label_frame_nums_ptr;
    if (merged_label_ptr) {
        auto snapshot_ptr = std::make_shared<std::unordered_set<unsigned int>>();
        {
            std::shared_lock<std::shared_mutex> lock(TrajectoryLabel::label_mutex);
            snapshot_ptr->insert(merged_label_ptr->neighbor_label_frame_num.begin(),
                                 merged_label_ptr->neighbor_label_frame_num.end());
        }
        premerge_history_neighbor_label_frame_nums_ptr = snapshot_ptr;
    }
    std::shared_ptr<const std::unordered_set<unsigned int>> premerge_current_frame_nums_snapshot_ptr;
    std::shared_ptr<const std::unordered_set<unsigned int>> premerge_history_frame_nums_snapshot_ptr;
    std::shared_ptr<const LoopPoseSnapshotMap> loop_pose_snapshot_ptr;
    std::shared_ptr<OverlapRemovalPlan> overlap_plan;
    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> stage1_reference_pose_before = Eigen::Matrix4d::Identity();
    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> stage1_reference_pose_after = Eigen::Matrix4d::Identity();
    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> stage1_delta = Eigen::Matrix4d::Identity();
    const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> stage1_tracking_pose_before = T_w_j;
    const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> stage1_tracking_prev_pose_before = T_w_j_1.matrix();
    int stage1_reference_frame_num = -1;
    bool stage1_has_reference = false;
    {
        std::lock_guard<std::mutex> pose_lock(pose_graph_lock);
        curl_pose_graph_ptr->set_loop_closure_pair_with_premerge(current_keyframe_ptr, history_keyframe_ptr, false,
                                                                 source_label_ptr, merged_label_ptr,
                                                                 premerge_current_neighbor_label_frame_nums_ptr,
                                                                 premerge_history_neighbor_label_frame_nums_ptr);
        if (auto reference_keyframe_before =
                (T_w_j_keyframe_frame_num >= 0
                     ? curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframe_by_frame_num(T_w_j_keyframe_frame_num)
                     : curl_voxel_mapping_ptr->spatial_hashing_ptr->latest_keyframe())) {
            stage1_reference_frame_num = reference_keyframe_before->frame_num;
            stage1_reference_pose_before = reference_keyframe_before->get_T_w_lidar();
            stage1_has_reference = true;
        }
        curl_pose_graph_ptr->solve_pose_graph_initial_only();
        // solve_pose_graph_initial_only calls solve_pose_graph_without_kdTree_update
        // which calls update_keyframe_pose_by_poseGraph(), so map is updated here.
        curl_pose_graph_ptr->kdTree_update_keyframe_poses();

        if (stage1_reference_frame_num >= 0) {
            if (auto reference_keyframe_after =
                    curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframe_by_frame_num(stage1_reference_frame_num)) {
                stage1_reference_pose_after = reference_keyframe_after->get_T_w_lidar();
                stage1_has_reference = true;
            } else if (auto latest_keyframe_after = curl_voxel_mapping_ptr->spatial_hashing_ptr->latest_keyframe()) {
                stage1_reference_frame_num = latest_keyframe_after->frame_num;
                stage1_reference_pose_after = latest_keyframe_after->get_T_w_lidar();
                stage1_has_reference = true;
            } else {
                stage1_has_reference = false;
            }
        } else if (auto latest_keyframe_after = curl_voxel_mapping_ptr->spatial_hashing_ptr->latest_keyframe()) {
            stage1_reference_frame_num = latest_keyframe_after->frame_num;
            stage1_reference_pose_after = latest_keyframe_after->get_T_w_lidar();
            stage1_has_reference = true;
        }

        if (stage1_has_reference) {
            stage1_delta = stage1_reference_pose_after * stage1_reference_pose_before.inverse();
        } else {
            stage1_delta = Eigen::Matrix4d::Identity();
        }
        
        const auto keyframes = curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframes_snapshot();
        auto mutable_pose_snapshot_ptr = std::make_shared<LoopPoseSnapshotMap>();
        mutable_pose_snapshot_ptr->reserve(keyframes.size() * 2 + 8);
        auto mutable_current_frame_nums_snapshot_ptr = std::make_shared<std::unordered_set<unsigned int>>();
        auto mutable_history_frame_nums_snapshot_ptr = std::make_shared<std::unordered_set<unsigned int>>();
        if (premerge_current_neighbor_label_frame_nums_ptr) {
            mutable_current_frame_nums_snapshot_ptr->reserve(premerge_current_neighbor_label_frame_nums_ptr->size() * 2 +
                                                             8);
        }
        if (premerge_history_neighbor_label_frame_nums_ptr) {
            mutable_history_frame_nums_snapshot_ptr->reserve(premerge_history_neighbor_label_frame_nums_ptr->size() * 2 +
                                                             8);
        }
        for (const auto &keyframe_ptr : keyframes) {
            if (!keyframe_ptr || keyframe_ptr->frame_num < 0 || keyframe_ptr->frame_num > current_keyframe_ptr->frame_num) {
                continue;
            }
            std::array<double, 16> pose_data{};
            Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(pose_data.data()) =
                keyframe_ptr->get_graph_pose_w_lidar_T().matrix();
            (*mutable_pose_snapshot_ptr)[keyframe_ptr->frame_num] = pose_data;
            if (premerge_current_neighbor_label_frame_nums_ptr && keyframe_ptr->trajectory_label_ptr) {
                if (premerge_current_neighbor_label_frame_nums_ptr->find(keyframe_ptr->trajectory_label_ptr->label_frame_num) !=
                    premerge_current_neighbor_label_frame_nums_ptr->end()) {
                    mutable_current_frame_nums_snapshot_ptr->insert(static_cast<unsigned int>(keyframe_ptr->frame_num));
                }
            }
            if (premerge_history_neighbor_label_frame_nums_ptr && keyframe_ptr->trajectory_label_ptr) {
                if (premerge_history_neighbor_label_frame_nums_ptr->find(keyframe_ptr->trajectory_label_ptr->label_frame_num) !=
                    premerge_history_neighbor_label_frame_nums_ptr->end()) {
                    mutable_history_frame_nums_snapshot_ptr->insert(static_cast<unsigned int>(keyframe_ptr->frame_num));
                }
            }
        }
        if (current_keyframe_ptr && current_keyframe_ptr->frame_num >= 0) {
            mutable_current_frame_nums_snapshot_ptr->insert(static_cast<unsigned int>(current_keyframe_ptr->frame_num));
        }
        if (history_keyframe_ptr && history_keyframe_ptr->frame_num >= 0 &&
            history_keyframe_ptr->frame_num <= current_keyframe_ptr->frame_num) {
            mutable_history_frame_nums_snapshot_ptr->insert(static_cast<unsigned int>(history_keyframe_ptr->frame_num));
        }
        loop_pose_snapshot_ptr = mutable_pose_snapshot_ptr;
        if (!mutable_current_frame_nums_snapshot_ptr->empty()) {
            premerge_current_frame_nums_snapshot_ptr = mutable_current_frame_nums_snapshot_ptr;
        }
        if (!mutable_history_frame_nums_snapshot_ptr->empty()) {
            premerge_history_frame_nums_snapshot_ptr = mutable_history_frame_nums_snapshot_ptr;
        }
        overlap_plan = curl_pose_graph_ptr->build_overlap_removal_plan_after_loop();
    }
    if (overlap_plan && !overlap_plan->remove_patch_ids.empty()) {
        curl_pose_graph_ptr->add_pending_overlap_removals(overlap_plan->remove_patch_ids);
        overlap_plan->pending_removals_active.store(true);
    }

    // Preserve incremental motion
    const Eigen::Isometry3d T_delta_odom = T_j_1_j;

    if (stage1_has_reference) {
        const Eigen::Matrix4d T_reference_old_j_1 =
            stage1_reference_pose_before.inverse() * stage1_tracking_prev_pose_before;
        // Re-anchor parent
        T_w_j_1.matrix() = stage1_reference_pose_after * T_reference_old_j_1;
    } else {
        T_w_j_1.matrix() = stage1_delta * stage1_tracking_prev_pose_before;
    }

    // Follow velocity constraint
    T_w_j = (T_w_j_1 * T_delta_odom).matrix();

    {
        std::lock_guard<std::mutex> pose_lock(pose_graph_lock);
        // Stage 1 Delta to Non-Backend is redundant/dangerous if map is already updated.
        // We SKIP it here to align with Stage 3 logic.
        // curl_voxel_mapping_ptr->spatial_hashing_ptr->apply_delta_to_non_backend_keyframes(stage1_delta);
        
        // Sync Anchor: Refresh pose from map
        if (T_w_j_keyframe_frame_num >= 0) {
            if (auto reference_keyframe_after =
                    curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframe_by_frame_num(T_w_j_keyframe_frame_num)) {
                T_w_j_keyframe = reference_keyframe_after->get_T_w_lidar();
            }
        } else if (auto latest_keyframe_after = curl_voxel_mapping_ptr->spatial_hashing_ptr->latest_keyframe()) {
            T_w_j_keyframe = latest_keyframe_after->get_T_w_lidar();
            T_w_j_keyframe_frame_num = latest_keyframe_after->frame_num;
        }
    }
    Eigen::Matrix4d T_last_matrix = Eigen::Isometry3d(T_w_j_keyframe).inverse().matrix() * T_w_j;
    T_last_matrix(Eigen::seq(0, 2), Eigen::seq(0, 2)) =
        Eigen::Quaterniond(Eigen::Matrix3d(T_last_matrix(Eigen::seq(0, 2), Eigen::seq(0, 2))))
            .normalized()
            .toRotationMatrix();
    T_lastKeyframe_keyframe = Eigen::Isometry3d(T_last_matrix);
    {
        ros::Time current_time = ros::Time::now();
        update_and_publish_trajectory(current_time, "strict_stage1_prequeue", lc_gen);
        publish_lidar_tf(current_time);
    }
    const int source_label_num = source_label_ptr ? source_label_ptr->label_frame_num : -1;
    const int merged_label_num = merged_label_ptr ? merged_label_ptr->label_frame_num : -1;
    if (source_label_ptr && merged_label_ptr) {
        current_keyframe_ptr->trajectory_label_ptr = merged_label_ptr;
    }
    int prev_curr_label_num = -1;
    int new_curr_label_num = -1;
    if (merged_label_ptr) {
        std::lock_guard<std::mutex> label_lock(trajectory_label_lock);
        prev_curr_label_num = curr_trajectory_label_ptr ? curr_trajectory_label_ptr->label_frame_num : -1;
        curr_trajectory_label_ptr = merged_label_ptr;
        current_label_frame_num = merged_label_ptr->label_frame_num;
        new_curr_label_num = current_label_frame_num;
    }
    lc_state.store(static_cast<int>(LCState::RUNNING));

    StrictBATask task;
    task.gen = lc_gen;
    task.current_keyframe_ptr = current_keyframe_ptr;
    task.history_keyframe_ptr = history_keyframe_ptr;
    task.source_label_ptr = source_label_ptr;
    task.merged_label_ptr = merged_label_ptr;
    task.premerge_current_neighbor_label_frame_nums_ptr = premerge_current_neighbor_label_frame_nums_ptr;
    task.premerge_history_neighbor_label_frame_nums_ptr = premerge_history_neighbor_label_frame_nums_ptr;
    task.premerge_current_frame_nums_snapshot_ptr = premerge_current_frame_nums_snapshot_ptr;
    task.premerge_history_frame_nums_snapshot_ptr = premerge_history_frame_nums_snapshot_ptr;
    task.loop_pose_snapshot_ptr = loop_pose_snapshot_ptr;
    task.overlap_plan = overlap_plan;
    task.ref_frame = current_keyframe_ptr->frame_num;
    start_strict_lc_task(task);
}

template <typename BasicType> void CurlTracking<BasicType>::apply_pending_lc_if_any() {
    std::deque<LCResult> results;
    {
        std::lock_guard<std::mutex> lock(lc_result_lock);
        if (pending_lc_results.empty()) {
            return;
        }
        results.swap(pending_lc_results);
    }
    bool strict_result_processed = false;
    auto mark_strict_stage3_done = [&](const uint64_t done_gen) {
        strict_result_processed = true;
        {
            std::lock_guard<std::mutex> done_lock(strict_stage3_mutex);
            if (done_gen > strict_stage3_done_gen.load()) {
                strict_stage3_done_gen.store(done_gen);
            }
        }
        strict_stage3_cv.notify_all();
    };
    for (auto &result : results) {
        std::atomic<uint64_t> *applied_counter_ptr =
            result.is_strict ? &lc_applied_strict : &lc_applied_non_strict;
        const uint64_t last_applied = applied_counter_ptr->load();
        if (result.gen <= last_applied) {
            if (result.is_strict) {
                if (result.overlap_plan && result.overlap_plan->pending_removals_active.exchange(false)) {
                    curl_pose_graph_ptr->remove_pending_overlap_removals(result.overlap_plan->remove_patch_ids);
                }
                mark_strict_stage3_done(result.gen);
            }
            continue;
        }
        if (result.gen > applied_counter_ptr->load()) {
            applied_counter_ptr->store(result.gen);
        }
        LCResult applied_result = std::move(result);
        if (!applied_result.merged_label_ptr && applied_result.history_keyframe_ptr) {
            applied_result.merged_label_ptr = applied_result.history_keyframe_ptr->trajectory_label_ptr;
        }
        Eigen::Matrix<double, 4, 4, Eigen::RowMajor> stage3_reference_pose_before = Eigen::Matrix4d::Identity();
        Eigen::Matrix<double, 4, 4, Eigen::RowMajor> stage3_reference_pose_after = Eigen::Matrix4d::Identity();
        const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> stage3_tracking_pose_before = T_w_j;
        const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> stage3_tracking_prev_pose_before = T_w_j_1.matrix();
        const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> stage3_tracking_keyframe_pose_before = T_w_j_keyframe;
        const int stage3_tracking_keyframe_frame_before = T_w_j_keyframe_frame_num;
        bool stage3_has_reference_before = false;
        bool stage3_has_reference_after = false;
        bool stage3_reference_from_tracking_parent = false;
        int stage3_reference_frame_num = -1;
        if (applied_result.is_strict && kDisableStrictStage3ApplyForDebug) {
            applied_result.deferred_pack.pending_ba_pose_graph_constrains.clear();
            applied_result.deferred_pack.pending_ba_remove_loop_constrains.clear();
            applied_result.deferred_pack.pending_ba_remove_icp_constrain_ptr = nullptr;
            applied_result.deferred_pack.pending_ba_patches.clear();
            applied_result.deferred_pack.pending_associated_keyframes.clear();
            post_strict_debug_gen.store(applied_result.gen);
            post_strict_debug_frames_left.store(120);
            if (applied_result.overlap_plan && applied_result.overlap_plan->pending_removals_active.exchange(false)) {
                curl_pose_graph_ptr->remove_pending_overlap_removals(applied_result.overlap_plan->remove_patch_ids);
            }
        }
        if (applied_result.is_strict && !kDisableStrictStage3ApplyForDebug) {
            int stage3_anchor_frame_num = -1;
            int stage3_current_frame_num = -1;
            int stage3_history_frame_num = -1;
            std::size_t stage3_current_assoc_before = 0;
            std::size_t stage3_history_assoc_before = 0;
            std::size_t stage3_anchor_assoc_before = 0;
            std::size_t stage3_current_assoc_after = 0;
            std::size_t stage3_history_assoc_after = 0;
            std::size_t stage3_anchor_assoc_after = 0;
            if (applied_result.current_keyframe_ptr) {
                stage3_current_frame_num = applied_result.current_keyframe_ptr->frame_num;
            }
            if (applied_result.history_keyframe_ptr) {
                stage3_history_frame_num = applied_result.history_keyframe_ptr->frame_num;
            }
            {
                std::lock_guard<std::mutex> pose_lock(pose_graph_lock);
                auto associated_patch_size =
                    [](const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe_ptr) -> std::size_t {
                    if (!keyframe_ptr) {
                        return 0;
                    }
                    std::shared_lock<std::shared_mutex> assoc_lock(keyframe_ptr->associated_patches_lock);
                    return keyframe_ptr->associated_patches.size();
                };
                applied_result.T_correction = Eigen::Matrix4d::Identity();
                std::shared_ptr<KeyframeInfo<BasicType>> stage3_anchor_keyframe_ptr;
                const auto keyframes_before_stage3 = curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframes_snapshot_reverse();
                for (const auto &keyframe_ptr : keyframes_before_stage3) {
                    if (!keyframe_ptr || !keyframe_ptr->is_backend_keyframe) {
                        continue;
                    }
                    stage3_anchor_frame_num = keyframe_ptr->frame_num;
                    stage3_anchor_keyframe_ptr = keyframe_ptr;
                    break;
                }
                stage3_current_assoc_before = associated_patch_size(applied_result.current_keyframe_ptr);
                stage3_history_assoc_before = associated_patch_size(applied_result.history_keyframe_ptr);
                stage3_anchor_assoc_before = associated_patch_size(stage3_anchor_keyframe_ptr);
                curl_pose_graph_ptr->set_loop_closure_pair_with_premerge(
                    applied_result.current_keyframe_ptr, applied_result.history_keyframe_ptr, false,
                    applied_result.source_label_ptr,
                    applied_result.merged_label_ptr ? applied_result.merged_label_ptr
                                                    : (applied_result.history_keyframe_ptr
                                                           ? applied_result.history_keyframe_ptr->trajectory_label_ptr
                                                           : nullptr),
                    applied_result.premerge_current_neighbor_label_frame_nums_ptr,
                    applied_result.premerge_history_neighbor_label_frame_nums_ptr,
                    applied_result.premerge_current_frame_nums_snapshot_ptr,
                    applied_result.premerge_history_frame_nums_snapshot_ptr);
                if (!kDisableStrictStage3PGOForDebug) {
                    if (auto reference_keyframe_before =
                            (T_w_j_keyframe_frame_num >= 0
                                 ? curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframe_by_frame_num(
                                       T_w_j_keyframe_frame_num)
                                 : stage3_anchor_keyframe_ptr)) {
                        stage3_reference_frame_num = reference_keyframe_before->frame_num;
                        stage3_reference_pose_before = reference_keyframe_before->get_T_w_lidar();
                        stage3_has_reference_before = true;
                        stage3_reference_from_tracking_parent =
                            (T_w_j_keyframe_frame_num >= 0 &&
                             reference_keyframe_before->frame_num == T_w_j_keyframe_frame_num);
                    } else if (auto latest_reference_before =
                                   curl_voxel_mapping_ptr->spatial_hashing_ptr->latest_keyframe()) {
                        stage3_reference_frame_num = latest_reference_before->frame_num;
                        stage3_reference_pose_before = latest_reference_before->get_T_w_lidar();
                        stage3_has_reference_before = true;
                        stage3_reference_from_tracking_parent = false;
                    }
                    curl_pose_graph_ptr->apply_pending_ba_pose_graph_updates(applied_result.deferred_pack);

                    if (!kDisableStrictStage3PoseGraphMapUpdateForDebug) {
                        // 1. Update Map FIRST
                        curl_voxel_mapping_ptr->spatial_hashing_ptr->update_keyframe_pose_by_poseGraph();
                        curl_pose_graph_ptr->kdTree_update_keyframe_poses();

                        // 2. Capture results AFTER map update
                        if (stage3_reference_frame_num >= 0) {
                            if (auto reference_keyframe_after =
                                    curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframe_by_frame_num(
                                        stage3_reference_frame_num)) {
                                stage3_reference_pose_after = reference_keyframe_after->get_T_w_lidar();
                                stage3_has_reference_after = true;
                            } else if (auto latest_reference_after =
                                           curl_voxel_mapping_ptr->spatial_hashing_ptr->latest_keyframe()) {
                                stage3_reference_frame_num = latest_reference_after->frame_num;
                                stage3_reference_pose_after = latest_reference_after->get_T_w_lidar();
                                stage3_has_reference_after = true;
                            }
                        } else if (auto latest_reference_after =
                                       curl_voxel_mapping_ptr->spatial_hashing_ptr->latest_keyframe()) {
                            stage3_reference_frame_num = latest_reference_after->frame_num;
                            stage3_reference_pose_after = latest_reference_after->get_T_w_lidar();
                            stage3_has_reference_after = true;
                        }

                        if (stage3_has_reference_before && stage3_has_reference_after) {
                            applied_result.T_correction =
                                stage3_reference_pose_after * stage3_reference_pose_before.inverse();
                        } else {
                            applied_result.T_correction = Eigen::Matrix4d::Identity();
                        }

                    } else {
                        applied_result.T_correction = Eigen::Matrix4d::Identity();
                    }
                } else {
                    applied_result.T_correction = Eigen::Matrix4d::Identity();
                }
                curl_pose_graph_ptr->apply_pending_ba_updates(applied_result.deferred_pack);
                curl_pose_graph_ptr->apply_pending_associated_patches_updates(applied_result.deferred_pack);
                if (stage3_current_frame_num >= 0) {
                    curl_pose_graph_ptr->apply_pending_associated_patches_updates_up_to_frame(
                        stage3_current_frame_num);
                } else {
                    curl_pose_graph_ptr->apply_pending_associated_patches_updates();
                }
                if (applied_result.overlap_plan && !applied_result.overlap_plan->remove_patch_ids.empty()) {
                    curl_pose_graph_ptr->apply_overlap_removal_plan(*applied_result.overlap_plan);
                    if (applied_result.overlap_plan->pending_removals_active.exchange(false)) {
                        curl_pose_graph_ptr->remove_pending_overlap_removals(
                            applied_result.overlap_plan->remove_patch_ids);
                    }
                }
                curl_voxel_mapping_ptr->is_strict_loop_closure_detected = false;
                // Clear clouds for keyframes in the same trajectory (before merge) with frame_num < current_keyframe's frame_num
                // Also reset the BA usage flags
                if (applied_result.is_strict && applied_result.premerge_current_frame_nums_snapshot_ptr && applied_result.current_keyframe_ptr) {
                    int current_frame_num = applied_result.current_keyframe_ptr->frame_num;
                    for (const auto &kf_ptr : curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframes_snapshot_reverse()) {
                        if (!kf_ptr || kf_ptr->frame_num >= current_frame_num) {
                            continue;
                        }
                        // Check if this keyframe is in the premerge trajectory
                        if (applied_result.premerge_current_frame_nums_snapshot_ptr->count(kf_ptr->frame_num) > 0) {
                            // Skip if being used by another running BA
                            if (kf_ptr->is_being_used_by_ba.load()) {
                                continue;
                            }
                            kf_ptr->seg_cloud_ptr.reset();
                            kf_ptr->ground_cloud_ptr.reset();                            
                        }
                    }
                }
                if (applied_result.source_label_ptr && applied_result.merged_label_ptr) {
                    TrajectoryLabel::merge_neighbor_label(applied_result.source_label_ptr,
                                                          applied_result.merged_label_ptr);
                    if (applied_result.current_keyframe_ptr) {
                        applied_result.current_keyframe_ptr->trajectory_label_ptr = applied_result.merged_label_ptr;
                    }
                }
                stage3_current_assoc_after = associated_patch_size(applied_result.current_keyframe_ptr);
                stage3_history_assoc_after = associated_patch_size(applied_result.history_keyframe_ptr);
                if (stage3_anchor_frame_num >= 0) {
                    if (auto stage3_anchor_keyframe_after =
                            curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframe_by_frame_num(
                                stage3_anchor_frame_num)) {
                        stage3_anchor_assoc_after = associated_patch_size(stage3_anchor_keyframe_after);
                    } else {
                        stage3_anchor_assoc_after = 0;
                    }
                } else {
                    stage3_anchor_assoc_after = 0;
                }
                const long long delta_current = static_cast<long long>(stage3_current_assoc_after) -
                                                static_cast<long long>(stage3_current_assoc_before);
                const long long delta_history = static_cast<long long>(stage3_history_assoc_after) -
                                                static_cast<long long>(stage3_history_assoc_before);
                const long long delta_anchor = static_cast<long long>(stage3_anchor_assoc_after) -
                                               static_cast<long long>(stage3_anchor_assoc_before);
            }
        }
        if (applied_result.is_strict && !kDisableStrictStage3ApplyForDebug) {
            const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> correction_delta = applied_result.T_correction;
            const double correction_translation_norm =
                correction_delta(Eigen::seq(0, 2), 3).template cast<double>().norm();
            Eigen::Matrix3d correction_rotation =
                correction_delta(Eigen::seq(0, 2), Eigen::seq(0, 2)).template cast<double>();
            correction_rotation = Eigen::Quaterniond(correction_rotation).normalized().toRotationMatrix();
            constexpr double kRad2Deg = 57.2957795130823208768;
            double correction_rotation_deg = std::abs(Eigen::AngleAxisd(correction_rotation).angle() * kRad2Deg);
            int curr_label_num_for_debug = -1;
            {
                std::lock_guard<std::mutex> label_lock(trajectory_label_lock);
                curr_label_num_for_debug =
                    curr_trajectory_label_ptr ? curr_trajectory_label_ptr->label_frame_num : -1;
            }
            if (kDisableStrictStage3CorrectionForDebug) {
            } else {
                // 3. Preserve incremental motion (velocity)
                const Eigen::Isometry3d T_delta_odom = T_j_1_j;
                const Eigen::Matrix4d old_T_w_j_1 = stage3_tracking_prev_pose_before;

                if (stage3_has_reference_before && stage3_has_reference_after) {
                    const Eigen::Matrix4d T_reference_old_j_1 =
                        stage3_reference_pose_before.inverse() * old_T_w_j_1;
                    // 4. Re-anchor parent (j-1)
                    T_w_j_1.matrix() = stage3_reference_pose_after * T_reference_old_j_1;
                } else {
                    T_w_j_1.matrix() = correction_delta * old_T_w_j_1;
                }

                // 5. Derive current pose (j) from new j-1 + original delta
                T_w_j = (T_w_j_1 * T_delta_odom).matrix();
                
                {
                    std::lock_guard<std::mutex> pose_lock(pose_graph_lock);
                    // 6. Sync Anchor: Refresh pose from map, keep frame_num invariant
                    if (T_w_j_keyframe_frame_num >= 0) {
                        if (auto cur_anchor = curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframe_by_frame_num(
                                T_w_j_keyframe_frame_num)) {
                            T_w_j_keyframe = cur_anchor->get_T_w_lidar();
                        } else {
                            // Fallback (should rarely happen)
                            T_w_j_keyframe = correction_delta * stage3_tracking_keyframe_pose_before;
                        }
                    } else if (auto latest_keyframe_ptr =
                                   curl_voxel_mapping_ptr->spatial_hashing_ptr->latest_keyframe()) {
                        T_w_j_keyframe = latest_keyframe_ptr->get_T_w_lidar();
                        T_w_j_keyframe_frame_num = latest_keyframe_ptr->frame_num;
                    } else {
                        T_w_j_keyframe = correction_delta * T_w_j_keyframe;
                    }
                }

                // 7. Update relative transform for next ICP
                Eigen::Matrix4d T_last_matrix = Eigen::Isometry3d(T_w_j_keyframe).inverse().matrix() * T_w_j;
                T_last_matrix(Eigen::seq(0, 2), Eigen::seq(0, 2)) =
                    Eigen::Quaterniond(Eigen::Matrix3d(T_last_matrix(Eigen::seq(0, 2), Eigen::seq(0, 2))))
                        .normalized()
                        .toRotationMatrix();
                T_lastKeyframe_keyframe = Eigen::Isometry3d(T_last_matrix);

            }
            ros::Time current_time = ros::Time::now();
            update_and_publish_trajectory(current_time, "strict_stage3_apply", applied_result.gen);
            publish_lidar_tf(current_time);
            post_strict_debug_gen.store(applied_result.gen);
            post_strict_debug_frames_left.store(120);
        }
        if (!applied_result.is_strict) {
            if (applied_result.source_label_ptr && applied_result.merged_label_ptr) {
                const int source_label_num = applied_result.source_label_ptr->label_frame_num;
                const int merged_label_num = applied_result.merged_label_ptr->label_frame_num;
                TrajectoryLabel::merge_neighbor_label(applied_result.source_label_ptr, applied_result.merged_label_ptr);
                if (applied_result.current_keyframe_ptr) {
                    applied_result.current_keyframe_ptr->trajectory_label_ptr = applied_result.merged_label_ptr;
                }
                int curr_label_before = -1;
                int curr_label_after = -1;
                bool switched_curr_label = false;
                std::lock_guard<std::mutex> label_lock(trajectory_label_lock);
                curr_label_before = curr_trajectory_label_ptr ? curr_trajectory_label_ptr->label_frame_num : -1;
                if (curr_trajectory_label_ptr &&
                    TrajectoryLabel::is_connected(curr_trajectory_label_ptr, applied_result.source_label_ptr)) {
                    curr_trajectory_label_ptr = applied_result.merged_label_ptr;
                    current_label_frame_num = applied_result.merged_label_ptr->label_frame_num;
                    switched_curr_label = true;
                }
                curr_label_after = curr_trajectory_label_ptr ? curr_trajectory_label_ptr->label_frame_num : -1;
            }
        } else {
            mark_strict_stage3_done(applied_result.gen);
        }
    }
    if (strict_result_processed) {
        lc_state.store(static_cast<int>(LCState::IDLE));
    }
}

template <typename BasicType>
void CurlTracking<BasicType>::update_sph_coeff_thread(
    const std::vector<std::tuple<Eigen::MatrixX<BasicType>, std::shared_ptr<PatchInfo<BasicType>>, double>>
        &succeed_associations,
    const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &T_w_lidar) {
    if (debug_config_ptr->is_evaluate_time) {
        update_sph_coeff_timer.start();
    }
#pragma omp parallel for num_threads(curl_tracking_config_ptr->map_update_thread_num) default(none)                    \
    shared(succeed_associations, T_w_lidar, curl_voxel_mapping_ptr)
    for (const auto &association : succeed_associations) {
        std::unique_lock<std::mutex> lock(std::get<1>(association)->patch_update_lock);
        std::get<1>(association)->data_asso_counter++;
        bool is_proj_axis_changed;
        if (!SH_table_config_ptr->is_SH_analytic_jacobian) {
            is_proj_axis_changed =
                std::get<1>(association)
                    ->update_patch_projection_plane(std::get<0>(association), T_w_j.cast<BasicType>(),
                                                    curl_voxel_mapping_ptr->curl_voxel_mapping_config_ptr
                                                        ->minimum_points_to_fix_projection_direction);
        } else {
            is_proj_axis_changed =
                std::get<1>(association)
                    ->update_patch_projection_plane_low_RAM(std::get<0>(association), T_w_j.cast<BasicType>(),
                                                            curl_voxel_mapping_ptr->curl_voxel_mapping_config_ptr
                                                                ->minimum_points_to_fix_projection_direction);
        }
        Eigen::Matrix<BasicType, 4, 4, Eigen::RowMajor> T_obj_j =
            (std::get<1>(association)->T_obj_lidar.matrix() *
             Eigen::Isometry3d(std::get<1>(association)->keyframe_ptr->get_T_w_lidar()).inverse().matrix() * T_w_j)
                .template cast<BasicType>();
        bool is_recons_updated = false;
        if (!SH_table_config_ptr->is_SH_analytic_jacobian) {
            is_recons_updated =
                std::get<1>(association)
                    ->patch_procession_ptr->update_sph_residuals_without_conformal_mapping(
                        (T_obj_j(Eigen::seq(0, 2), Eigen::seq(0, 2)) * std::get<0>(association)).colwise() +
                            T_obj_j(Eigen::seq(0, 2), 3),
                        std::get<0>(association).colwise().squaredNorm(), T_w_lidar(Eigen::seq(0, 2), 3));
        } else {
            is_recons_updated =
                std::get<1>(association)
                    ->patch_procession_ptr->update_sph_residuals_without_conformal_mapping_low_RAM(
                        (T_obj_j(Eigen::seq(0, 2), Eigen::seq(0, 2)) * std::get<0>(association)).colwise() +
                            T_obj_j(Eigen::seq(0, 2), 3),
                        T_w_lidar(Eigen::seq(0, 2), 3));
        }

        if (is_recons_updated) {
            const int current_frame_idx = frame_idx_vec.back();
            const auto latest_keyframe = curl_voxel_mapping_ptr->spatial_hashing_ptr->latest_keyframe();
            const int lc_state_value = lc_state.load();
            const bool is_strict_lc_active = strict_lc_running.load() ||
                                             lc_state_value == static_cast<int>(LCState::RUNNING) ||
                                             lc_state_value == static_cast<int>(LCState::APPLYING);
            if (!is_strict_lc_active && latest_keyframe &&
                std::abs(latest_keyframe->frame_idx - std::get<1>(association)->last_update_keyframe_idx) < 200) {
                std::get<1>(association)->last_update_keyframe_num = latest_keyframe->frame_num;
                std::get<1>(association)->last_update_keyframe_idx = current_frame_idx;
                curl_voxel_mapping_ptr->spatial_hashing_ptr->update_patch_keyframe(std::get<1>(association));
            }
            if (latest_keyframe) {
                std::get<1>(association)->checkUpdate_last_update_keyframe_num = latest_keyframe->frame_num;
            }
            if (debug_config_ptr->is_pub_dense_reconstruction) {
                curl_voxel_mapping_ptr->rt_publish_landmark_patch_recons(std::get<1>(association));
            }
        }
        if (is_proj_axis_changed) { // publish reconstruction
            if (std::get<1>(association)->color[0] + curl_tracking_config_ptr->color_change_step < 1) {
                std::get<1>(association)->color[0] += curl_tracking_config_ptr->color_change_step;
            }
            if (std::get<1>(association)->color[1] + curl_tracking_config_ptr->color_change_step < 1) {
                std::get<1>(association)->color[1] += curl_tracking_config_ptr->color_change_step;
            }
            if (std::get<1>(association)->color[2] + curl_tracking_config_ptr->color_change_step < 1) {
                std::get<1>(association)->color[2] += curl_tracking_config_ptr->color_change_step;
            }
            if (debug_config_ptr->is_pub_dense_reconstruction) {
                curl_voxel_mapping_ptr->rt_publish_landmark_patch_recons(std::get<1>(association));
            }
        }
    }

    if (debug_config_ptr->is_evaluate_time) {
        update_sph_time_vec.push_back(update_sph_coeff_timer.elapsedMilliseconds());
    }
}

template <typename BasicType>
void CurlTracking<BasicType>::update_sph_coeff_thread(
    const std::vector<std::tuple<Eigen::MatrixX<BasicType>, std::shared_ptr<PatchInfo<BasicType>>, double>>
        &succeed_associations,
    const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &T_w_lidar,
    const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe_ptr, bool _is_add_trajectory_segment) {
    if (debug_config_ptr->is_evaluate_time) {
        update_sph_coeff_timer.start();
    }
#pragma omp parallel for num_threads(curl_tracking_config_ptr->map_update_thread_num) default(none)                    \
    shared(succeed_associations, T_w_lidar, curl_voxel_mapping_ptr, keyframe_ptr, _is_add_trajectory_segment)
    for (const auto &association : succeed_associations) {
        std::unique_lock<std::mutex> lock(std::get<1>(association)->patch_update_lock);
        PatchId patch_id = std::get<1>(association)->key;
        // keyframe_ptr->associated_patches.insert(patch_id);
        keyframe_ptr->associated_patches_insert(patch_id);
        if (_is_add_trajectory_segment) {
            // if (std::get<1>(association)->label_frame_num == -1) {
            std::get<1>(association)->label_frame_num_set.insert(keyframe_ptr->frame_num);
            // }
        }
        std::get<1>(association)->data_asso_counter++;
        bool is_proj_axis_changed;
        if (!SH_table_config_ptr->is_SH_analytic_jacobian) {
            is_proj_axis_changed =
                std::get<1>(association)
                    ->update_patch_projection_plane(std::get<0>(association), T_w_j.cast<BasicType>(),
                                                    curl_voxel_mapping_ptr->curl_voxel_mapping_config_ptr
                                                        ->minimum_points_to_fix_projection_direction);
        } else {
            is_proj_axis_changed =
                std::get<1>(association)
                    ->update_patch_projection_plane_low_RAM(std::get<0>(association), T_w_j.cast<BasicType>(),
                                                            curl_voxel_mapping_ptr->curl_voxel_mapping_config_ptr
                                                                ->minimum_points_to_fix_projection_direction);
        }

        Eigen::Matrix<BasicType, 4, 4, Eigen::RowMajor> T_obj_j =
            (std::get<1>(association)->T_obj_lidar.matrix() *
             Eigen::Isometry3d(std::get<1>(association)->keyframe_ptr->get_T_w_lidar()).inverse().matrix() * T_w_j)
                .template cast<BasicType>();
        bool is_recons_updated = false;
        if (!SH_table_config_ptr->is_SH_analytic_jacobian) {
            is_recons_updated =
                std::get<1>(association)
                    ->patch_procession_ptr->update_sph_residuals_without_conformal_mapping(
                        (T_obj_j(Eigen::seq(0, 2), Eigen::seq(0, 2)) * std::get<0>(association)).colwise() +
                            T_obj_j(Eigen::seq(0, 2), 3),
                        std::get<0>(association).colwise().squaredNorm(), T_w_lidar(Eigen::seq(0, 2), 3));
        } else {
            is_recons_updated =
                std::get<1>(association)
                    ->patch_procession_ptr->update_sph_residuals_without_conformal_mapping_low_RAM(
                        (T_obj_j(Eigen::seq(0, 2), Eigen::seq(0, 2)) * std::get<0>(association)).colwise() +
                            T_obj_j(Eigen::seq(0, 2), 3),
                        T_w_lidar(Eigen::seq(0, 2), 3));
        }
        if (is_recons_updated) {
            const int current_frame_idx = frame_idx_vec.back();
            const auto latest_keyframe = curl_voxel_mapping_ptr->spatial_hashing_ptr->latest_keyframe();
            const int lc_state_value = lc_state.load();
            const bool is_strict_lc_active = strict_lc_running.load() ||
                                             lc_state_value == static_cast<int>(LCState::RUNNING) ||
                                             lc_state_value == static_cast<int>(LCState::APPLYING);
            if (!is_strict_lc_active && latest_keyframe &&
                std::abs(latest_keyframe->frame_idx - std::get<1>(association)->last_update_keyframe_idx) < 200) {
                std::get<1>(association)->last_update_keyframe_num = latest_keyframe->frame_num;
                std::get<1>(association)->last_update_keyframe_idx = current_frame_idx;
                curl_voxel_mapping_ptr->spatial_hashing_ptr->update_patch_keyframe(std::get<1>(association));
            }
            if (latest_keyframe) {
                std::get<1>(association)->checkUpdate_last_update_keyframe_num = latest_keyframe->frame_num;
            }
            if (debug_config_ptr->is_pub_dense_reconstruction) {
                curl_voxel_mapping_ptr->rt_publish_landmark_patch_recons(std::get<1>(association));
            }
        }
        // publish reconstruction
        if (is_proj_axis_changed) { // publish reconstruction
            if (std::get<1>(association)->color[0] + curl_tracking_config_ptr->color_change_step < 1) {
                std::get<1>(association)->color[0] += curl_tracking_config_ptr->color_change_step;
            }
            if (std::get<1>(association)->color[1] + curl_tracking_config_ptr->color_change_step < 1) {
                std::get<1>(association)->color[1] += curl_tracking_config_ptr->color_change_step;
            }
            if (std::get<1>(association)->color[2] + curl_tracking_config_ptr->color_change_step < 1) {
                std::get<1>(association)->color[2] += curl_tracking_config_ptr->color_change_step;
            }
            if (debug_config_ptr->is_pub_dense_reconstruction) {
                curl_voxel_mapping_ptr->rt_publish_landmark_patch_recons(std::get<1>(association));
            }
        }
    }
    if (debug_config_ptr->is_evaluate_time) {
        update_sph_time_vec.push_back(update_sph_coeff_timer.elapsedMilliseconds());
    }
}

template <typename BasicType>
bool CurlTracking<BasicType>::add_landmark_thread(bool is_new_keyframe, bool _is_add_trajectory_segment,
                                                  const std::vector<Eigen::MatrixX<BasicType>> &point_cloud_lidar_vec,
                                                  const std::vector<bool> &is_ground_cloud_vec, const double time,
                                                  const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &T_w_lidar,
                                                  const Eigen::Isometry3d &_T_lastKeyframe_keyframe, int _frame_idx,
                                                  const pcl::PointCloud<PointT>::Ptr &seg_cloud_ptr,
                                                  const pcl::PointCloud<PointT>::Ptr &ground_cloud_ptr,
                                                  std::shared_ptr<TrajectoryLabel> &_history_trajectory_label_ptr) {
    if (debug_config_ptr->is_evaluate_time) {
        keyframe_timer.start();
    }
    bool is_succeed = curl_voxel_mapping_ptr->rt_fix_voxel_initialization(
        is_new_keyframe, _is_add_trajectory_segment, point_cloud_lidar_vec, is_ground_cloud_vec, time, T_w_lidar,
        _T_lastKeyframe_keyframe, true, _frame_idx, seg_cloud_ptr, ground_cloud_ptr, _history_trajectory_label_ptr);
    if (debug_config_ptr->is_evaluate_time) {
        landmark_initialization_time_vec.push_back(keyframe_timer.elapsedMilliseconds());
    }

    return is_succeed;
}

// 4. make the return type as the pair, which can detect whether this can be added as middle map
template <typename BasicType>
typename CurlTracking<BasicType>::BOXSTATUS
CurlTracking<BasicType>::new_landmark_add_judge(std::shared_ptr<PatchInfo<BasicType>> patch_info_ptr, const double &IoU,
                                                const std::vector<double> &query_lower_bound,
                                                const std::vector<double> &query_upper_bound) {
    std::vector<double> map_lower_bound;
    std::vector<double> map_upper_bound;

    map_lower_bound = patch_info_ptr->get_lower_bound_w();
    map_upper_bound = patch_info_ptr->get_upper_bound_w();

    bool is_query_box_in_map_box =
        query_lower_bound[0] >= map_lower_bound[0] && query_lower_bound[1] >= map_lower_bound[1] &&
        query_lower_bound[2] >= map_lower_bound[2] && query_upper_bound[0] <= map_upper_bound[0] &&
        query_upper_bound[1] <= map_upper_bound[1] && query_upper_bound[2] <= map_upper_bound[2];
    bool is_map_box_in_query_box =
        query_lower_bound[0] <= map_lower_bound[0] && query_lower_bound[1] <= map_lower_bound[1] &&
        query_lower_bound[2] <= map_lower_bound[2] && query_upper_bound[0] >= map_upper_bound[0] &&
        query_upper_bound[1] >= map_upper_bound[1] && query_upper_bound[2] >= map_upper_bound[2];
    if (is_query_box_in_map_box || is_map_box_in_query_box) {
        return INSIDE; // if the newer box is inside of the map box
    } else {
        // calculate the IoU
        double IoU_value = curl::IoU_3d(map_lower_bound, map_upper_bound, query_lower_bound, query_upper_bound);
        // decide whether this should be initlize as a new key frame
        bool is_add_new_landmark;
        if (IoU_value < curl_voxel_mapping_ptr->curl_voxel_mapping_config_ptr->highest_IoU_new_landmark_thres) {
            return ADD;
        } else {
            return ABANDON;
        }
    }
}

template <typename BasicType>
void CurlTracking<BasicType>::update_and_publish_trajectory(ros::Time current_time, const char *reason, uint64_t gen) {
    trajectory.poses.clear();
    const auto keyframes = curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframes_snapshot();
    trajectory.poses.reserve(keyframes.size());
    bool is_first = true;
    std::shared_ptr<KeyframeInfo<BasicType>> last_keyframe_ptr;
    for (const auto &keyframe : keyframes) {
        geometry_msgs::PoseStamped pose;
        pose.header.frame_id = "map";
        pose.header.stamp = current_time;
        pose.pose.position.x = keyframe->get_graph_pose_w_lidar().p.x();
        pose.pose.position.y = keyframe->get_graph_pose_w_lidar().p.y();
        pose.pose.position.z = keyframe->get_graph_pose_w_lidar().p.z();
        pose.pose.orientation.x = keyframe->get_graph_pose_w_lidar().q.x();
        pose.pose.orientation.y = keyframe->get_graph_pose_w_lidar().q.y();
        pose.pose.orientation.z = keyframe->get_graph_pose_w_lidar().q.z();
        pose.pose.orientation.w = keyframe->get_graph_pose_w_lidar().q.w();
        trajectory.poses.push_back(pose);
        if (is_first) {
            last_keyframe_ptr = keyframe;
            is_first = false;
            trajectory_segment_marker = curl_voxel_mapping_ptr->set_default_marker(
                "map", "trajectory_segment", keyframe->trajectory_label_ptr->label_frame_num,
                keyframe->trajectory_label_ptr->color, visualization_msgs::Marker::LINE_LIST,
                trajectory_segment_marker_line_scale);
        } else {
            if (last_keyframe_ptr->trajectory_label_ptr->label_frame_num !=
                keyframe->trajectory_label_ptr->label_frame_num) {
                trajectory_segment_marker = curl_voxel_mapping_ptr->set_default_marker(
                    "map", "trajectory_segment", keyframe->trajectory_label_ptr->label_frame_num,
                    keyframe->trajectory_label_ptr->color, visualization_msgs::Marker::LINE_LIST,
                    trajectory_segment_marker_line_scale);
            }
            publish_labelled_trajectory(last_keyframe_ptr->get_graph_pose_w_lidar_T().matrix(),
                                        keyframe->get_graph_pose_w_lidar_T().matrix(), current_time);
            last_keyframe_ptr = keyframe;
        }
    }
    pub_trajectory.publish(trajectory);
}

template <typename BasicType>
void CurlTracking<BasicType>::publish_trajectory(const ros::Publisher &publisher,
                                                 const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_w_j,
                                                 nav_msgs::Path &path, ros::Time current_time) {
    // publish the trajectory
    geometry_msgs::PoseStamped pose;
    pose.header.frame_id = "map";
    pose.header.stamp = current_time;
    pose.pose.position.x = _T_w_j(0, 3);
    pose.pose.position.y = _T_w_j(1, 3);
    pose.pose.position.z = _T_w_j(2, 3);
    Eigen::Quaterniond quat = Eigen::Quaterniond(Eigen::Matrix3d(_T_w_j(Eigen::seq(0, 2), Eigen::seq(0, 2))));
    pose.pose.orientation.x = quat.x(); // the x component of your quaternion
    pose.pose.orientation.y = quat.y(); // the y component of your quaternion
    pose.pose.orientation.z = quat.z(); // the z component of your quaternion
    pose.pose.orientation.w = quat.w(); // the w component of your quaternion
    path.poses.push_back(pose);
    path.header.stamp = pose.header.stamp;
    publisher.publish(path);
}

template <typename BasicType>
void CurlTracking<BasicType>::publish_labelled_trajectory(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_w_j,
                                                          const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_w_j_1,
                                                          ros::Time current_time) {
    // publish the trajectory
    geometry_msgs::Point pt_start;
    pt_start.x = _T_w_j_1(0, 3);
    pt_start.y = _T_w_j_1(1, 3);
    pt_start.z = _T_w_j_1(2, 3);
    geometry_msgs::Point pt_end;
    pt_end.x = _T_w_j(0, 3);
    pt_end.y = _T_w_j(1, 3);
    pt_end.z = _T_w_j(2, 3);
    trajectory_segment_marker.points.push_back(pt_start);
    trajectory_segment_marker.points.push_back(pt_end);
    trajectory_segment_marker.header.stamp = current_time;
    pub_labelled_trajectory.publish(trajectory_segment_marker);
}

template <typename BasicType> void CurlTracking<BasicType>::publish_lidar_tf(ros::Time current_time) {
    transformStamped.header.frame_id = "map";
    transformStamped.child_frame_id = "lidar";
    transformStamped.transform.translation.x = T_w_j(0, 3);
    transformStamped.transform.translation.y = T_w_j(1, 3);
    transformStamped.transform.translation.z = T_w_j(2, 3);
    Eigen::Quaterniond quat = Eigen::Quaterniond(Eigen::Matrix3d(T_w_j(Eigen::seq(0, 2), Eigen::seq(0, 2))));
    transformStamped.transform.rotation.x = quat.x(); // the x component of your quaternion
    transformStamped.transform.rotation.y = quat.y(); // the y component of your quaternion
    transformStamped.transform.rotation.z = quat.z(); // the z component of your quaternion
    transformStamped.transform.rotation.w = quat.w(); // the w component of your quaternion
    transformStamped.header.stamp = current_time;
    tfb.sendTransform(transformStamped);
}

template <typename BasicType>
void CurlTracking<BasicType>::publish_point_cloud(
    ros::Time current_time,
    const std::vector<std::tuple<Eigen::MatrixX<BasicType>, std::shared_ptr<PatchInfo<BasicType>>, double>>
        &succeed_associations) {
    pcl::PointXYZI point;
    pcl::PointCloud<pcl::PointXYZI> used_cloud;
    for (const auto &succeed_asso : succeed_associations) {
        point.intensity = 0.0f;
        for (int j = 0; j < std::get<0>(succeed_asso).cols(); ++j) {
            point.x = std::get<0>(succeed_asso)(0, j);
            point.y = std::get<0>(succeed_asso)(1, j);
            point.z = std::get<0>(succeed_asso)(2, j);
            used_cloud.push_back(point);
        }
    }
    sensor_msgs::PointCloud2 used_cloud_msg;
    pcl::toROSMsg(used_cloud, used_cloud_msg);
    used_cloud_msg.header.frame_id = "lidar";
    used_cloud_msg.header.stamp = current_time;
    pub_used_cloud.publish(used_cloud_msg);
}

template <typename BasicType>
void CurlTracking<BasicType>::publish_query_map_bounding_box(
    ros::Time current_time,
    const std::vector<std::pair<std::vector<double>, std::vector<double>>> &query_bounding_box_vec,
    const std::vector<std::pair<std::vector<double>, std::vector<double>>> &map_bounding_box_vec) {
    float line_scale[3] = {0.05, 0, 0};
    std::array<float, 3> red_color{1, 0, 0};
    std::array<float, 3> green_color{0, 1, 0};
    visualization_msgs::Marker query_bounding_box = curl_voxel_mapping_ptr->set_default_marker(
        "map", "query_bounding_box", 0, red_color, visualization_msgs::Marker::LINE_LIST, line_scale);
    visualization_msgs::Marker map_bounding_box = curl_voxel_mapping_ptr->set_default_marker(
        "map", "map_bounding_box", 0, green_color, visualization_msgs::Marker::LINE_LIST, line_scale);
    for (auto &box_pair : query_bounding_box_vec) {
        Eigen::MatrixX<double> box_points = curl::generate_box_points<double>(box_pair.first, box_pair.second);
        for (int i = 0; i < box_points.cols(); ++i) {
            geometry_msgs::Point pt;
            pt.x = box_points(0, i);
            pt.y = box_points(1, i);
            pt.z = box_points(2, i);
            query_bounding_box.points.push_back(pt);
        }
    }
    for (auto &box_pair : map_bounding_box_vec) {
        Eigen::MatrixX<double> box_points = curl::generate_box_points<double>(box_pair.first, box_pair.second);
        for (int i = 0; i < box_points.cols(); ++i) {
            geometry_msgs::Point pt;
            pt.x = box_points(0, i);
            pt.y = box_points(1, i);
            pt.z = box_points(2, i);
            map_bounding_box.points.push_back(pt);
        }
    }
    pub_bounding_box.publish(query_bounding_box);
    pub_bounding_box.publish(map_bounding_box);
}

template <typename BasicType> void CurlTracking<BasicType>::log() {
    if (debug_config_ptr->is_evaluate_time) {
        std::string times_folder = debug_config_ptr->results_dir + "/times";
        curl::create_directory_if_not_exists(times_folder);
        std::string data_association_time_file = times_folder + "/data_association_time.txt";
        FileReaderBase::write_vector_txt_file(data_association_time_file, data_association_time_vec);
        std::string landmark_initialization_time_file = times_folder + "/landmark_initialization_time.txt";
        FileReaderBase::write_vector_txt_file(landmark_initialization_time_file, landmark_initialization_time_vec);
        std::string opt_time_file = times_folder + "/opt_time.txt";
        FileReaderBase::write_vector_txt_file(opt_time_file, opt_time_vec);
        std::string update_sph_time_file = times_folder + "/update_sph_time.txt";
        FileReaderBase::write_vector_txt_file(update_sph_time_file, update_sph_time_vec);
        std::string total_time_file = times_folder + "/total_time.txt";
        FileReaderBase::write_vector_txt_file(total_time_file, total_time_vec);
    }
    if (debug_config_ptr->is_save_final_results) {
        // Create result folder
        curl::create_directory_if_not_exists(debug_config_ptr->results_dir);
        // save odometry and trajectory
        std::cout << "Start saving odometry and trajectory" << std::endl;
        std::string odometry_file = debug_config_ptr->results_dir + "/odometry.txt";

        std::ofstream odom_outfile(odometry_file);
        Eigen::Isometry3d T_odom_w_j = Eigen::Isometry3d::Identity();
        odom_outfile << frame_idx_vec[0] << " " << T_odom_w_j.matrix()(0, 0) << " " << T_odom_w_j.matrix()(0, 1) << " "
                     << T_odom_w_j.matrix()(0, 2) << " " << T_odom_w_j.matrix()(0, 3) << " "
                     << T_odom_w_j.matrix()(1, 0) << " " << T_odom_w_j.matrix()(1, 1) << " "
                     << T_odom_w_j.matrix()(1, 2) << " " << T_odom_w_j.matrix()(1, 3) << " "
                     << T_odom_w_j.matrix()(2, 0) << " " << T_odom_w_j.matrix()(2, 1) << " "
                     << T_odom_w_j.matrix()(2, 2) << " " << T_odom_w_j.matrix()(2, 3) << " "
                     << T_odom_w_j.matrix()(3, 0) << " " << T_odom_w_j.matrix()(3, 1) << " "
                     << T_odom_w_j.matrix()(3, 2) << " " << T_odom_w_j.matrix()(3, 3) << std::endl;
        for (int idx = 1; idx < frame_idx_vec.size(); ++idx) {
            T_odom_w_j = T_odom_w_j * odometry_vector[idx - 1];
            odom_outfile << frame_idx_vec[idx] << " " << T_odom_w_j.matrix()(0, 0) << " " << T_odom_w_j.matrix()(0, 1)
                         << " " << T_odom_w_j.matrix()(0, 2) << " " << T_odom_w_j.matrix()(0, 3) << " "
                         << T_odom_w_j.matrix()(1, 0) << " " << T_odom_w_j.matrix()(1, 1) << " "
                         << T_odom_w_j.matrix()(1, 2) << " " << T_odom_w_j.matrix()(1, 3) << " "
                         << T_odom_w_j.matrix()(2, 0) << " " << T_odom_w_j.matrix()(2, 1) << " "
                         << T_odom_w_j.matrix()(2, 2) << " " << T_odom_w_j.matrix()(2, 3) << " "
                         << T_odom_w_j.matrix()(3, 0) << " " << T_odom_w_j.matrix()(3, 1) << " "
                         << T_odom_w_j.matrix()(3, 2) << " " << T_odom_w_j.matrix()(3, 3) << std::endl;
        }
        odom_outfile.close();
        std::cout << "Done!" << std::endl;
        std::cout << "Saving Optimized Trajectory" << std::endl;
        curl_voxel_mapping_ptr->spatial_hashing_ptr->save_keyframe_trajectory(debug_config_ptr->results_dir);
        curl_voxel_mapping_ptr->spatial_hashing_ptr->save_optimized_trajectories(debug_config_ptr->results_dir);
        std::cout << "Done!" << std::endl;
        size_t seg_num = 0;
        size_t ground_num = 0;
        size_t mask_size = 0;
        const auto keyframes = curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframes_snapshot();
        for (const auto &keyframe : keyframes) {
            for (const auto &patch_id : keyframe->local_patches) {
                if (auto patch_info_ptr = curl_voxel_mapping_ptr->spatial_hashing_ptr->get_patch_by_id(patch_id)) {
                    if (patch_info_ptr->patch_procession_ptr->get_sph_coeff_sum() == 0) {
                        continue;
                    }
                    if (patch_info_ptr->is_ground) {
                        ++ground_num;
                    } else {
                        ++seg_num;
                    }
                    mask_size += patch_info_ptr->patch_procession_ptr->get_mask_idx().size();
                }
            }
        }
        std::cout << "Summary seg_num: " << seg_num << " ground_num: " << ground_num << " mask number: " << mask_size
                  << std::endl;
        std::cout << "Map Size: "
                  << (seg_num * std::pow((SH_table_config_ptr->max_SH_degree + 1), 2) +
                      ground_num * std::pow((SH_table_config_ptr->ground_SH_degree + 1), 2)) *
                             64 / 8 / 1024 / 1024 +
                         mask_size * 32 / 8 / 1024 / 1024
                  << " MB" << std::endl;

        std::cout << "Ideal Map Size with binary mask: "
                  << (seg_num * std::pow((SH_table_config_ptr->max_SH_degree + 1), 2) +
                      ground_num * std::pow((SH_table_config_ptr->ground_SH_degree + 1), 2)) *
                             64 / 8 / 1024 / 1024 +
                         (seg_num + ground_num) *
                             std::pow(curl_voxel_mapping_ptr->curl_voxel_mapping_config_ptr->cut_threshold / 0.05, 2) /
                             8 / 1024 / 1024
                  << " MB" << std::endl;
        if (debug_config_ptr->is_save_final_map) {
            // save the config file
            std::cout << "Start Saving Config File" << std::endl;
            std::filesystem::copy_file("/home/user/catkin_ws/src/CURL-SLAM/config/config.yaml",
                                       debug_config_ptr->results_dir + "/config.yaml",
                                       std::filesystem::copy_options::overwrite_existing);
            std::cout << "End Saving Config File" << std::endl;
            // end
            std::string final_map_dir = debug_config_ptr->results_dir + "/final_map";
            curl::create_directory_if_not_exists(final_map_dir);
            curl_voxel_mapping_ptr->spatial_hashing_ptr->all_saving_patches();
            // save binary map
            std::cout << "Start Saving Binary Map" << std::endl;
            {
                std::ofstream debug_binary_file(final_map_dir + "/curl_map.bin", std::ios::binary | std::ios::out);
                for (const auto &patch_info_ptr : curl_voxel_mapping_ptr->spatial_hashing_ptr->all_saving_patches_ptr) {
                    // save is_ground
                    debug_binary_file.write(reinterpret_cast<const char *>(&patch_info_ptr->is_ground), sizeof(bool));
                    Eigen::MatrixXf T_w_obj =
                        (patch_info_ptr->keyframe_ptr->get_T_w_lidar() *
                         patch_info_ptr->T_obj_lidar.inverse().matrix())(Eigen::seq(0, 2), Eigen::all)
                            .template cast<float>();
                    // save T_w_obj
                    debug_binary_file.write(reinterpret_cast<const char *>(T_w_obj.data()),
                                            T_w_obj.size() * sizeof(float));
                    // save sph_coeff
                    debug_binary_file.write(
                        reinterpret_cast<const char *>(patch_info_ptr->patch_procession_ptr->get_sph_coeff().data()),
                        patch_info_ptr->patch_procession_ptr->get_sph_coeff().size() * sizeof(double));
                    // save binary_mask
                    Eigen::VectorXi binary_mask = patch_info_ptr->patch_procession_ptr->get_binary_mask();
                    int loop_times = std::ceil(binary_mask.size() / 64.0);
                    for (int loop = 0; loop < loop_times; ++loop) {
                        std::bitset<64> mask_bits;
                        for (int i = 0; i < 64; ++i) {
                            int mask_idx = loop * 64 + i;
                            if (mask_idx == binary_mask.size()) {
                                break;
                            }
                            mask_bits[63 - i] = binary_mask(mask_idx);
                        }
                        debug_binary_file.write(reinterpret_cast<const char *>(&mask_bits), sizeof(std::bitset<64>));
                    }
                }
                debug_binary_file.close();
            }
            std::cout << "End Saving Binary Map" << std::endl;

            // save serialization files

            // std::cout << "Start Serialization" << std::endl;
            // {
            //     std::ofstream debug_serialization_file(final_map_dir + "/serialization_map.txt");
            //     boost::archive::text_oarchive oa(debug_serialization_file);
            //     // write class instance to archive
            //     oa << curl_voxel_mapping_ptr;
            // }
            // std::cout << "End Serialization" << std::endl;
            // save reconstructed points
            std::cout << "Start Saving Reconstructed Points" << std::endl;
            pcl::PointCloud<PointT> map_cloud;
            int cloud_empty_counter = 0;
            int cloud_full_counter = 0;
            const auto keyframes = curl_voxel_mapping_ptr->spatial_hashing_ptr->keyframes_snapshot();
            for (const auto &keyframe : keyframes) {
                if (keyframe->seg_cloud_ptr != nullptr || keyframe->ground_cloud_ptr != nullptr) {
                    cloud_full_counter++;
                } else {
                    cloud_empty_counter++;
                }
                for (const auto &patch_id : keyframe->local_patches) {
                    if (auto patch_info_ptr = curl_voxel_mapping_ptr->spatial_hashing_ptr->get_patch_by_id(patch_id)) {
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
                        for (int idx = 0; idx < recons_v_world.cols(); ++idx) {
                            PointT pt;
                            pt.x = recons_v_world(0, idx);
                            pt.y = recons_v_world(1, idx);
                            pt.z = recons_v_world(2, idx);
                            pt.intensity = 0.0f;
                            pt.t = 0;
                            pt.label = 0;
                            map_cloud.push_back(pt);
                        }
                    }
                }
            }
            map_cloud.width = map_cloud.size();
            map_cloud.height = 1;
            map_cloud.is_dense = false;
            pcl::io::savePCDFileBinary(final_map_dir + "/map.pcd", map_cloud);
            std::cout << "End Saving Reconstructed Points" << std::endl;
            std::cout << "Cloud full counter: " << cloud_full_counter << std::endl;
            std::cout << "Cloud empty counter: " << cloud_empty_counter << std::endl;
        }
    }
}

template class CurlTracking<BT>; // this is very important
