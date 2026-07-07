#include "curl_slam/CurlVoxelMapping.h"
#include <limits>

template <typename BasicType>
CurlVoxelMapping<BasicType>::CurlVoxelMapping(ros::NodeHandle *nh,
                                              std::shared_ptr<CURL_VOXEL_MAPPING_CONFIG> _curl_voxel_mapping_config_ptr,
                                              std::shared_ptr<CURL_TRACKING_CONFIG> _curl_tracking_config_ptr,
                                              std::shared_ptr<SH_TABLE_CONFIG<BasicType>> _SH_table_config_ptr,
                                              std::shared_ptr<DIRECT_METHOD_CONFIG> _direct_method_config_ptr,
                                              std::shared_ptr<AABB_CONFIG> _aabb_config_ptr,
                                              std::shared_ptr<CURL_LOOP_CLOSURE_CONFIG> _curl_loop_closure_config_ptr,
                                              std::shared_ptr<DEBUG_CONFIG> _debug_config_ptr)
    : curl_voxel_mapping_config_ptr(_curl_voxel_mapping_config_ptr),
      curl_tracking_config_ptr(_curl_tracking_config_ptr), SH_table_config_ptr(_SH_table_config_ptr),
      direct_method_config_ptr(_direct_method_config_ptr), aabb_config_ptr(_aabb_config_ptr),
      curl_loop_closure_config_ptr(_curl_loop_closure_config_ptr), debug_config_ptr(_debug_config_ptr),
      rt_non_ground_pt_sub(*nh, "/patchworkpp/nonground_pc", 1000),
      rt_ground_pt_sub(*nh, "/patchworkpp/ground_pc", 1000) {
    frame_counter = 0;
    robust_initialization_counter = 0;
    pub_landmark_patch = nh->advertise<visualization_msgs::Marker>("/curl/landmark_patch", 1000);
    segment_pt_sub = nh->subscribe("/curl/segmented_pc", 1000,
                                   &CurlVoxelMapping<BasicType>::rt_callback_point_cloud_segments_seg_only, this);
    sync_ptr = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(MySyncPolicy(1000), rt_segment_pt_sub,
                                                                             rt_ground_pt_sub);
    non_ground_sync_ptr = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(
        MySyncPolicy(10), rt_non_ground_pt_sub, rt_ground_pt_sub);

    non_ground_sync_ptr->registerCallback(
        boost::bind(&CurlVoxelMapping<BasicType>::rt_callback_point_cloud_segments, this, _1, _2));

    spatial_hashing_ptr = std::make_shared<SpatialHashing<BasicType>>(curl_voxel_mapping_config_ptr->cut_threshold);
    omp_init_lock(&writelock);
    is_tracking_wait_BA = false;
    is_init = true;
    is_loop_closure_detected = false;
    is_strict_loop_closure_detected = false;
}

template <typename BasicType>
void CurlVoxelMapping<BasicType>::remove_points_inside_bounding_box(
    const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe_info_ptr,
    const Eigen::MatrixX<BasicType> &merged_seg_point_cloud_world,
    const Eigen::MatrixX<BasicType> &merged_ground_point_cloud_world,
    std::vector<std::pair<std::vector<int>, Eigen::MatrixX<BasicType>>> &seg_points_vec_world,
    std::vector<std::pair<std::vector<int>, Eigen::MatrixX<BasicType>>> &ground_points_vec_world) {
    std::unordered_map<std::vector<int>, std::vector<Eigen::Vector3<BasicType>>, VoxelHashFuncPrime> seg_grid_points =
        spatial_hashing_ptr->get_grid_indices(merged_seg_point_cloud_world);
    std::unordered_map<std::vector<int>, std::vector<Eigen::Vector3<BasicType>>, VoxelHashFuncPrime>
        ground_grid_points = spatial_hashing_ptr->get_grid_indices(merged_ground_point_cloud_world);
    // non-ground points
    std::vector<Eigen::Vector3<BasicType>> points_tmp;
    points_tmp.reserve(100);
    auto curr_trajectory_label_ptr = keyframe_info_ptr->trajectory_label_ptr;
    for (auto &grid_pair : seg_grid_points) {
        points_tmp.clear();
        auto the_grid = spatial_hashing_ptr->find(grid_pair.first);
        if (the_grid != spatial_hashing_ptr->end() && !the_grid->second.empty()) {
            for (const auto pt : grid_pair.second) {
                bool is_inside = false;
                for (const auto &patch_info_ptr : the_grid->second.patches) {
                    if (!patch_info_ptr->is_ground &&
                        (curl_voxel_mapping_config_ptr->is_use_all_associated_patches ||
                         TrajectoryLabel::is_connected(
                             curr_trajectory_label_ptr,
                             patch_info_ptr->keyframe_ptr->trajectory_label_ptr))) { // non-ground patch is valid
                        if (curl::is_3D_point_inside_box<BasicType>(patch_info_ptr->get_lower_bound_w(),
                                                                    patch_info_ptr->get_upper_bound_w(), pt)) {
                            is_inside = true;
                            break;
                        }
                    }
                }
                if (!is_inside) {
                    points_tmp.emplace_back(pt);
                }
            }
        } else {
            points_tmp = grid_pair.second;
        }
        if (points_tmp.size() > curl_voxel_mapping_config_ptr->minimum_points_to_add_new_patch_seg) {
            Eigen::MatrixX<BasicType> matrix(3, points_tmp.size());
            for (int i = 0; i < points_tmp.size(); ++i) {
                matrix(0, i) = points_tmp[i](0);
                matrix(1, i) = points_tmp[i](1);
                matrix(2, i) = points_tmp[i](2);
            }
            // seg_points_vec_world[grid_pair.first] = matrix;
            seg_points_vec_world.emplace_back(grid_pair.first, matrix);
        }
    }
    // ground points
    points_tmp.clear();
    for (auto &grid_pair : ground_grid_points) {
        points_tmp.clear();
        auto the_grid = spatial_hashing_ptr->find(grid_pair.first);
        if (the_grid != spatial_hashing_ptr->end() && !the_grid->second.empty()) {
            for (const auto pt : grid_pair.second) {
                bool is_inside = false;
                for (const auto &patch_info_ptr : the_grid->second.patches) {
                    if (patch_info_ptr->is_ground &&
                        (curl_voxel_mapping_config_ptr->is_use_all_associated_patches ||
                         TrajectoryLabel::is_connected(
                             curr_trajectory_label_ptr,
                             patch_info_ptr->keyframe_ptr->trajectory_label_ptr))) { // ground patch is valid
                        if (curl::is_3D_point_inside_box<BasicType>(patch_info_ptr->get_lower_bound_w(),
                                                                    patch_info_ptr->get_upper_bound_w(), pt)) {
                            is_inside = true;
                            break;
                        }
                    }
                }
                if (!is_inside) {
                    points_tmp.emplace_back(pt);
                }
            }
        } else {
            points_tmp.insert(points_tmp.end(), grid_pair.second.begin(), grid_pair.second.end());
        }
        if (points_tmp.size() > curl_voxel_mapping_config_ptr->minimum_points_to_add_new_patch_ground) {
            Eigen::MatrixX<BasicType> matrix(3, points_tmp.size());
            for (int i = 0; i < points_tmp.size(); ++i) {
                matrix(0, i) = points_tmp[i](0);
                matrix(1, i) = points_tmp[i](1);
                matrix(2, i) = points_tmp[i](2);
            }
            // ground_points_vec_world[grid_pair.first] = matrix;
            ground_points_vec_world.emplace_back(grid_pair.first, matrix);
        }
    }
    return;
}

template <typename BasicType>
void CurlVoxelMapping<BasicType>::re_clustering(
    const int frame_idx, const Eigen::MatrixX<BasicType> &merged_seg_point_cloud_world,
    const Eigen::MatrixX<BasicType> &merged_ground_point_cloud_world,
    std::unordered_map<std::vector<int>, Eigen::MatrixX<BasicType>, VoxelHashFuncPrime> &seg_points_vec_world,
    std::unordered_map<std::vector<int>, Eigen::MatrixX<BasicType>, VoxelHashFuncPrime> &ground_points_vec_world) {
    curl::divide_patches<BasicType>(
        merged_seg_point_cloud_world, seg_points_vec_world, curl_voxel_mapping_config_ptr->cut_threshold,
        curl_voxel_mapping_config_ptr->cut_threshold, curl_voxel_mapping_config_ptr->cut_threshold);
    curl::divide_patches<BasicType>(
        merged_ground_point_cloud_world, ground_points_vec_world, curl_voxel_mapping_config_ptr->cut_threshold,
        curl_voxel_mapping_config_ptr->cut_threshold, curl_voxel_mapping_config_ptr->cut_threshold);
}

template <typename BasicType>
bool CurlVoxelMapping<BasicType>::get_local_coordinate_with_eig(
    bool is_ground, const Eigen::Isometry3d &T_w_lidar, const Eigen::MatrixX<BasicType> &patch_cloud_w,
    const std::vector<double> &lower_bound_w, const std::vector<double> &upper_bound_w, Eigen::Isometry3d &T_obj_lidar,
    Eigen::Matrix4<BasicType> &cov_4, PROJECTION_AXIS &projection_axis) {
    Eigen::Isometry3d T_w_obj;
    T_w_obj.setIdentity();
    T_w_obj.matrix()(Eigen::seq(0, 2), 3) =
        Eigen::Vector3d((upper_bound_w[0] + lower_bound_w[0]) / 2, (upper_bound_w[1] + lower_bound_w[1]) / 2,
                        (upper_bound_w[2] + lower_bound_w[2]) / 2);
    Eigen::Matrix3d R_rot;
    Eigen::MatrixX<BasicType> homogeneous_pt(4, patch_cloud_w.cols());
    homogeneous_pt.setOnes();
    homogeneous_pt(Eigen::seq(0, 2), Eigen::all) = patch_cloud_w;
    cov_4 = homogeneous_pt * homogeneous_pt.transpose();
    // FileReaderBase::write_txt_file("/home/user/Extreme_2T/tmp/pt_w.txt", Eigen::MatrixXd(patch_cloud_w.transpose()));
    if (!is_ground) {
        Eigen::Matrix3<BasicType> cov =
            (cov_4(Eigen::seq(0, 2), Eigen::seq(0, 2)) - (1.0 / static_cast<double>(patch_cloud_w.cols())) *
                                                             cov_4(Eigen::seq(0, 2), 3) * cov_4(3, Eigen::seq(0, 2)))
                .array() /
            static_cast<double>(patch_cloud_w.cols());

        //        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3<BasicType>> es;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3<BasicType>> es;
        es.compute(cov);
        Eigen::Vector3<BasicType> shortest_axis = es.eigenvectors().col(0);
        BasicType dot_with_x = std::abs(shortest_axis.dot(Eigen::Vector3<BasicType>::UnitX()));
        BasicType dot_with_y = std::abs(shortest_axis.dot(Eigen::Vector3<BasicType>::UnitY()));
        BasicType dot_with_z = std::abs(shortest_axis.dot(Eigen::Vector3<BasicType>::UnitZ()));
        if (dot_with_x > dot_with_y && dot_with_x > dot_with_z) {
            R_rot = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitX()).toRotationMatrix() *
                    Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitY()).toRotationMatrix();
            projection_axis = PROJECTION_AXIS::X_AXIS;
        } else if (dot_with_y > dot_with_x && dot_with_y > dot_with_z) {
            R_rot = Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d::UnitX()).toRotationMatrix() *
                    Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
            projection_axis = PROJECTION_AXIS::Y_AXIS;
        } else {
            R_rot.setIdentity();
            projection_axis = PROJECTION_AXIS::Z_AXIS;
        }
    } else {
        R_rot.setIdentity();
    }
    T_w_obj.matrix()(Eigen::seq(0, 2), Eigen::seq(0, 2)) = R_rot;
    // FileReaderBase::write_txt_file("/home/user/Extreme_2T/tmp/T_w_obj.txt", Eigen::MatrixXd(T_w_obj.matrix()));
    T_obj_lidar = T_w_obj.inverse() * T_w_lidar;
    double dis = T_obj_lidar.translation().squaredNorm();
    if (dis > curl_voxel_mapping_config_ptr->largest_distance_squared &&
        curl_voxel_mapping_config_ptr->largest_distance_squared != -1) {
        return false;
    } else {
        return true;
    }
}

template <typename BasicType>
void CurlVoxelMapping<BasicType>::rt_callback_point_cloud_segments_seg_only(
    const sensor_msgs::PointCloud2::ConstPtr &segment_msg) {
    pcl::PointCloud<PointT>::Ptr seg_cloud(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr ground_cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*segment_msg, *seg_cloud);
    if (is_init) {
        if (seg_cloud->points[0].label != 0) {
            std::cerr << "First Label: " << seg_cloud->points[0].label << " is not 0!!!" << std::endl;
            ros::shutdown();
        }
        is_init = false;
    }
    if (curl_voxel_mapping_config_ptr->is_voxel_grid_filter &&
        robust_initialization_counter > direct_method_config_ptr->update_steps) {
        if (!seg_cloud->empty()) {
            curl::subSampleFrame(*seg_cloud, curl_voxel_mapping_config_ptr->leaf_size,
                                 static_cast<float>(curl_voxel_mapping_config_ptr->minimum_squared_dis),
                                 static_cast<float>(curl_voxel_mapping_config_ptr->maximum_squared_dis));
        }
        if (!ground_cloud->empty()) {
            curl::subSampleFrame(*ground_cloud, curl_voxel_mapping_config_ptr->leaf_size * 2,
                                 static_cast<float>(curl_voxel_mapping_config_ptr->minimum_squared_dis),
                                 static_cast<float>(curl_voxel_mapping_config_ptr->maximum_squared_dis));
        }
    } else {
        if (!seg_cloud->empty()) {
            curl::filterInvalidRangePoints(*seg_cloud,
                                           static_cast<float>(curl_voxel_mapping_config_ptr->minimum_squared_dis),
                                           static_cast<float>(curl_voxel_mapping_config_ptr->maximum_squared_dis));
        }
        if (!ground_cloud->empty()) {
            curl::filterInvalidRangePoints(*ground_cloud,
                                           static_cast<float>(curl_voxel_mapping_config_ptr->minimum_squared_dis),
                                           static_cast<float>(curl_voxel_mapping_config_ptr->maximum_squared_dis));
        }
        ++robust_initialization_counter;
    }
    preprocessing_queue_lock.lock();
    pcl_ptr_pair_queue.emplace(std::make_tuple(segment_msg->header.stamp.toSec(), seg_cloud, ground_cloud));
    preprocessing_queue_lock.unlock();
}

template <typename BasicType>
void CurlVoxelMapping<BasicType>::rt_callback_point_cloud_segments(
    const sensor_msgs::PointCloud2::ConstPtr &segment_msg, const sensor_msgs::PointCloud2::ConstPtr &ground_msg) {
    pcl::PointCloud<PointT>::Ptr seg_cloud(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr ground_cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*segment_msg, *seg_cloud);
    pcl::fromROSMsg(*ground_msg, *ground_cloud);
    if (curl_voxel_mapping_config_ptr->is_seg_only) {
        *seg_cloud += *ground_cloud;
        ground_cloud->clear();
    }
    if (debug_config_ptr->data_set == "KITTI" && curl_voxel_mapping_config_ptr->is_kitti_correct) {
        curl::intrinsic_correct<PointT>(*seg_cloud, 0.195);
        curl::intrinsic_correct<PointT>(*ground_cloud, 0.195);
    }
    if (is_init) {
        if (seg_cloud->points[0].label != 0) {
            std::cerr << "First Label: " << seg_cloud->points[0].label << " is not 0!!!" << std::endl;
            ros::shutdown();
        }
        is_init = false;
    }
    if (curl_voxel_mapping_config_ptr->is_voxel_grid_filter &&
        robust_initialization_counter > direct_method_config_ptr->update_steps) {
        if (!seg_cloud->empty()) {
            curl::subSampleFrame(*seg_cloud, curl_voxel_mapping_config_ptr->leaf_size,
                                 static_cast<float>(curl_voxel_mapping_config_ptr->minimum_squared_dis),
                                 static_cast<float>(curl_voxel_mapping_config_ptr->maximum_squared_dis));
        }
        if (!ground_cloud->empty()) {
            curl::subSampleFrame(*ground_cloud, curl_voxel_mapping_config_ptr->leaf_size * 2,
                                 static_cast<float>(curl_voxel_mapping_config_ptr->minimum_squared_dis),
                                 static_cast<float>(curl_voxel_mapping_config_ptr->maximum_squared_dis));
        }
    } else {
        if (!seg_cloud->empty()) {
            curl::filterInvalidRangePoints(*seg_cloud,
                                           static_cast<float>(curl_voxel_mapping_config_ptr->minimum_squared_dis),
                                           static_cast<float>(curl_voxel_mapping_config_ptr->maximum_squared_dis));
        }
        if (!ground_cloud->empty()) {
            curl::filterInvalidRangePoints(*ground_cloud,
                                           static_cast<float>(curl_voxel_mapping_config_ptr->minimum_squared_dis),
                                           static_cast<float>(curl_voxel_mapping_config_ptr->maximum_squared_dis));
        }
        ++robust_initialization_counter;
    }
    preprocessing_queue_lock.lock();
    pcl_ptr_pair_queue.emplace(std::make_tuple(segment_msg->header.stamp.toSec(), seg_cloud, ground_cloud));
    preprocessing_queue_lock.unlock();
}

// FIXME: change std::vector<std::vector<Eigen::MatrixX<BasicType>>> &patches_lidar_vec
// FIXME: remove this PointCloudInfo<BasicType> &_point_cloud_info into is_ground_vec and Time
template <typename BasicType>
bool CurlVoxelMapping<BasicType>::rt_fix_voxel_initialization(
    bool is_new_keyframe, bool is_add_trajectory_segment,
    const std::vector<Eigen::MatrixX<BasicType>> &patches_lidar_vec, const std::vector<bool> &is_ground_cloud_vec,
    const double time, const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &T_w_lidar_curr,
    const Eigen::Isometry3d &_T_lastKeyframe_keyframe, bool is_active, int frame_idx,
    const pcl::PointCloud<PointT>::Ptr &seg_cloud_ptr, const pcl::PointCloud<PointT>::Ptr &ground_cloud_ptr,
    std::shared_ptr<TrajectoryLabel> &_history_trajectory_label_ptr) {
    // patches_lidar_vec: First region_width_elements*region_width_elements is non-ground points and the last
    // region_width_elements*region_width_elements is ground points
    // std::vector<Eigen::MatrixX<BasicType>> patches_w_vec;
    std::vector<std::shared_ptr<PatchInfo<BasicType>>> patch_info_ptr_vec;
    // insert new keyframe
    std::shared_ptr<KeyframeInfo<BasicType>> keyframe_info_ptr;
    if (is_new_keyframe) {
        Eigen::Isometry3d T_w_lidar_relative = Eigen::Isometry3d::Identity();
        // integrate and sampling previous observation
        if (auto last_keyframe_info_ptr = spatial_hashing_ptr->latest_keyframe()) { // not the first one
            //     last_keyframe_info_ptr->merage_succeed_associations(
            //         curl_voxel_mapping_config_ptr->BA_region_width_elements,
            //         curl_voxel_mapping_config_ptr->BA_observation_outlier_rejection_leaf_size,
            //         curl_voxel_mapping_config_ptr->BA_observation_outlier_rejection_minimum_neighbours);
            // }
            T_w_lidar_relative = last_keyframe_info_ptr->get_graph_pose_w_lidar_T() * _T_lastKeyframe_keyframe;
        }

        // add new keyframe
        if (is_add_trajectory_segment) {
            spatial_hashing_ptr->clear_history_keyframe_clouds();
            keyframe_info_ptr = std::make_shared<KeyframeInfo<BasicType>>(
                time, T_w_lidar_curr, _T_lastKeyframe_keyframe, T_w_lidar_relative,
                std::make_shared<TrajectoryLabel>(spatial_hashing_ptr->keyframe_size(), debug_config_ptr->color_map));
            if (spatial_hashing_ptr->keyframe_size() > 1) { // not the first one
                if (!is_loop_closure_detected) {
                    if (auto last_keyframe = spatial_hashing_ptr->latest_keyframe()) {
                        TrajectoryLabel::insert_neighbor_label(keyframe_info_ptr->trajectory_label_ptr,
                                                              last_keyframe->trajectory_label_ptr);
                    }
                } else {
                    TrajectoryLabel::merge_neighbor_label(keyframe_info_ptr->trajectory_label_ptr,
                                                          _history_trajectory_label_ptr);
                    is_loop_closure_detected = false;
                }
            }
        } else {
            auto last_keyframe = spatial_hashing_ptr->latest_keyframe();
            keyframe_info_ptr = std::make_shared<KeyframeInfo<BasicType>>(
                time, T_w_lidar_curr, _T_lastKeyframe_keyframe, T_w_lidar_relative,
                last_keyframe ? last_keyframe->trajectory_label_ptr
                              : std::make_shared<TrajectoryLabel>(0, debug_config_ptr->color_map));
        }

        keyframe_info_ptr->frame_num = spatial_hashing_ptr->keyframe_size();
        spatial_hashing_ptr->insert_keyframe(keyframe_info_ptr);
        keyframe_info_ptr->is_active = is_active;
        keyframe_info_ptr->frame_idx = frame_idx;
        // NOTE: only keep the first observation
        //     keyframe_info_ptr->succeed_associations_insert(succeed_associations);
        // }
        // save raw point cloud into scan context
        if (curl_loop_closure_config_ptr->is_enable) {
            curl::subSampleFrame(*seg_cloud_ptr, 0.2);
            keyframe_info_ptr->seg_cloud_ptr = seg_cloud_ptr;
            curl::subSampleFrame(*ground_cloud_ptr, 0.2);
            keyframe_info_ptr->ground_cloud_ptr = ground_cloud_ptr;

            pcl::PointCloud<PointT> raw_cloud = (*seg_cloud_ptr) + (*ground_cloud_ptr);
            curl::subSampleFrame(raw_cloud, 0.5);
            makeAndSaveScancontextAndKeys(raw_cloud);
        }
    } else {
        keyframe_info_ptr = spatial_hashing_ptr->latest_keyframe();
        if (!keyframe_info_ptr) {
            return false;
        }
        // add new observations into the keyframe
        if (curl_loop_closure_config_ptr->is_enable) {
            pcl::PointCloud<PointT> transformed_seg_cloud, transformed_ground_cloud;
            pcl::transformPointCloud(*seg_cloud_ptr, transformed_seg_cloud, _T_lastKeyframe_keyframe.cast<float>());
            pcl::transformPointCloud(*ground_cloud_ptr, transformed_ground_cloud,
                                     _T_lastKeyframe_keyframe.cast<float>());
            *(keyframe_info_ptr->seg_cloud_ptr) = *(keyframe_info_ptr->seg_cloud_ptr) + transformed_seg_cloud;
            *(keyframe_info_ptr->ground_cloud_ptr) = *(keyframe_info_ptr->ground_cloud_ptr) + transformed_ground_cloud;
            curl::subSampleFrame(*(keyframe_info_ptr->seg_cloud_ptr), 0.2);
            curl::subSampleFrame(*(keyframe_info_ptr->ground_cloud_ptr), 0.2);
        }
    }

    if (!is_active) { // only the first frame can enter here
        keyframe_info_ptr->set_BA_T_w_lidar(T_w_lidar_curr);
        keyframe_info_ptr->is_BA_T_w_lidar_set = true;
    }
    if (patches_lidar_vec.empty()) {
        return is_new_keyframe;
    }
    int seg_points_num = 0, ground_points_num = 0;
    for (int idx = 0; idx < patches_lidar_vec.size(); ++idx) {
        if (!is_ground_cloud_vec[idx]) {
            seg_points_num += patches_lidar_vec[idx].cols();
        } else {
            ground_points_num += patches_lidar_vec[idx].cols();
        }
    }
    // add all points into together and divide them into new patches
    Eigen::MatrixX<BasicType> merged_seg_point_cloud_lidar(3, seg_points_num);
    Eigen::MatrixX<BasicType> merged_ground_point_cloud_lidar(3, ground_points_num);

    int seg_idx = 0, ground_idx = 0;
    for (int idx = 0; idx < patches_lidar_vec.size(); ++idx) {
        if (!is_ground_cloud_vec[idx]) {
            for (int i = 0; i < patches_lidar_vec[idx].cols(); ++i, ++seg_idx) {
                merged_seg_point_cloud_lidar(0, seg_idx) = patches_lidar_vec[idx](0, i);
                merged_seg_point_cloud_lidar(1, seg_idx) = patches_lidar_vec[idx](1, i);
                merged_seg_point_cloud_lidar(2, seg_idx) = patches_lidar_vec[idx](2, i);
            }
        } else {
            for (int i = 0; i < patches_lidar_vec[idx].cols(); ++i, ++ground_idx) {
                merged_ground_point_cloud_lidar(0, ground_idx) = patches_lidar_vec[idx](0, i);
                merged_ground_point_cloud_lidar(1, ground_idx) = patches_lidar_vec[idx](1, i);
                merged_ground_point_cloud_lidar(2, ground_idx) = patches_lidar_vec[idx](2, i);
            }
        }
    }

    Eigen::MatrixX<BasicType> merged_seg_point_cloud_world =
        (T_w_lidar_curr(Eigen::seq(0, 2), Eigen::seq(0, 2)).cast<BasicType>() * merged_seg_point_cloud_lidar)
            .colwise() +
        T_w_lidar_curr(Eigen::seq(0, 2), 3).cast<BasicType>();

    Eigen::MatrixX<BasicType> merged_ground_point_cloud_world =
        (T_w_lidar_curr(Eigen::seq(0, 2), Eigen::seq(0, 2)).cast<BasicType>() * merged_ground_point_cloud_lidar)
            .colwise() +
        T_w_lidar_curr(Eigen::seq(0, 2), 3).cast<BasicType>();
    // couple procedure between remove points and get local coordinate
    // std::unordered_map<std::vector<int>, Eigen::MatrixX<BasicType>, VoxelHashFuncPrime> seg_points_vec_world,
    //     ground_points_vec_world; // box world centroid and corresponding points in world frame
    std::vector<std::pair<std::vector<int>, Eigen::MatrixX<BasicType>>> seg_points_vec_world,
        ground_points_vec_world; // box world centroid and corresponding points in world frame
    remove_points_inside_bounding_box(keyframe_info_ptr, merged_seg_point_cloud_world, merged_ground_point_cloud_world,
                                      seg_points_vec_world, ground_points_vec_world);
    if (debug_config_ptr->is_pub_dense_reconstruction) {
        patch_info_ptr_vec.reserve(seg_points_vec_world.size() + ground_points_vec_world.size());
    }
    //    re_clustering(frame_idx, merged_seg_point_cloud_world, merged_ground_point_cloud_world, seg_points_vec_world,
    //                  ground_points_vec_world);
    // get bounding box and local coordinate
#pragma omp parallel for num_threads(curl_voxel_mapping_config_ptr->patch_initialization_thread_num) default(none)     \
    shared(seg_points_vec_world, direct_method_config_ptr, SH_table_config_ptr, curl_voxel_mapping_config_ptr,         \
           debug_config_ptr, keyframe_info_ptr, T_w_lidar_curr, is_new_keyframe, spatial_hashing_ptr,                  \
           patch_info_ptr_vec, is_add_trajectory_segment)
    for (auto &seg_patch_world : seg_points_vec_world) {
        bool is_successfully_added;
        std::vector<double> lower_bound_w(3), upper_bound_w(3);
        Eigen::Isometry3d T_obj_keyframe;
        upper_bound_w[0] = seg_patch_world.first[0] * curl_voxel_mapping_config_ptr->cut_threshold;
        upper_bound_w[1] = seg_patch_world.first[1] * curl_voxel_mapping_config_ptr->cut_threshold;
        upper_bound_w[2] = seg_patch_world.first[2] * curl_voxel_mapping_config_ptr->cut_threshold;
        lower_bound_w[0] = upper_bound_w[0] - curl_voxel_mapping_config_ptr->cut_threshold;
        lower_bound_w[1] = upper_bound_w[1] - curl_voxel_mapping_config_ptr->cut_threshold;
        lower_bound_w[2] = upper_bound_w[2] - curl_voxel_mapping_config_ptr->cut_threshold;
        Eigen::Matrix4<BasicType> cov_4;
        Eigen::Vector3d centroid_lidar;
        PROJECTION_AXIS projection_axis;
        is_successfully_added = get_local_coordinate_with_eig(
            false, Eigen::Isometry3d(keyframe_info_ptr->get_T_w_lidar()), seg_patch_world.second, lower_bound_w,
            upper_bound_w, T_obj_keyframe, cov_4, projection_axis);
        if (is_successfully_added) {

            // get bounding box in patch frame(obj) to get plane bounding

            // Eigen::Isometry3d T_obj_w =
            //     (Eigen::Isometry3d(keyframe_info_ptr->get_T_w_lidar()) * T_obj_keyframe.inverse()).inverse();
            Eigen::Isometry3d T_lidar_w = Eigen::Isometry3d(keyframe_info_ptr->get_T_w_lidar()).inverse();
            Eigen::Isometry3d T_obj_w = T_obj_keyframe * T_lidar_w;
            Eigen::MatrixX<BasicType> seg_patch_lidar =
                (T_lidar_w.matrix().cast<BasicType>()(Eigen::seq(0, 2), Eigen::seq(0, 2)) * seg_patch_world.second)
                    .colwise() +
                T_lidar_w.matrix().cast<BasicType>()(Eigen::seq(0, 2), 3);
            Eigen::VectorX<BasicType> seg_lidar_range_squared = seg_patch_lidar.colwise().squaredNorm();
            Eigen::MatrixX<BasicType> seg_patch_obj =
                (T_obj_w.matrix().cast<BasicType>()(Eigen::seq(0, 2), Eigen::seq(0, 2)) * seg_patch_world.second)
                    .colwise() +
                T_obj_w.matrix().cast<BasicType>()(Eigen::seq(0, 2), 3);

            // add this new patch into patch procession, generating new mask and sph coefficients
            std::shared_ptr<PatchProcession<BasicType>> patch_procession_ptr;
            if (!SH_table_config_ptr->is_SH_analytic_jacobian) {
                patch_procession_ptr = std::make_shared<PatchProcession<BasicType>>(
                    T_w_lidar_curr(Eigen::seq(0, 2), 3), direct_method_config_ptr, SH_table_config_ptr,
                    debug_config_ptr, seg_patch_obj, seg_lidar_range_squared,
                    curl_voxel_mapping_config_ptr->half_diag_cut_threshold, SH_table_config_ptr->max_SH_degree, false);
            } else {
                patch_procession_ptr = std::make_shared<PatchProcession<BasicType>>(
                    T_w_lidar_curr(Eigen::seq(0, 2), 3), direct_method_config_ptr, SH_table_config_ptr,
                    debug_config_ptr, seg_patch_obj, seg_lidar_range_squared,
                    curl_voxel_mapping_config_ptr->half_diag_cut_threshold, SH_table_config_ptr->max_SH_degree, false,
                    0);
            }
            std::shared_ptr<PatchInfo<BasicType>> patch_info_ptr;
            patch_info_ptr = std::make_shared<PatchInfo<BasicType>>(keyframe_info_ptr, T_obj_keyframe,
                                                                    patch_procession_ptr, false, is_new_keyframe, cov_4,
                                                                    projection_axis, seg_patch_obj.cols(), colors[0]);
            patch_info_ptr->initial_points_w_vec.push_back(seg_patch_world.second);
            patch_info_ptr->range_squared_distance_vec.insert(
                patch_info_ptr->range_squared_distance_vec.end(), seg_lidar_range_squared.data(),
                seg_lidar_range_squared.data() + seg_lidar_range_squared.size());
            if (is_add_trajectory_segment) {
                patch_info_ptr->label_frame_num_set.insert(keyframe_info_ptr->frame_num);
            }

            // add bounding box size into this patch_info_ptr
            patch_info_ptr->set_lower_bound_w(lower_bound_w);
            patch_info_ptr->set_upper_bound_w(upper_bound_w);
            for (unsigned int i = 0; i < 3; i++) {
                lower_bound_w[i] -= aabb_config_ptr->extra_skin_thickness;
                upper_bound_w[i] += aabb_config_ptr->extra_skin_thickness;
            }
            patch_info_ptr->set_enlarged_lower_bound_w(lower_bound_w);
            patch_info_ptr->set_enlarged_upper_bound_w(upper_bound_w);
#pragma omp critical
            {
                spatial_hashing_ptr->insert_patch(patch_info_ptr);
                patch_info_ptr_vec.push_back(patch_info_ptr);
            }
        }
    }
    // get ground cloud information
    // calculate ground R_lidar_obj by using all ground cloud together first
#pragma omp parallel for num_threads(curl_voxel_mapping_config_ptr->patch_initialization_thread_num) default(none)     \
    shared(ground_points_vec_world, direct_method_config_ptr, SH_table_config_ptr, curl_voxel_mapping_config_ptr,      \
           debug_config_ptr, keyframe_info_ptr, T_w_lidar_curr, is_new_keyframe, spatial_hashing_ptr,                  \
           patch_info_ptr_vec, is_add_trajectory_segment, writelock)
    for (auto &ground_patch_world : ground_points_vec_world) {
        bool is_successfully_added;
        std::vector<double> lower_bound_w(3), upper_bound_w(3);
        Eigen::Isometry3d T_obj_keyframe;
        upper_bound_w[0] = ground_patch_world.first[0] * curl_voxel_mapping_config_ptr->cut_threshold;
        upper_bound_w[1] = ground_patch_world.first[1] * curl_voxel_mapping_config_ptr->cut_threshold;
        upper_bound_w[2] = ground_patch_world.first[2] * curl_voxel_mapping_config_ptr->cut_threshold;
        lower_bound_w[0] = upper_bound_w[0] - curl_voxel_mapping_config_ptr->cut_threshold;
        lower_bound_w[1] = upper_bound_w[1] - curl_voxel_mapping_config_ptr->cut_threshold;
        lower_bound_w[2] = upper_bound_w[2] - curl_voxel_mapping_config_ptr->cut_threshold;
        Eigen::Matrix4<BasicType> cov_4;
        PROJECTION_AXIS projection_axis;
        is_successfully_added = get_local_coordinate_with_eig(
            true, Eigen::Isometry3d(keyframe_info_ptr->get_T_w_lidar()), ground_patch_world.second, lower_bound_w,
            upper_bound_w, T_obj_keyframe, cov_4, projection_axis);
        if (is_successfully_added) {
            // get bounding box in patch frame(obj) to get plane bounding
            Eigen::Isometry3d T_lidar_w = Eigen::Isometry3d(keyframe_info_ptr->get_T_w_lidar()).inverse();
            Eigen::Isometry3d T_obj_w = T_obj_keyframe * T_lidar_w;

            Eigen::VectorX<BasicType> ground_lidar_range_squared =
                ((T_lidar_w.matrix().cast<BasicType>()(Eigen::seq(0, 2), Eigen::seq(0, 2)) * ground_patch_world.second)
                     .colwise() +
                 T_lidar_w.matrix().cast<BasicType>()(Eigen::seq(0, 2), 3))
                    .colwise()
                    .squaredNorm();
            Eigen::MatrixX<BasicType> ground_patch_obj =
                (T_obj_w.matrix().cast<BasicType>()(Eigen::seq(0, 2), Eigen::seq(0, 2)) * ground_patch_world.second)
                    .colwise() +
                T_obj_w.matrix().cast<BasicType>()(Eigen::seq(0, 2), 3);
            // add this new patch into patch procession, generating new mask and sph coefficients
            std::shared_ptr<PatchProcession<BasicType>> patch_procession_ptr;
            if (!SH_table_config_ptr->is_SH_analytic_jacobian) {
                patch_procession_ptr = std::make_shared<PatchProcession<BasicType>>(
                    T_w_lidar_curr(Eigen::seq(0, 2), 3), direct_method_config_ptr, SH_table_config_ptr,
                    debug_config_ptr, ground_patch_obj, ground_lidar_range_squared,
                    curl_voxel_mapping_config_ptr->half_diag_cut_threshold, SH_table_config_ptr->ground_SH_degree,
                    true);
            } else {
                patch_procession_ptr = std::make_shared<PatchProcession<BasicType>>(
                    T_w_lidar_curr(Eigen::seq(0, 2), 3), direct_method_config_ptr, SH_table_config_ptr,
                    debug_config_ptr, ground_patch_obj, ground_lidar_range_squared,
                    curl_voxel_mapping_config_ptr->half_diag_cut_threshold, SH_table_config_ptr->ground_SH_degree, true,
                    0);
            }

            std::shared_ptr<PatchInfo<BasicType>> patch_info_ptr;
            patch_info_ptr = std::make_shared<PatchInfo<BasicType>>(
                keyframe_info_ptr, T_obj_keyframe, patch_procession_ptr, true, is_new_keyframe, cov_4, projection_axis,
                ground_patch_obj.cols(), colors[0]);
            if (is_add_trajectory_segment) {
                patch_info_ptr->label_frame_num_set.insert(keyframe_info_ptr->frame_num);
            }

            // add bounding box size into this patch_info_ptr
            patch_info_ptr->set_lower_bound_w(lower_bound_w);
            patch_info_ptr->set_upper_bound_w(upper_bound_w);
            for (unsigned int i = 0; i < 3; i++) {
                lower_bound_w[i] -= aabb_config_ptr->extra_skin_thickness;
                upper_bound_w[i] += aabb_config_ptr->extra_skin_thickness;
            }
            patch_info_ptr->set_enlarged_lower_bound_w(lower_bound_w);
            patch_info_ptr->set_enlarged_upper_bound_w(upper_bound_w);
            omp_set_lock(&writelock);

            spatial_hashing_ptr->insert_patch(patch_info_ptr);
            // for visualization
            // patches_w_vec.push_back(ground_patch_world.second);
            patch_info_ptr_vec.push_back(patch_info_ptr);

            omp_unset_lock(&writelock);
        }
    }

    // if (spatial_hashing_ptr->active_keyframe_size() > curl_voxel_mapping_config_ptr->BA_window_size &&
    //     local_BA_cv.notify_one();
    //     std::unique_lock<std::mutex> ul(BA_wait_lock);
    //     tracking_wait_BA_cv.wait(ul);
    // }
    // rt_publish_landmark_cloud(patches_w_vec);
    if (debug_config_ptr->is_pub_dense_reconstruction) {
        for (auto &patch_info_ptr : patch_info_ptr_vec) {
            rt_publish_landmark_patch_recons(patch_info_ptr);
        }
    }
    return is_new_keyframe;
}

template <typename BasicType>
void CurlVoxelMapping<BasicType>::rt_publish_landmark_patch_recons(
    const std::shared_ptr<PatchInfo<BasicType>> &patch_info_ptr) {
    visualization_msgs::Marker patch_recons =
        set_default_marker("map", "recons_patch", patch_info_ptr->key, patch_info_ptr->color);
    Eigen::Isometry3d T_w_obj =
        Eigen::Isometry3d(patch_info_ptr->keyframe_ptr->get_T_w_lidar()) * patch_info_ptr->T_obj_lidar.inverse();
    Eigen::MatrixX<BasicType> recons_v_obj =
        patch_info_ptr->patch_procession_ptr->get_recons_v_obj_cached().transpose();
    Eigen::MatrixX<BasicType> recons_v_world =
        (T_w_obj.rotation().cast<BasicType>() * recons_v_obj).colwise() + T_w_obj.translation().cast<BasicType>();

    for (int j = 0; j < recons_v_world.cols(); ++j) {
        geometry_msgs::Point pt;
        pt.x = recons_v_world(0, j);
        pt.y = recons_v_world(1, j);
        pt.z = recons_v_world(2, j);
        if (std::abs(recons_v_obj(2, j)) < curl_voxel_mapping_config_ptr->cut_threshold) {
            patch_recons.points.push_back(pt);
        }
    }
    pub_landmark_patch.publish(patch_recons);
}

template <typename BasicType> void CurlVoxelMapping<BasicType>::refresh_dense_reconstruction_markers() {
    if (!debug_config_ptr || !debug_config_ptr->is_pub_dense_reconstruction) {
        return;
    }
    visualization_msgs::Marker clear_marker;
    clear_marker.header.frame_id = "map";
    clear_marker.header.stamp = ros::Time::now();
    clear_marker.ns = "recons_patch";
    clear_marker.action = visualization_msgs::Marker::DELETEALL;
    pub_landmark_patch.publish(clear_marker);

    std::vector<std::shared_ptr<PatchInfo<BasicType>>> patch_ptrs;
    const auto keyframes = spatial_hashing_ptr->keyframes_snapshot();
    for (const auto &keyframe_ptr : keyframes) {
        if (!keyframe_ptr) {
            continue;
        }
        std::unique_lock<std::mutex> ul_local_patches(keyframe_ptr->local_patches_lock);
        for (const auto &patch_id : keyframe_ptr->local_patches) {
            if (auto patch_ptr = spatial_hashing_ptr->get_patch_by_id(patch_id)) {
                patch_ptrs.push_back(patch_ptr);
            }
        }
    }

    for (const auto &patch_ptr : patch_ptrs) {
        rt_publish_landmark_patch_recons(patch_ptr);
    }
}

template <typename BasicType>
visualization_msgs::Marker
CurlVoxelMapping<BasicType>::set_default_marker(const std::string &frame_id, const std::string &ns, const int &id,
                                                const std::array<float, 3> &color, int32_t type, const float scale[3],
                                                const float position[3], const std::string &text) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = ros::Time::now();
    marker.ns = ns;
    marker.action = visualization_msgs::Marker::ADD;
    marker.id = id;
    marker.color.r = color[0];
    marker.color.g = color[1];
    marker.color.b = color[2];
    marker.color.a = 1;
    marker.type = type;
    marker.scale.x = scale[0];
    marker.scale.y = scale[1];
    marker.scale.z = scale[2];
    if (type == visualization_msgs::Marker::TEXT_VIEW_FACING) {
        marker.pose.position.x = position[0];
        marker.pose.position.y = position[1];
        marker.pose.position.z = position[2];
        marker.text = text;
    }

    marker.pose.orientation.w = 1.0;
    return marker;
}

template <typename BasicType>
void CurlVoxelMapping<BasicType>::makeAndSaveScancontextAndKeys(pcl::PointCloud<PointT> &scan_down) {
    scManager.makeAndSaveScancontextAndKeys(scan_down);
}

template <typename BasicType> std::pair<int, float> CurlVoxelMapping<BasicType>::detectLoopClosureID(void) {
    return scManager.detectLoopClosureID();
}

template <typename BasicType>
std::pair<int, float> CurlVoxelMapping<BasicType>::detectLoopClosureID(const int query_idx) {
    return scManager.detectLoopClosureID(query_idx);
}

template <typename BasicType>
std::pair<std::vector<double>, std::vector<double>>
CurlVoxelMapping<BasicType>::preprocessing(const double time, const pcl::PointCloud<PointT>::Ptr &seg_cloud_ptr,
                                           const pcl::PointCloud<PointT>::Ptr &ground_cloud_ptr,
                                           const Eigen::Matrix4f _T_w_j,
                                           std::vector<std::vector<Eigen::MatrixX<BasicType>>> &_point_cloud_vec,
                                           PointCloudInfo<BasicType> &_point_cloud_info) {
    _point_cloud_info.time = time;
    _point_cloud_info.seg_cloud_ptr = seg_cloud_ptr;
    _point_cloud_info.ground_cloud_ptr = ground_cloud_ptr;
    // divide the point cloud into several patches
    pcl::PointCloud<PointT> transformed_seg_cloud, transformed_ground_cloud;
    pcl::transformPointCloud(*seg_cloud_ptr, transformed_seg_cloud, _T_w_j);
    pcl::transformPointCloud(*ground_cloud_ptr, transformed_ground_cloud, _T_w_j);
    // get scan lower bound and upper bound in world coordinate (need to include the upper bound of the grid)
    auto bounds_w = curl::get_bounding_box_w<PointT>(transformed_seg_cloud, transformed_ground_cloud);
    // divide the point cloud into several patches
    double seg_leaf_array[3] = {curl_voxel_mapping_config_ptr->cut_threshold,
                                curl_voxel_mapping_config_ptr->cut_threshold,
                                curl_voxel_mapping_config_ptr->cut_threshold};
    std::vector<std::vector<Eigen::MatrixX<BasicType>>> seg_cloud_vec;
    seg_cloud_vec = curl::divide_patches<PointT, BasicType>(
        transformed_seg_cloud, *seg_cloud_ptr, seg_leaf_array,
        curl_voxel_mapping_config_ptr->patch_minimun_pts_num_after_filter, bounds_w.first, bounds_w.second,
        curl_tracking_config_ptr->region_width_elements);
    std::vector<std::vector<bool>> seg_is_ground_cloud_vec(seg_cloud_vec.size());
    for (int i = 0; i < seg_cloud_vec.size(); ++i) {
        seg_is_ground_cloud_vec[i].resize(seg_cloud_vec[i].size(), false);
    }
    // divide the point cloud into several patches
    double ground_leaf_array[3] = {curl_voxel_mapping_config_ptr->cut_threshold,
                                   curl_voxel_mapping_config_ptr->cut_threshold,
                                   curl_voxel_mapping_config_ptr->cut_threshold};
    std::vector<std::vector<Eigen::MatrixX<BasicType>>> ground_cloud_vec;
    ground_cloud_vec = curl::divide_patches<PointT, BasicType>(
        transformed_ground_cloud, *ground_cloud_ptr, ground_leaf_array,
        curl_voxel_mapping_config_ptr->patch_minimun_pts_num_after_filter, bounds_w.first, bounds_w.second,
        curl_tracking_config_ptr->region_width_elements);
    std::vector<std::vector<bool>> ground_is_ground_cloud_vec(ground_cloud_vec.size());
    for (int i = 0; i < ground_cloud_vec.size(); ++i) {
        ground_is_ground_cloud_vec[i].resize(ground_cloud_vec[i].size(), true);
    }
    // calculate patches' information (local coordinate extra)
    _point_cloud_vec.reserve(seg_cloud_vec.size() + ground_cloud_vec.size());
    _point_cloud_vec.insert(_point_cloud_vec.end(), seg_cloud_vec.begin(), seg_cloud_vec.end());
    _point_cloud_info.is_ground_cloud_vec.reserve(seg_cloud_vec.size() + ground_cloud_vec.size());
    _point_cloud_info.is_ground_cloud_vec.insert(_point_cloud_info.is_ground_cloud_vec.end(),
                                                 seg_is_ground_cloud_vec.begin(), seg_is_ground_cloud_vec.end());
    // get ground cloud information
    // calculate ground R_lidar_obj by using all ground cloud together first
    for (int i = 0; i < ground_cloud_vec.size(); ++i) {
        _point_cloud_vec[i].insert(_point_cloud_vec[i].end(), ground_cloud_vec[i].begin(), ground_cloud_vec[i].end());
        _point_cloud_info.is_ground_cloud_vec[i].insert(_point_cloud_info.is_ground_cloud_vec[i].end(),
                                                        ground_is_ground_cloud_vec[i].begin(),
                                                        ground_is_ground_cloud_vec[i].end());
    }
    return bounds_w;
}

template <typename BasicType>
std::pair<std::vector<double>, std::vector<double>>
CurlVoxelMapping<BasicType>::preprocessing(const double time, const pcl::PointCloud<PointT>::Ptr &seg_cloud_ptr,
                                           const pcl::PointCloud<PointT>::Ptr &ground_cloud_ptr,
                                           const Eigen::Matrix4d &_T_j_1_j, const Eigen::Matrix4f _T_w_j,
                                           std::vector<std::vector<Eigen::MatrixX<BasicType>>> &_point_cloud_vec,
                                           PointCloudInfo<BasicType> &_point_cloud_info) {
    _point_cloud_info.time = time;
    _point_cloud_info.seg_cloud_ptr = seg_cloud_ptr;
    _point_cloud_info.ground_cloud_ptr = ground_cloud_ptr;
    if (curl_voxel_mapping_config_ptr->is_deskew) {
        curl::deskewing(*_point_cloud_info.seg_cloud_ptr, _T_j_1_j);
        curl::deskewing(*_point_cloud_info.ground_cloud_ptr, _T_j_1_j);
    }
    // divide the point cloud into several patches
    pcl::PointCloud<PointT> transformed_seg_cloud, transformed_ground_cloud;
    pcl::transformPointCloud(*seg_cloud_ptr, transformed_seg_cloud, _T_w_j);
    pcl::transformPointCloud(*ground_cloud_ptr, transformed_ground_cloud, _T_w_j);
    // get scan lower bound and upper bound in world coordinate (need to include the upper bound of the grid)
    auto bounds_w = curl::get_bounding_box_w<PointT>(transformed_seg_cloud, transformed_ground_cloud);
    // divide the point cloud into several patches
    double seg_leaf_array[3] = {curl_voxel_mapping_config_ptr->cut_threshold,
                                curl_voxel_mapping_config_ptr->cut_threshold,
                                curl_voxel_mapping_config_ptr->cut_threshold};
    std::vector<std::vector<Eigen::MatrixX<BasicType>>> seg_cloud_vec;
    seg_cloud_vec = curl::divide_patches<PointT, BasicType>(
        transformed_seg_cloud, *seg_cloud_ptr, seg_leaf_array,
        curl_voxel_mapping_config_ptr->patch_minimun_pts_num_after_filter, bounds_w.first, bounds_w.second,
        curl_tracking_config_ptr->region_width_elements);
    std::vector<std::vector<bool>> seg_is_ground_cloud_vec(seg_cloud_vec.size());
    for (int i = 0; i < seg_cloud_vec.size(); ++i) {
        seg_is_ground_cloud_vec[i].resize(seg_cloud_vec[i].size(), false);
    }
    // divide the point cloud into several patches
    double ground_leaf_array[3] = {curl_voxel_mapping_config_ptr->cut_threshold,
                                   curl_voxel_mapping_config_ptr->cut_threshold,
                                   curl_voxel_mapping_config_ptr->cut_threshold};
    std::vector<std::vector<Eigen::MatrixX<BasicType>>> ground_cloud_vec;
    ground_cloud_vec = curl::divide_patches<PointT, BasicType>(
        transformed_ground_cloud, *ground_cloud_ptr, ground_leaf_array,
        curl_voxel_mapping_config_ptr->patch_minimun_pts_num_after_filter, bounds_w.first, bounds_w.second,
        curl_tracking_config_ptr->region_width_elements);
    std::vector<std::vector<bool>> ground_is_ground_cloud_vec(ground_cloud_vec.size());
    for (int i = 0; i < ground_cloud_vec.size(); ++i) {
        ground_is_ground_cloud_vec[i].resize(ground_cloud_vec[i].size(), true);
    }
    // calculate patches' information (local coordinate extra)
    _point_cloud_vec.reserve(seg_cloud_vec.size() + ground_cloud_vec.size());
    _point_cloud_vec.insert(_point_cloud_vec.end(), seg_cloud_vec.begin(), seg_cloud_vec.end());
    _point_cloud_info.is_ground_cloud_vec.reserve(seg_cloud_vec.size() + ground_cloud_vec.size());
    _point_cloud_info.is_ground_cloud_vec.insert(_point_cloud_info.is_ground_cloud_vec.end(),
                                                 seg_is_ground_cloud_vec.begin(), seg_is_ground_cloud_vec.end());
    // get ground cloud information
    // calculate ground R_lidar_obj by using all ground cloud together first
    for (int i = 0; i < ground_cloud_vec.size(); ++i) {
        _point_cloud_vec[i].insert(_point_cloud_vec[i].end(), ground_cloud_vec[i].begin(), ground_cloud_vec[i].end());
        _point_cloud_info.is_ground_cloud_vec[i].insert(_point_cloud_info.is_ground_cloud_vec[i].end(),
                                                        ground_is_ground_cloud_vec[i].begin(),
                                                        ground_is_ground_cloud_vec[i].end());
    }
    return bounds_w;
}

template <typename BasicType>
bool CurlVoxelMapping<BasicType>::curl_registration_method(
    const double cloud_time, const int _last_keyframe_idx, const pcl::PointCloud<PointT>::Ptr &seg_cloud_ptr,
    const pcl::PointCloud<PointT>::Ptr &ground_cloud_ptr, const int _current_label_frame_num,
    const std::shared_ptr<TrajectoryLabel> &_curr_trajectory_label_ptr, const bool is_use_all_associated_patches,
    const double highest_IoU_new_landmark_thres, const int minimum_observation_num, const bool is_voxel_grid_filter,
    const double leaf_size, const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_w_j_keyframe,
    const Eigen::Matrix4d &_T_j_1_j, Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_w_j,
    std::vector<std::vector<Eigen::MatrixX<BasicType>>> &_point_cloud_vec, PointCloudInfo<BasicType> &_point_cloud_info,
    std::vector<std::tuple<Eigen::MatrixX<BasicType>, std::shared_ptr<PatchInfo<BasicType>>, double>>
        &succeed_associations,
    std::pair<std::vector<double>, std::vector<double>> &bounds_w,
    std::unordered_set<std::array<int, 2>, Voxel2DHashFuncPrimeArray> &new_box_map, int &_number_patches,
    bool &_is_add_keyframe, bool &_is_add_trajectory_segment, std::vector<BasicType> &_our_costs,
    std::vector<std::pair<std::vector<double>, std::vector<double>>> &query_bounding_box_vec,
    std::vector<std::pair<std::vector<double>, std::vector<double>>> &map_bounding_box_vec) {
    bounds_w = preprocessing(cloud_time, seg_cloud_ptr, ground_cloud_ptr, _T_j_1_j, _T_w_j.cast<float>(),
                             _point_cloud_vec, _point_cloud_info);
    //            _point_cloud_info.T_obj_lidar_vec.resize(_point_cloud_vec.size());
    _point_cloud_info.intersected_score_pairs_vec.resize(_point_cloud_vec.size());
    _point_cloud_info.bounding_box_pair_vec.resize(_point_cloud_vec.size());

    ceres::Problem problem;

    ceres::Manifold *SE3_manifold = SE3Manifold::Create();
    ceres::LossFunctionWrapper *loss_function = new ceres::LossFunctionWrapper(
        new ceres::HuberLoss(curl_tracking_config_ptr->seg_kernel_threshold), ceres::DO_NOT_TAKE_OWNERSHIP);

    // new data-association starts here
    //            std::vector<std::vector<double>> entire_scores(_point_cloud_vec.size());
    int intersect_counter = 0;
    int trajectory_label_frame_num_counter = 0;
    const Eigen::Matrix<BasicType, 3, 3> R_w_j =
        _T_w_j(Eigen::seq(0, 2), Eigen::seq(0, 2)).template cast<BasicType>();
    const Eigen::Matrix<BasicType, 3, 1> t_w_j = _T_w_j(Eigen::seq(0, 2), 3).template cast<BasicType>();

#pragma omp parallel for num_threads(curl_voxel_mapping_config_ptr->data_association_thread_num) default(none)         \
    shared(_point_cloud_vec, _point_cloud_info, trajectory_label_frame_num_counter, _curr_trajectory_label_ptr,        \
           _T_w_j, _current_label_frame_num, is_use_all_associated_patches, intersect_counter, R_w_j, t_w_j)
    for (int i = 0; i < _point_cloud_vec.size(); ++i) { // NOTE: each region
        _point_cloud_info.intersected_score_pairs_vec[i].resize(_point_cloud_vec[i].size());
        _point_cloud_info.bounding_box_pair_vec[i].resize(_point_cloud_vec[i].size());
        for (int j = 0; j < _point_cloud_vec[i].size(); ++j) { // NOTE: elements of region
            // TODO: this part has been done in function preprocessing, which can merge these two together to
            // speed-up
            const Eigen::MatrixX<BasicType> &patch_cloud = _point_cloud_vec[i][j];
            Eigen::Matrix<BasicType, 3, 1> min_pt;
            Eigen::Matrix<BasicType, 3, 1> max_pt;
            min_pt.setConstant(std::numeric_limits<BasicType>::max());
            max_pt.setConstant(std::numeric_limits<BasicType>::lowest());
            for (int c = 0; c < patch_cloud.cols(); ++c) {
                const Eigen::Matrix<BasicType, 3, 1> p = R_w_j * patch_cloud.col(c) + t_w_j;
                min_pt = min_pt.cwiseMin(p);
                max_pt = max_pt.cwiseMax(p);
            }

            std::vector<double> lower_bound_w(3);
            lower_bound_w[0] = static_cast<double>(min_pt(0));
            lower_bound_w[1] = static_cast<double>(min_pt(1));
            lower_bound_w[2] = static_cast<double>(min_pt(2));
            std::vector<double> upper_bound_w(3);
            upper_bound_w[0] = static_cast<double>(max_pt(0));
            upper_bound_w[1] = static_cast<double>(max_pt(1));
            upper_bound_w[2] = static_cast<double>(max_pt(2));
            _point_cloud_info.bounding_box_pair_vec[i][j] = std::make_pair(lower_bound_w, upper_bound_w);
            std::vector<std::pair<std::shared_ptr<PatchInfo<BasicType>>, double>> intersected_pairs =
                spatial_hashing_ptr->query_data_association_overlap(lower_bound_w, upper_bound_w);

            for (const auto &pair : intersected_pairs) {
                if (is_use_all_associated_patches ||
                    TrajectoryLabel::is_connected(_curr_trajectory_label_ptr,
                                                  pair.first->keyframe_ptr->trajectory_label_ptr)) {
                    if (pair.first->is_ground != _point_cloud_info.is_ground_cloud_vec[i][j]) {
                        continue;
                    }
                    _point_cloud_info.intersected_score_pairs_vec[i][j].emplace_back(pair.first, pair.second);
                    #pragma omp atomic
                    ++intersect_counter;
                    if (pair.first->is_contain_label_frame_num(_current_label_frame_num)) {
                        #pragma omp atomic
                        ++trajectory_label_frame_num_counter;
                    }
                }
            }
        }
    }
    // for new landmark addition
    // 1. add middle_map_new_box_map to detect which patch need to be added into the middle map
    for (int i = 0; i < _point_cloud_vec.size(); ++i) {
        for (int j = 0; j < _point_cloud_vec[i].size(); ++j) {
            new_box_map.insert(std::array<int, 2>{i, j});
        }
    }

    std::vector<std::vector<ScanDataAsso<BasicType>>> initial_scan_data_asso_vec(_point_cloud_vec.size());
    // for visualization
    std::vector<std::vector<std::pair<std::vector<double>, std::vector<double>>>> initial_query_bounding_box_vec(
        _point_cloud_vec.size());
    //            std::unordered_map<std::vector<int>, std::pair<std::vector<double>, std::vector<double>>,
    //                               Voxel2DHashFuncPrime>
    //                initial_query_bounding_box_vec;
    // detect whether has ground or seg pairs
    // reject invalid overlap pairs
    // 2. modify the following for loop to make sure middle map can work well
    int initial_counter = 0;
    for (int i = 0; i < _point_cloud_vec.size(); ++i) {
        initial_scan_data_asso_vec[i].resize(_point_cloud_vec[i].size(), std::array<int, 2>{-1, -1});
        initial_query_bounding_box_vec[i].resize(_point_cloud_vec[i].size());
        for (int j = 0; j < _point_cloud_vec[i].size(); ++j) {
            std::array<int, 2> patch_key_idx = {i, j};
            if (!_point_cloud_info.intersected_score_pairs_vec[i][j].empty()) {
                // add information to ScanDataAsso
                initial_scan_data_asso_vec[i][j].point_idx = {i, j};
                for (auto &pair : _point_cloud_info.intersected_score_pairs_vec[i][j]) {
                    // for new landmark addition
                    // NOTE: Raw cloud bounding boxes inside the map patches need to be removed from new_box_map
                    // NOTE: All intersected patches are going to be used for update
                    // NOTE: Parts of intersected patches used for pose esitmation
                    // NOTE: This has to use enlarged bounding box
                    if (curl::is_query_inside_map_box(_point_cloud_info.bounding_box_pair_vec[i][j].first,
                                                      _point_cloud_info.bounding_box_pair_vec[i][j].second,
                                                      pair.first->get_enlarged_lower_bound_w(),
                                                      pair.first->get_enlarged_upper_bound_w())) {
                        if (!curl_tracking_config_ptr->is_frame_to_frame) {
                            new_box_map.erase(patch_key_idx);
                        }
                    } else if (curl::IoU_surface_area(_point_cloud_info.bounding_box_pair_vec[i][j].first,
                                                      _point_cloud_info.bounding_box_pair_vec[i][j].second,
                                                      pair.first->get_enlarged_lower_bound_w(),
                                                      pair.first->get_enlarged_upper_bound_w()) >
                               highest_IoU_new_landmark_thres) {
                        // NOTE: if the IoU is larger than the threshold then remove this patch from new_box_map
                        if (!curl_tracking_config_ptr->is_frame_to_frame) {
                            new_box_map.erase(patch_key_idx);
                        }
                    }
                    // for visualization
                    initial_query_bounding_box_vec[i][j] = _point_cloud_info.bounding_box_pair_vec[i][j];
                    // add information to ScanDataAsso
                    initial_scan_data_asso_vec[i][j].patch_info_ptr_vec.push_back(pair.first);
                    // add information to patch_info_ptr
                    pair.first->data_asso.points_idx_vec.push_back(std::array<int, 2>{i, j}); // FIXME:
                    pair.first->data_asso.IoU_vec.push_back(pair.second);                     // FIXME:
                    ++initial_counter;
                    break; // because we need the largest IoU is enough
                }
            }
        }
    }
    // update this to avoid too many patches
    std::vector<std::vector<std::tuple<std::array<int, 2>, double, std::shared_ptr<PatchInfo<BasicType>>>>> ground_sets(
        _point_cloud_vec.size());
    std::vector<std::vector<std::tuple<std::array<int, 2>, double, std::shared_ptr<PatchInfo<BasicType>>>>> seg_sets(
        _point_cloud_vec.size());
    // update data-association
    int best_counter = 0;
    //            std::vector<double> entire_scores_ground;
    //            std::vector<double> entire_scores_seg;
    for (int i = 0; i < initial_scan_data_asso_vec.size(); ++i) {
        for (int j = 0; j < initial_scan_data_asso_vec[i].size(); ++j) {
            if (!initial_scan_data_asso_vec[i][j].patch_info_ptr_vec.empty()) {
                // NOTE: select the best pairs only
                std::vector<int> sorted_scan_data_asso_idx = initial_scan_data_asso_vec[i][j].sort_according_to_score();
                std::shared_ptr<PatchInfo<BasicType>> highest_score_patch_info_ptr =
                    initial_scan_data_asso_vec[i][j].patch_info_ptr_vec[sorted_scan_data_asso_idx[0]];
                std::vector<int> sorted_map_data_asso =
                    highest_score_patch_info_ptr->data_asso.sort_according_to_score();
                // if both are the best selection for each other then data-association successed
                if (initial_scan_data_asso_vec[i][j].point_idx ==
                    highest_score_patch_info_ptr->data_asso.points_idx_vec[sorted_map_data_asso[0]]) {
                    // use push_back to make sure no invalid scores inside
                    ++best_counter;
                    if (_point_cloud_info.is_ground_cloud_vec[i][j]) {
                        ground_sets[i].emplace_back(
                            std::array<int, 2>{i, j},
                            highest_score_patch_info_ptr->data_asso.IoU_vec[sorted_map_data_asso[0]],
                            highest_score_patch_info_ptr);
                        //                                entire_scores_ground.push_back(
                        // highest_score_patch_info_ptr->data_asso.IoU_vec[sorted_map_data_asso[0]]);
                    } else {
                        seg_sets[i].emplace_back(
                            std::array<int, 2>{i, j},
                            highest_score_patch_info_ptr->data_asso.IoU_vec[sorted_map_data_asso[0]],
                            highest_score_patch_info_ptr);
                        //                                entire_scores_seg.push_back(
                        // highest_score_patch_info_ptr->data_asso.IoU_vec[sorted_map_data_asso[0]]);
                    }
                }
            }
        }
    }
    // clear all associated information (important)
    for (auto &iter_a : initial_scan_data_asso_vec) {
        for (auto &iter_b : iter_a) {
            iter_b.clear();
        }
    }
    // sort seg_sets and ground_sets
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
    //            double ground_thres = 0;
    //            if (!entire_scores_ground.empty()) {
    //                ground_thres = curl::IQR_thres_with_bdy<double>(entire_scores_ground,
    // curl_tracking_config_ptr->ground_IQR_scale_factor,
    // curl_tracking_config_ptr->min_num_ground_pairs);
    //            }
    //            double seg_thres = 0;
    //            if (!entire_scores_seg.empty()) {
    //                seg_thres =
    //                    curl::IQR_thres_with_bdy<double>(entire_scores_seg,
    //                    curl_tracking_config_ptr->seg_IQR_scale_factor,
    //                                                     curl_tracking_config_ptr->min_num_seg_pairs);
    //            }

    //             control the number of pairs, patch with higher IoU gets higher priority
    //            std::vector<int> init_succeed_idx;
    //            std::vector<std::tuple<std::array<int, 2>, double, std::shared_ptr<PatchInfo<BasicType>>>>
    //                succeed_tuple_vec; // FIXME: Check whether this is useful
    // differentiate the update associations and pose optimization associations
    int true_ground_num = 0;
    int true_seg_num = 0;
    std::vector<int> pose_succeed_associations_idx;
    if (curl_tracking_config_ptr->max_region_ground_pairs != -1 &&
        curl_tracking_config_ptr->max_region_seg_pairs != -1) {
        pose_succeed_associations_idx.reserve(
            curl_tracking_config_ptr->region_width_elements * curl_tracking_config_ptr->region_width_elements *
                curl_tracking_config_ptr->max_region_ground_pairs +
            curl_tracking_config_ptr->region_width_elements * curl_tracking_config_ptr->region_width_elements *
                curl_tracking_config_ptr->max_region_seg_pairs);
    } else {
        pose_succeed_associations_idx.reserve(2000);
    }
    for (const auto &region : ground_sets) {
        int region_counter = 0;
        for (const auto &ground_set : region) {
            succeed_associations.emplace_back(_point_cloud_vec[std::get<0>(ground_set)[0]][std::get<0>(ground_set)[1]],
                                              std::get<2>(ground_set), std::get<1>(ground_set));
            // if (std::get<2>(ground_set)->keyframe_ptr->frame_idx != _last_keyframe_idx) {
            //     cleaned_succeed_associations.emplace_back(
            //         _point_cloud_vec[std::get<0>(ground_set)[0]][std::get<0>(ground_set)[1]],
            //         std::get<2>(ground_set), std::get<1>(ground_set));
            // }
            if (region_counter < curl_tracking_config_ptr->max_region_ground_pairs ||
                curl_tracking_config_ptr->max_region_ground_pairs == -1) {
                pose_succeed_associations_idx.push_back(succeed_associations.size() - 1);
                //                        if (std::get<1>(ground_set) > ground_thres) {
                // succeed_tuple_vec.push_back(ground_set);
                // pose_succeed_associations.emplace_back(
                //     _point_cloud_vec[std::get<0>(ground_set)[0]][std::get<0>(ground_set)[1]],
                //     std::get<2>(ground_set), std::get<1>(ground_set));
                // for visualization
                query_bounding_box_vec.emplace_back(
                    initial_query_bounding_box_vec[std::get<0>(ground_set)[0]][std::get<0>(ground_set)[1]]);
                aabb::AABB bounding_box = aabb::AABB(std::get<2>(ground_set)->get_enlarged_lower_bound_w(),
                                                     std::get<2>(ground_set)->get_enlarged_upper_bound_w());
                // spatial_hashing_ptr->get_AABB(->key);
                map_bounding_box_vec.emplace_back(bounding_box.lowerBound, bounding_box.upperBound);
                ++true_ground_num;
                ++region_counter;
                //                        }
            }
        }
    }
    for (const auto &region : seg_sets) {
        int region_counter = 0;
        for (const auto seg_set : region) {

            succeed_associations.emplace_back(_point_cloud_vec[std::get<0>(seg_set)[0]][std::get<0>(seg_set)[1]],
                                              std::get<2>(seg_set), std::get<1>(seg_set));
            // if (std::get<2>(seg_set)->keyframe_ptr->frame_idx != _last_keyframe_idx) {
            //     cleaned_succeed_associations.emplace_back(
            //         _point_cloud_vec[std::get<0>(seg_set)[0]][std::get<0>(seg_set)[1]], std::get<2>(seg_set),
            //         std::get<1>(seg_set));
            // }
            if (region_counter < curl_tracking_config_ptr->max_region_seg_pairs ||
                curl_tracking_config_ptr->max_region_seg_pairs == -1) {
                pose_succeed_associations_idx.push_back(succeed_associations.size() - 1);
                //                        if (std::get<1>(seg_set) > seg_thres) {
                //                        succeed_tuple_vec.push_back(seg_set);
                // pose_succeed_associations.emplace_back(
                //     _point_cloud_vec[std::get<0>(seg_set)[0]][std::get<0>(seg_set)[1]], std::get<2>(seg_set),
                //     std::get<1>(seg_set));
                // for visualization
                query_bounding_box_vec.emplace_back(
                    initial_query_bounding_box_vec[std::get<0>(seg_set)[0]][std::get<0>(seg_set)[1]]);
                // aabb::AABB bounding_box =
                //     spatial_hashing_ptr->get_AABB(std::get<2>(seg_set)->key);
                aabb::AABB bounding_box = aabb::AABB(std::get<2>(seg_set)->get_enlarged_lower_bound_w(),
                                                     std::get<2>(seg_set)->get_enlarged_upper_bound_w());
                map_bounding_box_vec.emplace_back(bounding_box.lowerBound, bounding_box.upperBound);
                ++true_seg_num;
                ++region_counter;
                //                        }
            }
        }
    }
    _number_patches += (true_ground_num + true_seg_num);

    if (true_ground_num + true_seg_num < 5) {
        return false;
        // T_j_1_j.setIdentity();
        // continue;
    }

    _is_add_trajectory_segment = false;
    _is_add_keyframe = false;
    if (trajectory_label_frame_num_counter < minimum_observation_num) {
        _is_add_trajectory_segment = true;
        _is_add_keyframe = true;
    }

    // add these into residual block
    // if patches number of problem_seg or problem_ground is too small, use one-step optimization
    std::vector<ceres::CostFunction *> cost_function_vec; // update the pyramid information for the cost function
    int largest_pyramid_depth = 0;
    for (const auto &pose_asso_idx : pose_succeed_associations_idx) {
        auto &succeed_asso = succeed_associations[pose_asso_idx];
        // int i = std::get<0>(succeed_tuple);
        std::shared_ptr<PatchInfo<BasicType>> patch_info_ptr = std::get<1>(succeed_asso);
        largest_pyramid_depth =
            std::max(patch_info_ptr->patch_procession_ptr->get_pyramid_depth(), largest_pyramid_depth);
        int opt_step = 1;
        if (patch_info_ptr->is_ground) {
            opt_step = curl_tracking_config_ptr->ground_opt_step;
        }
        double weight = std::get<2>(succeed_asso);
        // add pairs into residual block
        if (patch_info_ptr->patch_procession_ptr->get_pyramid_depth() == 0 ||
            patch_info_ptr->patch_procession_ptr->get_sph_coeff_data() == nullptr ||
            patch_info_ptr->patch_procession_ptr->get_sph_coeff_sum() == 0) {
            continue;
        }
        Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_o_w =
            patch_info_ptr->T_obj_lidar.matrix() *
            Eigen::Isometry3d(patch_info_ptr->keyframe_ptr->get_T_w_lidar()).inverse().matrix();
        // Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_o_j = T_o_w * _T_w_j;
        Eigen::MatrixX<BasicType> measured_cloud;
        if (!is_voxel_grid_filter) {
            if (!std::get<1>(succeed_asso)->is_ground) {
                measured_cloud = curl::subSampleFrame<BasicType>(std::get<0>(succeed_asso), leaf_size);
            } else {
                measured_cloud = curl::subSampleFrame<BasicType>(std::get<0>(succeed_asso), leaf_size * 2);
            }
        } else {
            measured_cloud = std::get<0>(succeed_asso);
        }
        for (int k = 0; k < measured_cloud.cols(); k += opt_step) {
            Eigen::Vector3d p_j = (measured_cloud.col(k)).template cast<double>();
            ceres::CostFunction *couple_fix_sph_cost_func;
            couple_fix_sph_cost_func = CoupleFixSphDirectFactor<BasicType>::Create(
                T_o_w, p_j, patch_info_ptr->patch_procession_ptr, SH_table_config_ptr,
                curl_tracking_config_ptr->seg_max_valid_residual);
            if (curl_tracking_config_ptr->is_robust_kernel) {
                problem.AddResidualBlock(couple_fix_sph_cost_func, loss_function, _T_w_j.data());
            } else {
                problem.AddResidualBlock(couple_fix_sph_cost_func, nullptr, _T_w_j.data());
            }
            cost_function_vec.push_back(couple_fix_sph_cost_func);
        }
        // update data-association information for update sph coeff and visualization
        // succeed_idx.push_back(i);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = curl_tracking_config_ptr->max_num_iterations;
    // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;
    options.update_state_every_iteration = curl_tracking_config_ptr->update_state_every_iteration;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.use_nonmonotonic_steps = false;
    options.num_threads = curl_tracking_config_ptr->opt_thread_num;

    // if (curl_tracking_config_ptr->is_stop_after_cost_increased) {
    //     options.callbacks.push_back(&stop_callback);
    // }
    ceres::Solver::Summary summary;
    if (SH_table_config_ptr->is_SH_analytic_jacobian) {
        problem.SetManifold(_T_w_j.data(), SE3_manifold);
        ceres::Solve(options, &problem, &summary);
    } else {
        for (int pyramid_idx = largest_pyramid_depth; pyramid_idx > 0; --pyramid_idx) {
            for (const auto &cost_functor : cost_function_vec) {
                static_cast<CoupleFixSphDirectFactor<BasicType> *>(cost_functor)->update_pyramid_idx();
            }
            problem.SetManifold(_T_w_j.data(), SE3_manifold);
            ceres::Solve(options, &problem, &summary);
        }
    }
    // normalize T_w_j
    Eigen::Matrix3d R_tmp = _T_w_j(Eigen::seq(0, 2), Eigen::seq(0, 2));
    _T_w_j(Eigen::seq(0, 2), Eigen::seq(0, 2)) = Eigen::Quaterniond(R_tmp).normalized().toRotationMatrix();
    // judge whether need the new keyframe
    // Judgement of adding new keyframe
    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_j_j_keyframe =
        Eigen::Isometry3d(_T_w_j).inverse().matrix() * _T_w_j_keyframe;
    double distance_to_last_keyframe = T_j_j_keyframe(Eigen::seq(0, 2), 3).squaredNorm();
    double radian_to_last_keyframe = std::acos(
        std::max(std::min(0.5 * (T_j_j_keyframe(Eigen::seq(0, 2), Eigen::seq(0, 2)).trace() - 1), 1.0), -1.0));
    if (distance_to_last_keyframe > curl_voxel_mapping_config_ptr->keyframe_mini_squared_dis ||
        radian_to_last_keyframe > curl_voxel_mapping_config_ptr->keyframe_radian_change ||
        (_our_costs.size() > 1 && (curl_voxel_mapping_config_ptr->keyframe_IQR_cost_thres != -1) &&
         summary.final_cost >
             curl::IQR_thres<BasicType>(_our_costs, curl_voxel_mapping_config_ptr->keyframe_IQR_cost_thres))) {
        _is_add_keyframe = true;
    }
    _our_costs.push_back(summary.final_cost);
    return true;
}

template <typename BasicType>
bool CurlVoxelMapping<BasicType>::curl_registration_method_minimum(
    const pcl::PointCloud<PointT>::Ptr &seg_cloud_ptr, const pcl::PointCloud<PointT>::Ptr &ground_cloud_ptr,
    const bool is_voxel_grid_filter, const double leaf_size, const std::unordered_set<PatchId> &associated_patches,
    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_w_j) {
    if (associated_patches.empty()) {
        return false;
    }
    std::vector<std::vector<Eigen::MatrixX<BasicType>>> _point_cloud_vec;
    PointCloudInfo<BasicType> _point_cloud_info;
    std::vector<std::tuple<Eigen::MatrixX<BasicType>, std::shared_ptr<PatchInfo<BasicType>>, double>>
        pose_succeed_associations;

    preprocessing(0, seg_cloud_ptr, ground_cloud_ptr, _T_w_j.cast<float>(), _point_cloud_vec, _point_cloud_info);
    //            _point_cloud_info.T_obj_lidar_vec.resize(_point_cloud_vec.size());
    _point_cloud_info.intersected_score_pairs_vec.resize(_point_cloud_vec.size());
    _point_cloud_info.bounding_box_pair_vec.resize(_point_cloud_vec.size());

    ceres::Problem problem;

    ceres::Manifold *SE3_manifold = SE3Manifold::Create();
    ceres::LossFunctionWrapper *loss_function = new ceres::LossFunctionWrapper(
        new ceres::HuberLoss(curl_tracking_config_ptr->seg_kernel_threshold), ceres::DO_NOT_TAKE_OWNERSHIP);

    // new data-association starts here
    //            std::vector<std::vector<double>> entire_scores(_point_cloud_vec.size());
    int intersect_counter = 0;
    const Eigen::Matrix<BasicType, 3, 3> R_w_j =
        _T_w_j(Eigen::seq(0, 2), Eigen::seq(0, 2)).template cast<BasicType>();
    const Eigen::Matrix<BasicType, 3, 1> t_w_j = _T_w_j(Eigen::seq(0, 2), 3).template cast<BasicType>();
#pragma omp parallel for num_threads(curl_voxel_mapping_config_ptr->data_association_thread_num) default(none)         \
    shared(_point_cloud_vec, _point_cloud_info, _T_w_j, intersect_counter, associated_patches, R_w_j, t_w_j)
    for (int i = 0; i < _point_cloud_vec.size(); ++i) { // NOTE: each region
        _point_cloud_info.intersected_score_pairs_vec[i].resize(_point_cloud_vec[i].size());
        _point_cloud_info.bounding_box_pair_vec[i].resize(_point_cloud_vec[i].size());
        for (int j = 0; j < _point_cloud_vec[i].size(); ++j) { // NOTE: elements of region
            // TODO: this part has been done in function preprocessing, which can merge these two together to
            // speed-up
            const Eigen::MatrixX<BasicType> &patch_cloud = _point_cloud_vec[i][j];
            Eigen::Matrix<BasicType, 3, 1> min_pt;
            Eigen::Matrix<BasicType, 3, 1> max_pt;
            min_pt.setConstant(std::numeric_limits<BasicType>::max());
            max_pt.setConstant(std::numeric_limits<BasicType>::lowest());
            for (int c = 0; c < patch_cloud.cols(); ++c) {
                const Eigen::Matrix<BasicType, 3, 1> p = R_w_j * patch_cloud.col(c) + t_w_j;
                min_pt = min_pt.cwiseMin(p);
                max_pt = max_pt.cwiseMax(p);
            }

            std::vector<double> lower_bound_w(3);
            lower_bound_w[0] = static_cast<double>(min_pt(0));
            lower_bound_w[1] = static_cast<double>(min_pt(1));
            lower_bound_w[2] = static_cast<double>(min_pt(2));
            std::vector<double> upper_bound_w(3);
            upper_bound_w[0] = static_cast<double>(max_pt(0));
            upper_bound_w[1] = static_cast<double>(max_pt(1));
            upper_bound_w[2] = static_cast<double>(max_pt(2));
            _point_cloud_info.bounding_box_pair_vec[i][j] = std::make_pair(lower_bound_w, upper_bound_w);
            std::vector<std::pair<std::shared_ptr<PatchInfo<BasicType>>, double>> intersected_pairs =
                spatial_hashing_ptr->query_data_association_overlap(lower_bound_w, upper_bound_w);

            for (const auto &pair : intersected_pairs) {
                PatchId patch_id = pair.first->key;
                if (pair.first->is_ground != _point_cloud_info.is_ground_cloud_vec[i][j] ||
                    associated_patches.find(patch_id) == associated_patches.end()) {
                    continue;
                }
                _point_cloud_info.intersected_score_pairs_vec[i][j].emplace_back(pair.first, pair.second);
                #pragma omp atomic
                ++intersect_counter;
            }
        }
    }

    std::vector<std::vector<ScanDataAsso<BasicType>>> initial_scan_data_asso_vec(_point_cloud_vec.size());
    // for visualization
    std::vector<std::vector<std::pair<std::vector<double>, std::vector<double>>>> initial_query_bounding_box_vec(
        _point_cloud_vec.size());
    // 2. modify the following for loop to make sure middle map can work well
    int initial_counter = 0;
    for (int i = 0; i < _point_cloud_vec.size(); ++i) {
        initial_scan_data_asso_vec[i].resize(_point_cloud_vec[i].size(), std::array<int, 2>{-1, -1});
        initial_query_bounding_box_vec[i].resize(_point_cloud_vec[i].size());
        for (int j = 0; j < _point_cloud_vec[i].size(); ++j) {
            std::array<int, 2> patch_key_idx = {i, j};
            if (!_point_cloud_info.intersected_score_pairs_vec[i][j].empty()) {
                // add information to ScanDataAsso
                initial_scan_data_asso_vec[i][j].point_idx = {i, j};
                for (auto &pair : _point_cloud_info.intersected_score_pairs_vec[i][j]) {
                    // for visualization
                    initial_query_bounding_box_vec[i][j] = _point_cloud_info.bounding_box_pair_vec[i][j];
                    // add information to ScanDataAsso
                    initial_scan_data_asso_vec[i][j].patch_info_ptr_vec.push_back(pair.first);
                    // add information to patch_info_ptr
                    pair.first->data_asso.points_idx_vec.push_back(std::array<int, 2>{i, j}); // FIXME:
                    pair.first->data_asso.IoU_vec.push_back(pair.second);                     // FIXME:
                    ++initial_counter;
                    break; // because we need the largest IoU is enough
                }
            }
        }
    }
    // update this to avoid too many patches
    std::vector<std::vector<std::tuple<std::array<int, 2>, double, std::shared_ptr<PatchInfo<BasicType>>>>> ground_sets(
        _point_cloud_vec.size());
    std::vector<std::vector<std::tuple<std::array<int, 2>, double, std::shared_ptr<PatchInfo<BasicType>>>>> seg_sets(
        _point_cloud_vec.size());
    // update data-association
    // FIXME: check the correctness of the entire_scores
    int best_counter = 0;
    //            std::vector<double> entire_scores_ground;
    //            std::vector<double> entire_scores_seg;
    for (int i = 0; i < initial_scan_data_asso_vec.size(); ++i) {
        for (int j = 0; j < initial_scan_data_asso_vec[i].size(); ++j) {
            if (!initial_scan_data_asso_vec[i][j].patch_info_ptr_vec.empty()) {
                // NOTE: select the best pairs only
                std::vector<int> sorted_scan_data_asso_idx = initial_scan_data_asso_vec[i][j].sort_according_to_score();
                std::shared_ptr<PatchInfo<BasicType>> highest_score_patch_info_ptr =
                    initial_scan_data_asso_vec[i][j].patch_info_ptr_vec[sorted_scan_data_asso_idx[0]];
                std::vector<int> sorted_map_data_asso =
                    highest_score_patch_info_ptr->data_asso.sort_according_to_score();
                // if both are the best selection for each other then data-association successed
                if (initial_scan_data_asso_vec[i][j].point_idx ==
                    highest_score_patch_info_ptr->data_asso.points_idx_vec[sorted_map_data_asso[0]]) {
                    // use push_back to make sure no invalid scores inside
                    ++best_counter;
                    //                            entire_scores_region[i].push_back(
                    // highest_score_patch_info_ptr->data_asso.IoU_vec[sorted_map_data_asso[0]]);
                    // update data-association information for update sph coeff and visualization
                    if (_point_cloud_info.is_ground_cloud_vec[i][j]) {
                        ground_sets[i].emplace_back(
                            std::array<int, 2>{i, j},
                            highest_score_patch_info_ptr->data_asso.IoU_vec[sorted_map_data_asso[0]],
                            highest_score_patch_info_ptr);
                        //                                entire_scores_ground.push_back(
                        // highest_score_patch_info_ptr->data_asso.IoU_vec[sorted_map_data_asso[0]]);
                    } else {
                        seg_sets[i].emplace_back(
                            std::array<int, 2>{i, j},
                            highest_score_patch_info_ptr->data_asso.IoU_vec[sorted_map_data_asso[0]],
                            highest_score_patch_info_ptr);
                        //                                entire_scores_seg.push_back(
                        // highest_score_patch_info_ptr->data_asso.IoU_vec[sorted_map_data_asso[0]]);
                    }
                }
            }
        }
    }
    // clear all associated information (important)
    for (auto &iter_a : initial_scan_data_asso_vec) {
        for (auto &iter_b : iter_a) {
            iter_b.clear();
        }
    }
    // sort seg_sets and ground_sets
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
    int true_ground_num = 0;
    int true_seg_num = 0;
    for (const auto &region : ground_sets) {
        int region_counter = 0;
        for (const auto &ground_set : region) {
            if (region_counter < curl_tracking_config_ptr->max_region_ground_pairs ||
                curl_tracking_config_ptr->max_region_ground_pairs == -1) {
                pose_succeed_associations.emplace_back(
                    _point_cloud_vec[std::get<0>(ground_set)[0]][std::get<0>(ground_set)[1]], std::get<2>(ground_set),
                    std::get<1>(ground_set));
                aabb::AABB bounding_box = aabb::AABB(std::get<2>(ground_set)->get_enlarged_lower_bound_w(),
                                                     std::get<2>(ground_set)->get_enlarged_upper_bound_w());
                ++true_ground_num;
                ++region_counter;
                //                        }
            }
        }
    }
    for (const auto &region : seg_sets) {
        int region_counter = 0;
        for (const auto seg_set : region) {
            if (region_counter < curl_tracking_config_ptr->max_region_seg_pairs ||
                curl_tracking_config_ptr->max_region_seg_pairs == -1) {
                pose_succeed_associations.emplace_back(
                    _point_cloud_vec[std::get<0>(seg_set)[0]][std::get<0>(seg_set)[1]], std::get<2>(seg_set),
                    std::get<1>(seg_set));
                aabb::AABB bounding_box = aabb::AABB(std::get<2>(seg_set)->get_enlarged_lower_bound_w(),
                                                     std::get<2>(seg_set)->get_enlarged_upper_bound_w());
                ++true_seg_num;
                ++region_counter;
                //                        }
            }
        }
    }
    if (true_ground_num + true_seg_num < 5) {
        return false;
    }

    // add these into residual block
    // if patches number of problem_seg or problem_ground is too small, use one-step optimization
    std::vector<ceres::CostFunction *> cost_function_vec; // update the pyramid information for the cost function
    int largest_pyramid_depth = 0;
    for (auto &succeed_asso : pose_succeed_associations) {
        // TODO: save all the patches reconstructed cloud into world coordinate
        // int i = std::get<0>(succeed_tuple);
        std::shared_ptr<PatchInfo<BasicType>> patch_info_ptr = std::get<1>(succeed_asso);
        largest_pyramid_depth =
            std::max(patch_info_ptr->patch_procession_ptr->get_pyramid_depth(), largest_pyramid_depth);
        int opt_step = 1;
        if (patch_info_ptr->is_ground) {
            opt_step = curl_tracking_config_ptr->ground_opt_step;
        }
        double weight = std::get<2>(succeed_asso);
        // add pairs into residual block
        if (patch_info_ptr->patch_procession_ptr->get_pyramid_depth() == 0 ||
            patch_info_ptr->patch_procession_ptr->get_sph_coeff_data() == nullptr ||
            patch_info_ptr->patch_procession_ptr->get_sph_coeff_sum() == 0) {
            continue;
        }
        Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_o_w =
            patch_info_ptr->T_obj_lidar.matrix() *
            Eigen::Isometry3d(patch_info_ptr->keyframe_ptr->get_T_w_lidar()).inverse().matrix();
        // Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_o_j = T_o_w * _T_w_j;
        Eigen::MatrixX<BasicType> measured_cloud;
        if (!is_voxel_grid_filter) {
            if (!std::get<1>(succeed_asso)->is_ground) {
                measured_cloud = curl::subSampleFrame<BasicType>(std::get<0>(succeed_asso), leaf_size);
            } else {
                measured_cloud = curl::subSampleFrame<BasicType>(std::get<0>(succeed_asso), leaf_size * 2);
            }
        } else {
            measured_cloud = std::get<0>(succeed_asso);
        }
        for (int k = 0; k < measured_cloud.cols(); k += opt_step) {
            Eigen::Vector3d p_j = (measured_cloud.col(k)).template cast<double>();
            ceres::CostFunction *couple_fix_sph_cost_func;
            couple_fix_sph_cost_func = CoupleFixSphDirectFactor<BasicType>::Create(
                T_o_w, p_j, patch_info_ptr->patch_procession_ptr, SH_table_config_ptr,
                curl_tracking_config_ptr->seg_max_valid_residual);
            if (curl_tracking_config_ptr->is_robust_kernel) {
                problem.AddResidualBlock(couple_fix_sph_cost_func, loss_function, _T_w_j.data());
            } else {
                problem.AddResidualBlock(couple_fix_sph_cost_func, nullptr, _T_w_j.data());
            }
            cost_function_vec.push_back(couple_fix_sph_cost_func);
        }
        // update data-association information for update sph coeff and visualization
        // succeed_idx.push_back(i);
    }
    ceres::Solver::Options options;
    options.max_num_iterations = curl_tracking_config_ptr->max_num_iterations;
    // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;
    options.update_state_every_iteration = curl_tracking_config_ptr->update_state_every_iteration;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.use_nonmonotonic_steps = false;
    options.num_threads = curl_tracking_config_ptr->opt_thread_num;

    ceres::Solver::Summary summary;
    if (SH_table_config_ptr->is_SH_analytic_jacobian) {
        problem.SetManifold(_T_w_j.data(), SE3_manifold);
        ceres::Solve(options, &problem, &summary);
    } else {
        for (int pyramid_idx = largest_pyramid_depth; pyramid_idx > 0; --pyramid_idx) {
            for (const auto &cost_functor : cost_function_vec) {
                static_cast<CoupleFixSphDirectFactor<BasicType> *>(cost_functor)->update_pyramid_idx();
            }
            problem.SetManifold(_T_w_j.data(), SE3_manifold);
            ceres::Solve(options, &problem, &summary);
        }
    }

    // normalize T_w_j
    Eigen::Matrix3d R_tmp = _T_w_j(Eigen::seq(0, 2), Eigen::seq(0, 2));
    _T_w_j(Eigen::seq(0, 2), Eigen::seq(0, 2)) = Eigen::Quaterniond(R_tmp).normalized().toRotationMatrix();
    return true;
}

template class CurlVoxelMapping<BT>; // this is very important
