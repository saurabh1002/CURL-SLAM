/**
 * @file load_config.h
 * @author zkc (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-09-11
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef EXAMPLE_LOAD_CONFIG
#define EXAMPLE_LOAD_CONFIG
#include "curl_slam/FileReaderBase.h"
#include "yaml-cpp/yaml.h"
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

struct FILE_READER_CONFIG {
    std::string data_set;
    std::string root_dir;
    std::string sequence;
    std::string scans_dir;
    double scan_scale_factor;
    std::string poses_dir;
    std::string semantics_dir;
    int start_frame;
    int end_frame;
    std::string scan_file_type;
    std::string semantic_file_type;
    std::string poses_file_type;
    bool is_debug = false;
    std::string transformation_obj_LiDAR_dir;
    std::string conformal_map_dir;
    std::string gamma_gradient_dir;
    std::string valid_idx_dir;
    std::string sph_coeff_dir;
    std::string odometry_file;
    std::string odometry_noised_file;
    std::string trajectory_file;
    std::string trajectory_noised_file;
    int obj_num = 0;
    double gamma_rso;
};

struct DIRECT_METHOD_CONFIG {
    double minimum_rso;
    int minimum_img_rso;
    double mask_gap_squared;
    int update_initialization_times;
    int update_steps;
    double update_distance;
    bool is_average;
    int pyramid_minimum_img_rso;
};

template <typename T> struct SH_TABLE_CONFIG {
    // TODO: add fixed sph_dir for spherical harmonics coefficients extraction and update
    SH_TABLE_CONFIG() {
        gaussian_blur_kernel_5.resize(5, 5);
        gaussian_blur_kernel_5 << 1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36, 24, 6, 4, 16, 24, 16, 4, 1, 4, 6, 4, 1;
        gaussian_blur_kernel_5 /= 256;
        X_Jac << 1, 0;
        Y_Jac << 0, 1;
        Z_Jac << 0, 0, 1;
        XY_Jac << 1, 0, 0, 0, 1, 0;
    }
    bool is_speed_up;
    int max_SH_degree;
    int warping_SH_degree;
    int ground_SH_degree;
    T SH_scale;
    int azi_rso;
    int elev_rso;
    int init_SPH_pass;
    int init_SPH_granularity;
    std::string inv_method;
    T bad_thres;
    T medium_thres;

    int update_azi_rso;
    int update_elev_rso;
    T sampling_factor;
    bool is_SH_analytic_jacobian;

    Eigen::MatrixX<T> SH_table;
    Eigen::MatrixX<T> SH_G_theta_table;
    Eigen::MatrixX<T> SH_G_phi_table;
    Eigen::VectorXi update_idx;

    Eigen::MatrixX<T> I_table;
    Eigen::MatrixX<T> I_dirs; // height images' unified spherical coordinate
    Eigen::MatrixX<T> xy_grid;
    Eigen::MatrixX<T> I_table_square; // because we use IRF, this need to be calculated for each degree
    std::vector<Eigen::MatrixX<T>> I_table_square_vec;

    Eigen::LDLT<Eigen::MatrixX<T>> I_table_square_ldlt; // this is for update of the matrix
    // for IRF version of update, and this can be extended for any degree
    std::vector<Eigen::LDLT<Eigen::MatrixX<T>>> I_table_square_ldlt_vec;

    Eigen::MatrixX<T> gaussian_blur_kernel_5;

    Eigen::Matrix<double, 1, 2> X_Jac;
    Eigen::Matrix<double, 1, 2> Y_Jac;
    Eigen::Matrix<double, 1, 3> Z_Jac;
    Eigen::Matrix<double, 2, 3> XY_Jac;

  public:
    T get_azi_low() const { return (M_PI * (1 - SH_scale)); }
    T get_azi_high() const { return (2 * M_PI * SH_scale + get_azi_low()); }
    T get_elev_low() const { return (M_PI / 2 * (1 - SH_scale)); }
    T get_elev_high() const { return (M_PI * SH_scale + get_elev_low()); }
};

struct DEBUG_CONFIG {
    std::string results_dir;
    std::string data_set;
    std::string seq;
    bool is_save_final_results;
    bool is_pub_dense_reconstruction;
    bool is_save_final_map;
    Eigen::MatrixXf color_map;
};

struct AABB_CONFIG {
    double extra_skin_thickness;
};

struct CURL_TRACKING_CONFIG {
    double frequency;
    bool is_frame_to_frame;
    int map_update_thread_num;
    int region_width_elements;
    int max_region_seg_pairs;
    int max_region_ground_pairs;
    int local_window_size = -1; // -1 means unlimited
    int max_num_iterations;
    bool update_state_every_iteration;
    bool is_robust_kernel;
    double seg_kernel_threshold;
    double seg_max_valid_residual;
    int opt_thread_num;
    int ground_opt_step;
    float color_change_step;
};

struct CURL_CONTINUOUS_RECONS_CONFIG {
    int patch_img_rso;
    int SH_degree;
    double IQR_factor;
    double length_thres;
    double open3d_speed;
    bool is_fix_length_threshold;
    bool is_save_pcd;
    bool is_save_curlmap;
    bool operator==(const CURL_CONTINUOUS_RECONS_CONFIG &other) const {

        return patch_img_rso == other.patch_img_rso && SH_degree == other.SH_degree && IQR_factor == other.IQR_factor &&
               length_thres == other.length_thres && is_fix_length_threshold == other.is_fix_length_threshold;
    }

    bool operator!=(const CURL_CONTINUOUS_RECONS_CONFIG &other) const { return !((*this) == other); }
};

struct CURL_VOXEL_MAPPING_CONFIG {
    int patch_initialization_thread_num;
    int data_association_thread_num;
    int overlap_plan_thread_num;
    int PGO_thread_num;
    int BA_thread_num;
    int BA_recons_downsample_stride = 1;
    double minimum_squared_dis;
    double maximum_squared_dis;
    double max_patch_search_box_times;
    bool is_use_all_associated_patches;
    bool is_kitti_correct;
    bool is_deskew;
    int minimum_observation_num;
    bool is_seg_only;
    double cut_threshold;
    double half_diag_cut_threshold;
    int patch_minimun_pts_num_after_filter;
    bool is_voxel_grid_filter;
    double leaf_size;
    double keyframe_mini_squared_dis;
    double keyframe_radian_change;
    double keyframe_IQR_cost_thres;
    double points_regularizer_weight_square_root;
    double max_residual;
    std::string opt_status;
    int max_num_iterations;
    double function_tolerance;
    int minimum_points_to_fix_projection_direction;
    int minimum_points_to_add_new_patch_seg;
    int minimum_points_to_add_new_patch_ground;
    double highest_IoU_new_landmark_thres;
    double largest_distance_squared;
};

struct CURL_LOOP_CLOSURE_CONFIG {
    bool is_enable;
    double search_radius_squared;
    double simple_loop_closure_search_radius_squared;
    double remove_overlap_keyframe_patches_region_squared;
    double remove_overlap_iou_thres = 0.1;
    //    int valid_history_frame_idx_region;
    //    double frame_diff;
    double minimum_middle_segment_valid_dis_squared;
    double association_rate_for_opt_history_keyframe;
    int region_width_elements;
};

void load_config_file(const std::string &dir, FILE_READER_CONFIG &file_reader_config) {
    YAML::Node config = YAML::LoadFile(dir);
    file_reader_config.data_set = config["DATA_SET"].as<decltype(file_reader_config.data_set)>();
    file_reader_config.root_dir =
        config[file_reader_config.data_set]["root_dir"].as<decltype(file_reader_config.root_dir)>();
    file_reader_config.sequence =
        config[file_reader_config.data_set]["sequence"].as<decltype(file_reader_config.sequence)>();
    file_reader_config.scans_dir =
        config[file_reader_config.data_set]["scans_dir"].as<decltype(file_reader_config.scans_dir)>();
    file_reader_config.scan_scale_factor =
        config[file_reader_config.data_set]["scan_scale_factor"].as<decltype(file_reader_config.scan_scale_factor)>();
    if (config[file_reader_config.data_set]["poses_dir"]) {
        file_reader_config.poses_dir =
            config[file_reader_config.data_set]["poses_dir"].as<decltype(file_reader_config.poses_dir)>();
    }

    file_reader_config.semantics_dir =
        config[file_reader_config.data_set]["semantics_dir"].as<decltype(file_reader_config.semantics_dir)>();
    file_reader_config.start_frame =
        config[file_reader_config.data_set]["start_frame"].as<decltype(file_reader_config.start_frame)>();
    file_reader_config.end_frame =
        config[file_reader_config.data_set]["end_frame"].as<decltype(file_reader_config.end_frame)>();
    file_reader_config.scan_file_type =
        config[file_reader_config.data_set]["scan_file_type"].as<decltype(file_reader_config.scan_file_type)>();
    file_reader_config.semantic_file_type =
        config[file_reader_config.data_set]["semantic_file_type"].as<decltype(file_reader_config.semantic_file_type)>();
    if (config[file_reader_config.data_set]["poses_file_type"]) {
        file_reader_config.poses_file_type =
            config[file_reader_config.data_set]["poses_file_type"].as<decltype(file_reader_config.poses_file_type)>();
    }
    if (config[file_reader_config.data_set]["is_debug"]) {
        file_reader_config.is_debug =
            config[file_reader_config.data_set]["is_debug"].as<decltype(file_reader_config.is_debug)>();
    }
    if (config[file_reader_config.data_set]["transformation_obj_LiDAR_dir"]) {
        file_reader_config.transformation_obj_LiDAR_dir =
            config[file_reader_config.data_set]["transformation_obj_LiDAR_dir"]
                .as<decltype(file_reader_config.transformation_obj_LiDAR_dir)>();
    }
    if (config[file_reader_config.data_set]["conformal_map_dir"]) {
        file_reader_config.conformal_map_dir = config[file_reader_config.data_set]["conformal_map_dir"]
                                                   .as<decltype(file_reader_config.conformal_map_dir)>();
    }
    if (config[file_reader_config.data_set]["gamma_gradient_dir"]) {
        file_reader_config.gamma_gradient_dir = config[file_reader_config.data_set]["gamma_gradient_dir"]
                                                    .as<decltype(file_reader_config.gamma_gradient_dir)>();
    }
    if (config[file_reader_config.data_set]["valid_idx_dir"]) { // valid_idx_dir
        file_reader_config.valid_idx_dir =
            config[file_reader_config.data_set]["valid_idx_dir"].as<decltype(file_reader_config.valid_idx_dir)>();
    }
    if (config[file_reader_config.data_set]["sph_coeff_dir"]) {
        file_reader_config.sph_coeff_dir =
            config[file_reader_config.data_set]["sph_coeff_dir"].as<decltype(file_reader_config.sph_coeff_dir)>();
    }
    if (config[file_reader_config.data_set]["odometry_file"]) {
        file_reader_config.odometry_file =
            config[file_reader_config.data_set]["odometry_file"].as<decltype(file_reader_config.odometry_file)>();
    }
    if (config[file_reader_config.data_set]["odometry_noised_file"]) {
        file_reader_config.odometry_noised_file = config[file_reader_config.data_set]["odometry_noised_file"]
                                                      .as<decltype(file_reader_config.odometry_noised_file)>();
    }
    if (config[file_reader_config.data_set]["trajectory_file"]) {
        file_reader_config.trajectory_file =
            config[file_reader_config.data_set]["trajectory_file"].as<decltype(file_reader_config.trajectory_file)>();
    }
    if (config[file_reader_config.data_set]["trajectory_noised_file"]) {
        file_reader_config.trajectory_noised_file = config[file_reader_config.data_set]["trajectory_noised_file"]
                                                        .as<decltype(file_reader_config.trajectory_noised_file)>();
    }
    if (config[file_reader_config.data_set]["obj_num"]) {
        file_reader_config.obj_num =
            config[file_reader_config.data_set]["obj_num"].as<decltype(file_reader_config.obj_num)>();
    }
    if (config[file_reader_config.data_set]["gamma_rso"]) {
        file_reader_config.gamma_rso =
            config[file_reader_config.data_set]["gamma_rso"].as<decltype(file_reader_config.gamma_rso)>();
    }
}

void load_direct_method_config(const std::string &dir, DIRECT_METHOD_CONFIG &direct_method_config) {
    YAML::Node config = YAML::LoadFile(dir);
    if (config["DIRECT_METHOD"]["CURL_ODOMETRY"]["w"]) {
        int w = config["DIRECT_METHOD"]["CURL_ODOMETRY"]["w"].as<int>();
        if (config["SphVoxelUpdate"]["minimum_dis"] && w > 0) {
            double minimum_dis = config["SphVoxelUpdate"]["minimum_dis"].as<double>();
            if (minimum_dis > 0) {
                direct_method_config.minimum_rso = minimum_dis / static_cast<double>(w);
            }
        }
    } else if (config["DIRECT_METHOD"]["CURL_ODOMETRY"]["minimum_rso"]) {
        direct_method_config.minimum_rso =
            config["DIRECT_METHOD"]["CURL_ODOMETRY"]["minimum_rso"].as<decltype(direct_method_config.minimum_rso)>();
    }
    if (config["DIRECT_METHOD"]["CURL_ODOMETRY"]["minimum_img_rso"]) {
        direct_method_config.minimum_img_rso = config["DIRECT_METHOD"]["CURL_ODOMETRY"]["minimum_img_rso"]
                                                   .as<decltype(direct_method_config.minimum_img_rso)>();
    }
    if (config["DIRECT_METHOD"]["CURL_ODOMETRY"]["mask_gap"]) {
        double mask_gap = config["DIRECT_METHOD"]["CURL_ODOMETRY"]["mask_gap"].as<double>();
        direct_method_config.mask_gap_squared = mask_gap * mask_gap;
    }
    if (config["DIRECT_METHOD"]["CURL_ODOMETRY"]["update_initialization_times"]) {
        direct_method_config.update_initialization_times =
            config["DIRECT_METHOD"]["CURL_ODOMETRY"]["update_initialization_times"]
                .as<decltype(direct_method_config.update_initialization_times)>();
    }
    if (config["DIRECT_METHOD"]["CURL_ODOMETRY"]["update_steps"]) {
        direct_method_config.update_steps =
            config["DIRECT_METHOD"]["CURL_ODOMETRY"]["update_steps"].as<decltype(direct_method_config.update_steps)>();
    }
    if (config["DIRECT_METHOD"]["CURL_ODOMETRY"]["update_distance"]) {
        direct_method_config.update_distance = config["DIRECT_METHOD"]["CURL_ODOMETRY"]["update_distance"]
                                                   .as<decltype(direct_method_config.update_distance)>();
    }
    if (config["DIRECT_METHOD"]["BA"]["is_average"]) {
        direct_method_config.is_average =
            config["DIRECT_METHOD"]["BA"]["is_average"].as<decltype(direct_method_config.is_average)>();
    }
    if (config["DIRECT_METHOD"]["CURL_ODOMETRY"]["pyramid_minimum_img_rso"]) {
        direct_method_config.pyramid_minimum_img_rso =
            config["DIRECT_METHOD"]["CURL_ODOMETRY"]["pyramid_minimum_img_rso"]
                .as<decltype(direct_method_config.pyramid_minimum_img_rso)>();
    }
}
template <typename T> void load_SH_table_config(const std::string &dir, SH_TABLE_CONFIG<T> &SH_table_config) {
    YAML::Node config = YAML::LoadFile(dir);
    SH_table_config.is_speed_up = config["SH_TABLE"]["is_speed_up"].as<decltype(SH_table_config.is_speed_up)>();
    SH_table_config.max_SH_degree = config["SH_TABLE"]["max_SH_degree"].as<decltype(SH_table_config.max_SH_degree)>();
    SH_table_config.warping_SH_degree =
        config["SH_TABLE"]["warping_SH_degree"].as<decltype(SH_table_config.warping_SH_degree)>();
    if (config["SH_TABLE"]["ground_SH_degree"]) {
        SH_table_config.ground_SH_degree =
            config["SH_TABLE"]["ground_SH_degree"].as<decltype(SH_table_config.ground_SH_degree)>();
    }
    SH_table_config.SH_scale = config["SH_TABLE"]["SH_scale"].as<decltype(SH_table_config.SH_scale)>();
    SH_table_config.azi_rso = config["SH_TABLE"]["azi_rso"].as<decltype(SH_table_config.azi_rso)>();
    SH_table_config.elev_rso = config["SH_TABLE"]["elev_rso"].as<decltype(SH_table_config.elev_rso)>();
    SH_table_config.init_SPH_pass = config["SH_TABLE"]["init_SPH_pass"].as<decltype(SH_table_config.init_SPH_pass)>();
    SH_table_config.init_SPH_granularity =
        config["SH_TABLE"]["init_SPH_granularity"].as<decltype(SH_table_config.init_SPH_granularity)>();
    SH_table_config.inv_method = config["SH_TABLE"]["inv_method"].as<decltype(SH_table_config.inv_method)>();
    SH_table_config.bad_thres = config["SH_TABLE"]["bad_thres"].as<decltype(SH_table_config.bad_thres)>();
    if (config["SH_TABLE"]["medium_thres"]) {
        SH_table_config.medium_thres = config["SH_TABLE"]["medium_thres"].as<decltype(SH_table_config.medium_thres)>();
    }
    if (config["SH_TABLE"]["sampling_factor"]) {
        SH_table_config.sampling_factor =
            config["SH_TABLE"]["sampling_factor"].as<decltype(SH_table_config.sampling_factor)>();
    }
    if (config["SH_TABLE"]["is_SH_analytic_jacobian"]) {
        SH_table_config.is_SH_analytic_jacobian =
            config["SH_TABLE"]["is_SH_analytic_jacobian"].as<decltype(SH_table_config.is_SH_analytic_jacobian)>();
    }
}

void load_debug_config(const std::string &dir, DEBUG_CONFIG &debug_config, const std::string seq = "") {
    YAML::Node config = YAML::LoadFile(dir);
    if (config["DEBUG"]["data_set"]) {
        debug_config.data_set = config["DEBUG"]["data_set"].as<decltype(debug_config.data_set)>();
    }
    if (config["DEBUG"]["seq"]) { // 0~10
        if (seq == "") {
            debug_config.seq = config["DEBUG"]["seq"].as<decltype(debug_config.seq)>();
        } else {
            debug_config.seq = seq;
        }
    }
    if (config["DEBUG"]["results_dir"]) {
        debug_config.results_dir =
            config["DEBUG"]["results_dir"].as<decltype(debug_config.results_dir)>() + "/" + debug_config.seq;
    }
    if (config["DEBUG"]["is_save_final_results"]) {
        debug_config.is_save_final_results =
            config["DEBUG"]["is_save_final_results"].as<decltype(debug_config.is_save_final_results)>();
    }
    if (config["DEBUG"]["PUB"]["is_pub_dense_reconstruction"]) {
        debug_config.is_pub_dense_reconstruction = config["DEBUG"]["PUB"]["is_pub_dense_reconstruction"]
                                                       .as<decltype(debug_config.is_pub_dense_reconstruction)>();
    }
    if (config["DEBUG"]["is_save_final_map"]) {
        debug_config.is_save_final_map =
            config["DEBUG"]["is_save_final_map"].as<decltype(debug_config.is_save_final_map)>();
    }
}

void load_aabb_config(const std::string &dir, AABB_CONFIG &aabb_config) {
    YAML::Node config = YAML::LoadFile(dir);
    if (config["AABB"]["extra_skin_thickness"]) {
        aabb_config.extra_skin_thickness =
            config["AABB"]["extra_skin_thickness"].as<decltype(aabb_config.extra_skin_thickness)>();
    }
}

void load_curl_tracking_config(const std::string &dir, CURL_TRACKING_CONFIG &curl_tracking_config) {
    YAML::Node config = YAML::LoadFile(dir);
    if (config["CURL_TRACKING"]["frequency"]) {
        curl_tracking_config.frequency =
            config["CURL_TRACKING"]["frequency"].as<decltype(curl_tracking_config.frequency)>();
    }
    if (config["CURL_TRACKING"]["is_frame_to_frame"]) {
        curl_tracking_config.is_frame_to_frame =
            config["CURL_TRACKING"]["is_frame_to_frame"].as<decltype(curl_tracking_config.is_frame_to_frame)>();
    }
    if (config["CURL_TRACKING"]["map_update_thread_num"]) {
        curl_tracking_config.map_update_thread_num =
            config["CURL_TRACKING"]["map_update_thread_num"].as<decltype(curl_tracking_config.map_update_thread_num)>();
    }
    if (config["CURL_TRACKING"]["DATA_ASSOCIATION"]["region_width_elements"]) {
        curl_tracking_config.region_width_elements =
            config["CURL_TRACKING"]["DATA_ASSOCIATION"]["region_width_elements"]
                .as<decltype(curl_tracking_config.region_width_elements)>();
    }
    if (config["CURL_TRACKING"]["DATA_ASSOCIATION"]["max_region_seg_pairs"]) {
        curl_tracking_config.max_region_seg_pairs = config["CURL_TRACKING"]["DATA_ASSOCIATION"]["max_region_seg_pairs"]
                                                        .as<decltype(curl_tracking_config.max_region_seg_pairs)>();
    }
    if (config["CURL_TRACKING"]["DATA_ASSOCIATION"]["max_region_ground_pairs"]) {
        curl_tracking_config.max_region_ground_pairs =
            config["CURL_TRACKING"]["DATA_ASSOCIATION"]["max_region_ground_pairs"]
                .as<decltype(curl_tracking_config.max_region_ground_pairs)>();
    }
    if (config["CURL_TRACKING"]["LOCAL_MAP"]["local_window_size"]) {
        curl_tracking_config.local_window_size =
            config["CURL_TRACKING"]["LOCAL_MAP"]["local_window_size"]
                .as<decltype(curl_tracking_config.local_window_size)>();
    }
    if (config["CURL_TRACKING"]["OPTIMIZATION"]["max_num_iterations"]) {
        curl_tracking_config.max_num_iterations = config["CURL_TRACKING"]["OPTIMIZATION"]["max_num_iterations"]
                                                      .as<decltype(curl_tracking_config.max_num_iterations)>();
    }
    if (config["CURL_TRACKING"]["OPTIMIZATION"]["update_state_every_iteration"]) {
        curl_tracking_config.update_state_every_iteration =
            config["CURL_TRACKING"]["OPTIMIZATION"]["update_state_every_iteration"]
                .as<decltype(curl_tracking_config.update_state_every_iteration)>();
    }
    if (config["CURL_TRACKING"]["OPTIMIZATION"]["is_robust_kernel"]) {
        curl_tracking_config.is_robust_kernel = config["CURL_TRACKING"]["OPTIMIZATION"]["is_robust_kernel"]
                                                    .as<decltype(curl_tracking_config.is_robust_kernel)>();
    }
    if (config["CURL_TRACKING"]["OPTIMIZATION"]["seg_kernel_threshold"]) {
        curl_tracking_config.seg_kernel_threshold = config["CURL_TRACKING"]["OPTIMIZATION"]["seg_kernel_threshold"]
                                                        .as<decltype(curl_tracking_config.seg_kernel_threshold)>();
    }
    if (config["CURL_TRACKING"]["OPTIMIZATION"]["seg_max_valid_residual"]) {
        curl_tracking_config.seg_max_valid_residual = config["CURL_TRACKING"]["OPTIMIZATION"]["seg_max_valid_residual"]
                                                          .as<decltype(curl_tracking_config.seg_max_valid_residual)>();
    }
    if (config["CURL_TRACKING"]["OPTIMIZATION"]["opt_thread_num"]) {
        curl_tracking_config.opt_thread_num = config["CURL_TRACKING"]["OPTIMIZATION"]["opt_thread_num"]
                                                  .as<decltype(curl_tracking_config.opt_thread_num)>();
    }
    if (config["CURL_TRACKING"]["OPTIMIZATION"]["ground_opt_step"]) {
        curl_tracking_config.ground_opt_step = config["CURL_TRACKING"]["OPTIMIZATION"]["ground_opt_step"]
                                                   .as<decltype(curl_tracking_config.ground_opt_step)>();
    }
    if (config["CURL_TRACKING"]["VISUALIZATION"]["color_change_step"]) {
        curl_tracking_config.color_change_step = config["CURL_TRACKING"]["VISUALIZATION"]["color_change_step"]
                                                     .as<decltype(curl_tracking_config.color_change_step)>();
    }
}

void load_curl_continuous_recons_config(const std::string &dir,
                                        CURL_CONTINUOUS_RECONS_CONFIG &curl_continuous_recons_config) {
    YAML::Node config = YAML::LoadFile(dir);
    if (config["CONTINUOUS_RECONS"]["w"]) {
        curl_continuous_recons_config.patch_img_rso =
            config["CONTINUOUS_RECONS"]["w"].as<decltype(curl_continuous_recons_config.patch_img_rso)>() +
            1;
    }
}

void load_curl_voxel_mapping_config(const std::string &dir, CURL_VOXEL_MAPPING_CONFIG &curl_voxel_update_config) {
    YAML::Node config = YAML::LoadFile(dir);
    if (config["SphVoxelUpdate"]["patch_initialization_thread_num"]) {
        curl_voxel_update_config.patch_initialization_thread_num =
            config["SphVoxelUpdate"]["patch_initialization_thread_num"]
                .as<decltype(curl_voxel_update_config.patch_initialization_thread_num)>();
    }
    if (config["SphVoxelUpdate"]["data_association_thread_num"]) {
        curl_voxel_update_config.data_association_thread_num =
            config["SphVoxelUpdate"]["data_association_thread_num"]
                .as<decltype(curl_voxel_update_config.data_association_thread_num)>();
    }
    if (config["SphVoxelUpdate"]["overlap_plan_thread_num"]) {
        curl_voxel_update_config.overlap_plan_thread_num =
            config["SphVoxelUpdate"]["overlap_plan_thread_num"]
                .as<decltype(curl_voxel_update_config.overlap_plan_thread_num)>();
    }
    if (config["SphVoxelUpdate"]["PGO_thread_num"]) {
        curl_voxel_update_config.PGO_thread_num =
            config["SphVoxelUpdate"]["PGO_thread_num"]
                .as<decltype(curl_voxel_update_config.PGO_thread_num)>();
    }
    if (config["SphVoxelUpdate"]["BA_thread_num"]) {
        curl_voxel_update_config.BA_thread_num =
            config["SphVoxelUpdate"]["BA_thread_num"].as<decltype(curl_voxel_update_config.BA_thread_num)>();
    }
    if (config["SphVoxelUpdate"]["BA_recons_downsample_stride"]) {
        curl_voxel_update_config.BA_recons_downsample_stride =
            config["SphVoxelUpdate"]["BA_recons_downsample_stride"]
                .as<decltype(curl_voxel_update_config.BA_recons_downsample_stride)>();
        if (curl_voxel_update_config.BA_recons_downsample_stride < 1) {
            curl_voxel_update_config.BA_recons_downsample_stride = 1;
        }
    }
    if (config["SphVoxelUpdate"]["minimum_dis"]) {
        double minimum_dis = config["SphVoxelUpdate"]["minimum_dis"].as<double>();
        if (minimum_dis == -1) {
            curl_voxel_update_config.minimum_squared_dis = std::numeric_limits<double>::lowest();
        } else {
            curl_voxel_update_config.minimum_squared_dis = minimum_dis * minimum_dis;
        }
    }
    if (config["SphVoxelUpdate"]["maximum_dis"]) {
        double maximum_dis = config["SphVoxelUpdate"]["maximum_dis"].as<double>();
        if (maximum_dis == -1) {
            curl_voxel_update_config.maximum_squared_dis = std::numeric_limits<double>::max();
        } else {
            curl_voxel_update_config.maximum_squared_dis = maximum_dis * maximum_dis;
        }
    }
    if (config["SphVoxelUpdate"]["ODOMETRY"]["max_patch_search_box_times"]) {
        curl_voxel_update_config.max_patch_search_box_times =
            config["SphVoxelUpdate"]["ODOMETRY"]["max_patch_search_box_times"]
                .as<decltype(curl_voxel_update_config.max_patch_search_box_times)>();
    }
    if (config["SphVoxelUpdate"]["ODOMETRY"]["is_use_all_associated_patches"]) {
        curl_voxel_update_config.is_use_all_associated_patches =
            config["SphVoxelUpdate"]["ODOMETRY"]["is_use_all_associated_patches"]
                .as<decltype(curl_voxel_update_config.is_use_all_associated_patches)>();
    }
    if (config["SphVoxelUpdate"]["ODOMETRY"]["is_kitti_correct"]) {
        curl_voxel_update_config.is_kitti_correct = config["SphVoxelUpdate"]["ODOMETRY"]["is_kitti_correct"]
                                                        .as<decltype(curl_voxel_update_config.is_kitti_correct)>();
    }
    if (config["SphVoxelUpdate"]["ODOMETRY"]["is_deskew"]) {
        curl_voxel_update_config.is_deskew =
            config["SphVoxelUpdate"]["ODOMETRY"]["is_deskew"].as<decltype(curl_voxel_update_config.is_deskew)>();
    }
    if (config["SphVoxelUpdate"]["TRAJECTORY_SEGMENTATION"]["minimum_observation_num"]) {
        curl_voxel_update_config.minimum_observation_num =
            config["SphVoxelUpdate"]["TRAJECTORY_SEGMENTATION"]["minimum_observation_num"]
                .as<decltype(curl_voxel_update_config.minimum_observation_num)>();
    }
    if (config["SphVoxelUpdate"]["SEGMENTATION"]["is_seg_only"]) {
        curl_voxel_update_config.is_seg_only = config["SphVoxelUpdate"]["SEGMENTATION"]["is_seg_only"]
                                                   .as<decltype(curl_voxel_update_config.is_seg_only)>();
    }
    if (config["SphVoxelUpdate"]["SEGMENTATION"]["cut_threshold"]) {
        curl_voxel_update_config.cut_threshold = config["SphVoxelUpdate"]["SEGMENTATION"]["cut_threshold"]
                                                     .as<decltype(curl_voxel_update_config.cut_threshold)>();
        curl_voxel_update_config.half_diag_cut_threshold = curl_voxel_update_config.cut_threshold / 2;
    }
    if (config["SphVoxelUpdate"]["FILTER"]["patch_minimun_pts_num_after_filter"]) {
        curl_voxel_update_config.patch_minimun_pts_num_after_filter =
            config["SphVoxelUpdate"]["FILTER"]["patch_minimun_pts_num_after_filter"]
                .as<decltype(curl_voxel_update_config.patch_minimun_pts_num_after_filter)>();
    }
    if (config["SphVoxelUpdate"]["FILTER"]["is_voxel_grid_filter"]) {
        curl_voxel_update_config.is_voxel_grid_filter =
            config["SphVoxelUpdate"]["FILTER"]["is_voxel_grid_filter"]
                .as<decltype(curl_voxel_update_config.is_voxel_grid_filter)>();
    }
    if (config["SphVoxelUpdate"]["FILTER"]["leaf_size"]) {
        curl_voxel_update_config.leaf_size =
            config["SphVoxelUpdate"]["FILTER"]["leaf_size"].as<decltype(curl_voxel_update_config.leaf_size)>();
    }
    if (config["SphVoxelUpdate"]["LOCAL_BA"]["keyframe_mini_squared_dis"]) {
        curl_voxel_update_config.keyframe_mini_squared_dis =
            config["SphVoxelUpdate"]["LOCAL_BA"]["keyframe_mini_squared_dis"]
                .as<decltype(curl_voxel_update_config.keyframe_mini_squared_dis)>();
    }
    if (config["SphVoxelUpdate"]["LOCAL_BA"]["keyframe_degree_change"]) {
        double keyframe_degree_change =
            config["SphVoxelUpdate"]["LOCAL_BA"]["keyframe_degree_change"].as<double>();
        curl_voxel_update_config.keyframe_radian_change = keyframe_degree_change * M_PI / 180;
    }
    if (config["SphVoxelUpdate"]["LOCAL_BA"]["keyframe_IQR_cost_thres"]) {
        curl_voxel_update_config.keyframe_IQR_cost_thres =
            config["SphVoxelUpdate"]["LOCAL_BA"]["keyframe_IQR_cost_thres"]
                .as<decltype(curl_voxel_update_config.keyframe_IQR_cost_thres)>();
    }
    if (config["SphVoxelUpdate"]["LOCAL_BA"]["points_regularizer_weight_square_root"]) {
        curl_voxel_update_config.points_regularizer_weight_square_root =
            config["SphVoxelUpdate"]["LOCAL_BA"]["points_regularizer_weight_square_root"]
                .as<decltype(curl_voxel_update_config.points_regularizer_weight_square_root)>();
    }
    if (config["SphVoxelUpdate"]["LOCAL_BA"]["max_residual"]) {
        curl_voxel_update_config.max_residual =
            config["SphVoxelUpdate"]["LOCAL_BA"]["max_residual"].as<decltype(curl_voxel_update_config.max_residual)>();
    }
    if (config["SphVoxelUpdate"]["LOCAL_BA"]["opt_status"]) {
        curl_voxel_update_config.opt_status =
            config["SphVoxelUpdate"]["LOCAL_BA"]["opt_status"].as<decltype(curl_voxel_update_config.opt_status)>();
    }
    if (config["SphVoxelUpdate"]["LOCAL_BA"]["max_num_iterations"]) {
        curl_voxel_update_config.max_num_iterations = config["SphVoxelUpdate"]["LOCAL_BA"]["max_num_iterations"]
                                                          .as<decltype(curl_voxel_update_config.max_num_iterations)>();
    }
    if (config["SphVoxelUpdate"]["LOCAL_BA"]["function_tolerance"]) {
        curl_voxel_update_config.function_tolerance = config["SphVoxelUpdate"]["LOCAL_BA"]["function_tolerance"]
                                                          .as<decltype(curl_voxel_update_config.function_tolerance)>();
    }
    if (config["SphVoxelUpdate"]["NEW_LANDMARK_GENERATION"]["minimum_points_to_fix_projection_direction"]) {
        curl_voxel_update_config.minimum_points_to_fix_projection_direction =
            config["SphVoxelUpdate"]["NEW_LANDMARK_GENERATION"]["minimum_points_to_fix_projection_direction"]
                .as<decltype(curl_voxel_update_config.minimum_points_to_fix_projection_direction)>();
    }
    if (config["SphVoxelUpdate"]["NEW_LANDMARK_GENERATION"]["minimum_points_to_add_new_patch_seg"]) {
        curl_voxel_update_config.minimum_points_to_add_new_patch_seg =
            config["SphVoxelUpdate"]["NEW_LANDMARK_GENERATION"]["minimum_points_to_add_new_patch_seg"]
                .as<decltype(curl_voxel_update_config.minimum_points_to_add_new_patch_seg)>();
    }
    if (config["SphVoxelUpdate"]["NEW_LANDMARK_GENERATION"]["minimum_points_to_add_new_patch_ground"]) {
        curl_voxel_update_config.minimum_points_to_add_new_patch_ground =
            config["SphVoxelUpdate"]["NEW_LANDMARK_GENERATION"]["minimum_points_to_add_new_patch_ground"]
                .as<decltype(curl_voxel_update_config.minimum_points_to_add_new_patch_ground)>();
    }
    if (config["SphVoxelUpdate"]["NEW_LANDMARK_GENERATION"]["highest_IoU_new_landmark_thres"]) {
        curl_voxel_update_config.highest_IoU_new_landmark_thres =
            config["SphVoxelUpdate"]["NEW_LANDMARK_GENERATION"]["highest_IoU_new_landmark_thres"]
                .as<decltype(curl_voxel_update_config.highest_IoU_new_landmark_thres)>();
    }
    if (config["SphVoxelUpdate"]["NEW_LANDMARK_GENERATION"]["largest_distance"]) {
        double largest_distance =
            config["SphVoxelUpdate"]["NEW_LANDMARK_GENERATION"]["largest_distance"].as<double>();
        if (largest_distance == -1) {
            curl_voxel_update_config.largest_distance_squared = -1;
        } else {
            curl_voxel_update_config.largest_distance_squared = largest_distance * largest_distance;
        }
    }
}

void load_loop_closure_config(const std::string &dir, CURL_LOOP_CLOSURE_CONFIG &curl_loop_closure_config) {
    YAML::Node config = YAML::LoadFile(dir);
    if (config["LOOP_CLOSURE"]["is_enable"]) {
        curl_loop_closure_config.is_enable =
            config["LOOP_CLOSURE"]["is_enable"].as<decltype(curl_loop_closure_config.is_enable)>();
    }
    if (config["LOOP_CLOSURE"]["search_radius"]) {
        double search_radius = config["LOOP_CLOSURE"]["search_radius"].as<double>();
        curl_loop_closure_config.search_radius_squared = search_radius * search_radius;
    }
    if (config["LOOP_CLOSURE"]["simple_loop_closure_search_radius"]) {
        double simple_loop_closure_search_radius =
            config["LOOP_CLOSURE"]["simple_loop_closure_search_radius"].as<double>();
        curl_loop_closure_config.simple_loop_closure_search_radius_squared =
            simple_loop_closure_search_radius * simple_loop_closure_search_radius;
    }
    if (config["LOOP_CLOSURE"]["remove_overlap_keyframe_patches_region"]) {
        double remove_overlap_keyframe_patches_region =
            config["LOOP_CLOSURE"]["remove_overlap_keyframe_patches_region"].as<double>();
        curl_loop_closure_config.remove_overlap_keyframe_patches_region_squared =
            remove_overlap_keyframe_patches_region * remove_overlap_keyframe_patches_region;
    }
    if (config["LOOP_CLOSURE"]["remove_overlap_iou_thres"]) {
        curl_loop_closure_config.remove_overlap_iou_thres =
            config["LOOP_CLOSURE"]["remove_overlap_iou_thres"].as<double>();
    }
    if (config["LOOP_CLOSURE"]["minimum_middle_segment_valid_dis"]) {
        double minimum_middle_segment_valid_dis =
            config["LOOP_CLOSURE"]["minimum_middle_segment_valid_dis"].as<double>();
        curl_loop_closure_config.minimum_middle_segment_valid_dis_squared =
            minimum_middle_segment_valid_dis * minimum_middle_segment_valid_dis;
    }
    if (config["LOOP_CLOSURE"]["association_rate_for_opt_history_keyframe"]) {
        curl_loop_closure_config.association_rate_for_opt_history_keyframe =
            config["LOOP_CLOSURE"]["association_rate_for_opt_history_keyframe"]
                .as<decltype(curl_loop_closure_config.association_rate_for_opt_history_keyframe)>();
    }
    if (config["LOOP_CLOSURE"]["DATA_ASSOCIATION"]["region_width_elements"]) {
        curl_loop_closure_config.region_width_elements =
            config["LOOP_CLOSURE"]["DATA_ASSOCIATION"]["region_width_elements"]
                .as<decltype(curl_loop_closure_config.region_width_elements)>();
    }
}

#endif // EXAMPLE_LOAD_CONFIG
