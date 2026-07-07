#include "curl_slam/CurlContinuousRecons.h"

void SimplePatch::set_boundary(const double _half_diag_side) {
    half_diag_side = _half_diag_side;
    min_x = -_half_diag_side;
    max_x = _half_diag_side;
    min_y = -_half_diag_side;
    max_y = _half_diag_side;
}

template <typename BasicType>
void SimplePatch::calculate_default_recons_v_mesh(
    const std::shared_ptr<SH_TABLE_CONFIG<BasicType>> SH_table_config_ptr) {
    is_default_mesh_calculated = true;
    Eigen::MatrixXd I = Eigen::MatrixXd::Ones(default_img_iso, default_img_iso) * D_INVALID;
    Eigen::MatrixXd I_dirs;
    Eigen::MatrixXd xy_grid;
    Eigen::VectorXd recons_v;
    curl::mesh_grid<double>(min_x, max_x, default_img_iso, max_y, min_y, default_img_iso, xy_grid);
    recons_v = (curl::invLeastSquaresSHT_table<BasicType>(sph_coeff.cast<BasicType>(),
                                                          SH_table_config_ptr->I_dirs(mask_idx_vec, Eigen::all),
                                                          *SH_table_config_ptr, sph_degree))
                   .template cast<double>();
    default_mesh = curl::meshing_with_idx(xy_grid(mask_idx_vec, 0), xy_grid(mask_idx_vec, 1));
    curl::calculate_lengths_areas_of_mesh(default_mesh, recons_v, lengths, areas);
}

template <typename BasicType>
Eigen::MatrixXd SimplePatch::calculate_recons_v_local_continuous_recons(
    const int patch_img_rso, int sph_degree_CR, double IQR_factor, const bool is_fix_length_threshold,
    const double _length_thres, const std::shared_ptr<SH_TABLE_CONFIG<BasicType>> SH_table_config_ptr) {
    if (sph_degree_CR > sph_degree) {
        sph_degree_CR = sph_degree;
    }
    if (!is_default_mesh_calculated) {
        // if (true) {
        calculate_default_recons_v_mesh<BasicType>(SH_table_config_ptr);
    }
    if (lengths.empty()) {
        return Eigen::MatrixXd();
    }
    Eigen::VectorXd recons_v;
    Eigen::MatrixXd I_dirs_CR;
    curl::mesh_grid<double>(SH_table_config_ptr->get_azi_low(), SH_table_config_ptr->get_azi_high(), patch_img_rso,
                            SH_table_config_ptr->get_elev_high(), SH_table_config_ptr->get_elev_low(), patch_img_rso,
                            I_dirs_CR);
    Eigen::MatrixXd xy_grid_CR;
    curl::mesh_grid<double>(min_x, max_x, patch_img_rso, max_y, min_y, patch_img_rso, xy_grid_CR);
    Eigen::MatrixXd v_local_recons(3, xy_grid_CR.rows());
    v_local_recons(Eigen::seq(0, 1), Eigen::all) = xy_grid_CR.transpose();
    v_local_recons.row(2) =
        (curl::invLeastSquaresSHT_table<BasicType>(sph_coeff.cast<BasicType>(), I_dirs_CR.cast<BasicType>(),
                                                   *SH_table_config_ptr, sph_degree_CR))
            .template cast<double>();
    // start to calculate the mask
    double length_thres = curl::IQR_thres<double>(lengths, IQR_factor);
    if (!is_fix_length_threshold) {
        length_thres = curl::IQR_thres<double>(lengths, IQR_factor);
    } else {
        length_thres = _length_thres;
    }

    double area_thres = curl::IQR_thres<double>(areas, IQR_factor);
    std::vector<int> mask_idx_vec_CR;
    curl::mask_generation(default_mesh, xy_grid_CR, length_thres, area_thres, mask_idx_vec_CR);

    return v_local_recons(Eigen::all, mask_idx_vec_CR);
}

Eigen::VectorXi SimplePatch::get_binary_mask() const {
    Eigen::VectorXi binary_mask(default_img_iso * default_img_iso);
    binary_mask.setZero();
    binary_mask(mask_idx_vec).setOnes();
    return binary_mask;
}

template <typename BasicType>
CurlContinuousRecons<BasicType>::CurlContinuousRecons(
    ros::NodeHandle *_nh, const std::shared_ptr<CURL_CONTINUOUS_RECONS_CONFIG> _curl_continuous_recons_config_ptr,
    const std::shared_ptr<CURL_VOXEL_MAPPING_CONFIG> &_curl_voxel_mapping_config_ptr,
    const std::shared_ptr<SH_TABLE_CONFIG<BasicType>> &_SH_table_config_ptr,
    const std::shared_ptr<DIRECT_METHOD_CONFIG> &_direct_method_config_ptr,
    const std::shared_ptr<DEBUG_CONFIG> &_debug_config_ptr)
    : nh(_nh), curl_continuous_recons_config_ptr(_curl_continuous_recons_config_ptr),
      curl_voxel_mapping_config_ptr(_curl_voxel_mapping_config_ptr), SH_table_config_ptr(_SH_table_config_ptr),
      direct_method_config_ptr(_direct_method_config_ptr), debug_config_ptr(_debug_config_ptr) {
    f = std::bind(&CurlContinuousRecons<BasicType>::param_callback, this, std::placeholders::_1, std::placeholders::_2);
    server.setCallback(f);
    open3d_display_pcd_ptr = std::make_shared<open3d::geometry::PointCloud>();
}

template <typename BasicType> void CurlContinuousRecons<BasicType>::run() {
    ros::Rate rate(10); // 10 hz
    reconstruction();
    current_curl_continuous_recons_config = *curl_continuous_recons_config_ptr;
    open3d_vis.CreateVisualizerWindow("Reconstruction", 1920, 1080);
    open3d_vis.AddGeometry(open3d_display_pcd_ptr);
    while (ros::ok()) {
        if (current_curl_continuous_recons_config != *curl_continuous_recons_config_ptr) {
            reconstruction();
            current_curl_continuous_recons_config = *curl_continuous_recons_config_ptr;
            open3d_vis.CreateVisualizerWindow("Reconstruction", 1920, 1080);
            open3d_vis.AddGeometry(open3d_display_pcd_ptr);
        } else {
            Eigen::Matrix3d R = open3d_display_pcd_ptr->GetRotationMatrixFromXYZ(
                Eigen::Vector3d(0, 0, M_1_PI * 2 / 1000 * curl_continuous_recons_config_ptr->open3d_speed));
            open3d_display_pcd_ptr->Rotate(R, open3d_display_pcd_center);
            open3d_vis.UpdateGeometry();
            open3d_vis.PollEvents();
            open3d_vis.UpdateRender();
            ros::spinOnce();
        }
        rate.sleep();
    }
}

template <typename BasicType> void CurlContinuousRecons<BasicType>::run_reconstruction() {
    ros::Rate rate(10); // 10 hz
    reconstruction_new();
    current_curl_continuous_recons_config = *curl_continuous_recons_config_ptr;
    open3d_vis.CreateVisualizerWindow("Reconstruction", 1920, 1080);
    open3d_vis.AddGeometry(open3d_display_pcd_ptr);
    open3d_vis.GetRenderOption().point_size_ = 1;
    bool saved_pcd_flag = false;
    bool saved_curlmap_flag = false;
    while (ros::ok()) {
        if (current_curl_continuous_recons_config != *curl_continuous_recons_config_ptr) {
            reconstruction_new();
            current_curl_continuous_recons_config = *curl_continuous_recons_config_ptr;
            open3d_vis.CreateVisualizerWindow("Reconstruction", 1920, 1080);
            open3d_vis.AddGeometry(open3d_display_pcd_ptr);
            saved_pcd_flag = false;
            saved_curlmap_flag = false;
        } else {
            Eigen::Matrix3d R = open3d_display_pcd_ptr->GetRotationMatrixFromXYZ(
                Eigen::Vector3d(0, 0, M_1_PI * 2 / 1000 * curl_continuous_recons_config_ptr->open3d_speed));
            open3d_display_pcd_ptr->Rotate(R, open3d_display_pcd_center);
            open3d_vis.UpdateGeometry();
            open3d_vis.PollEvents();
            open3d_vis.UpdateRender();
            // for saving
            if (curl_continuous_recons_config_ptr->is_save_pcd && !saved_pcd_flag) {
                std::cout << "start saving pcd" << std::endl;
                std::string filename;
                if (!curl_continuous_recons_config_ptr->is_fix_length_threshold) {
                    filename = debug_config_ptr->results_dir + "/final_map/recons_patch_img_rso_" +
                               std::to_string(curl_continuous_recons_config_ptr->patch_img_rso) + "_degree_" +
                               std::to_string(curl_continuous_recons_config_ptr->SH_degree) + "_IQR_factor_" +
                               std::to_string(curl_continuous_recons_config_ptr->IQR_factor) + ".pcd";
                } else {
                    filename = debug_config_ptr->results_dir + "/final_map/recons_patch_img_rso_" +
                               std::to_string(curl_continuous_recons_config_ptr->patch_img_rso) + "_degree_" +
                               std::to_string(curl_continuous_recons_config_ptr->SH_degree) + "_IQR_factor_" +
                               std::to_string(curl_continuous_recons_config_ptr->IQR_factor) + "_length_thres_" +
                               std::to_string(curl_continuous_recons_config_ptr->length_thres) + ".pcd";
                }
                // save open3d_display_pcd_ptr into filename
                open3d::io::WritePointCloud(filename, *open3d_display_pcd_ptr);
                std::cout << "finish saving pcd" << std::endl;
                saved_pcd_flag = true;
            }
            if (curl_continuous_recons_config_ptr->is_save_curlmap && !saved_curlmap_flag) {
                std::cout << "start saving curlmap" << std::endl;
                std::string filename = debug_config_ptr->results_dir + "/final_map/degree_" +
                                       std::to_string(curl_continuous_recons_config_ptr->SH_degree) + ".bin";
                // save open3d_display_pcd_ptr into filename
                save_curlmap(filename, current_curl_continuous_recons_config.SH_degree);
                std::cout << "finish saving curlmap" << std::endl;
                saved_curlmap_flag = true;
            }
            ros::spinOnce();
        }
        rate.sleep();
    }
}

template <typename BasicType>
void CurlContinuousRecons<BasicType>::param_callback(curl_slam::ContinuousReconsConfig &config, uint32_t level) {
    curl_continuous_recons_config_ptr->patch_img_rso = config.w + 1;
    curl_continuous_recons_config_ptr->open3d_speed = config.open3d_speed;
    curl_continuous_recons_config_ptr->IQR_factor = config.IQR_factor;
    curl_continuous_recons_config_ptr->SH_degree = config.SH_degree;
    curl_continuous_recons_config_ptr->is_save_pcd = config.is_save_pcd;
    curl_continuous_recons_config_ptr->is_save_curlmap = config.is_save_curlmap;
    curl_continuous_recons_config_ptr->length_thres = config.length_thres;
    curl_continuous_recons_config_ptr->is_fix_length_threshold = config.is_fix_length_threshold;
}

template <typename BasicType> void CurlContinuousRecons<BasicType>::reconstruction() {
    open3d_pcd.points_.clear();

    // Convert unordered_set to vector for parallel processing
    std::vector<std::shared_ptr<PatchInfo<BasicType>>> patch_info_vector(all_saving_patches_ptr.begin(),
                                                                         all_saving_patches_ptr.end());
#pragma omp parallel for num_threads(omp_get_num_procs()) default(none) shared(                                        \
    patch_info_vector, open3d_pcd, SH_table_config_ptr, direct_method_config_ptr, curl_continuous_recons_config_ptr)
    for (const auto &patch_info_ptr : patch_info_vector) {
        patch_info_ptr->patch_procession_ptr->set_config_ptr(SH_table_config_ptr, direct_method_config_ptr);
        Eigen::Isometry3d T_w_obj =
            Eigen::Isometry3d(patch_info_ptr->keyframe_ptr->get_T_w_lidar()) * patch_info_ptr->T_obj_lidar.inverse();
        Eigen::MatrixX<BasicType> recons_v_obj =
            patch_info_ptr->patch_procession_ptr->calculate_recons_v_local_continuous_recons(
                curl_continuous_recons_config_ptr->patch_img_rso, curl_continuous_recons_config_ptr->SH_degree);
        Eigen::MatrixXd recons_v_world =
            (T_w_obj.rotation() * recons_v_obj.template cast<double>()).colwise() + T_w_obj.translation();
        // filter invalid points
        std::vector<BasicType> recons_v_obj_valid_vec;
        recons_v_obj_valid_vec.reserve(recons_v_obj_valid_vec.size());
        for (int i = 0; i < recons_v_obj.cols(); ++i) {
            if (std::abs(recons_v_obj(2, i)) < curl_voxel_mapping_config_ptr->half_diag_cut_threshold) {
#pragma omp critical
                open3d_pcd.points_.emplace_back(recons_v_world.col(i).template cast<double>());
            }
        }
    }

    open3d_pcd.EstimateNormals(open3d::geometry::KDTreeSearchParamKNN(100), true);
    *open3d_display_pcd_ptr = open3d_pcd;
    open3d_display_pcd_center = open3d_pcd.GetCenter();
}

template <typename BasicType> void CurlContinuousRecons<BasicType>::reconstruction_new() {
    open3d_pcd.points_.clear();
    double progress_counter = 0;
#pragma omp parallel for num_threads(omp_get_num_procs()) default(none) shared(                                        \
    simple_patches, curl_continuous_recons_config_ptr, SH_table_config_ptr, open3d_pcd, progress_counter, std::cout)
    for (auto &simple_patch : simple_patches) {
        Eigen::MatrixXd recons_v_obj = simple_patch.calculate_recons_v_local_continuous_recons<BasicType>(
            curl_continuous_recons_config_ptr->patch_img_rso, curl_continuous_recons_config_ptr->SH_degree,
            curl_continuous_recons_config_ptr->IQR_factor, curl_continuous_recons_config_ptr->is_fix_length_threshold,
            curl_continuous_recons_config_ptr->length_thres, SH_table_config_ptr);
        Eigen::MatrixXd recons_v_world =
            (simple_patch.T_w_obj(Eigen::seq(0, 2), Eigen::seq(0, 2)) * recons_v_obj).colwise() +
            simple_patch.T_w_obj(Eigen::seq(0, 2), 3);
        // filter invalid points
        std::vector<BasicType> recons_v_obj_valid_vec;
        recons_v_obj_valid_vec.reserve(recons_v_obj_valid_vec.size());
#pragma omp critical
        {
            for (int i = 0; i < recons_v_obj.cols(); ++i) {
                if (std::abs(recons_v_obj(2, i)) < simple_patch.half_diag_side) {
                    open3d_pcd.points_.emplace_back(recons_v_world.col(i));
                }
            }
            ++progress_counter;
        }
    }

    open3d_pcd.EstimateNormals(open3d::geometry::KDTreeSearchParamKNN(100), true);
    *open3d_display_pcd_ptr = open3d_pcd;
    open3d_display_pcd_center = open3d_pcd.GetCenter();
}

template <typename BasicType> bool CurlContinuousRecons<BasicType>::load_from_binary(const std::string &filename) {
    std::ifstream infile;
    infile.open(filename, std::ios::binary | std::ios::in);
    Eigen::MatrixXf T_tmp(3, 4);
    while (true) {
        SimplePatch simple_patch;
        if (!infile.read(reinterpret_cast<char *>(&simple_patch.is_ground), sizeof(bool))) {
            return true;
        }
        simple_patch.set_boundary(curl_voxel_mapping_config_ptr->half_diag_cut_threshold);
        simple_patch.sph_degree =
            simple_patch.is_ground ? SH_table_config_ptr->ground_SH_degree : SH_table_config_ptr->max_SH_degree;
        if (!infile.read(reinterpret_cast<char *>(T_tmp.data()), T_tmp.size() * sizeof(float))) {
            return false;
        }
        simple_patch.T_w_obj.setIdentity();
        simple_patch.T_w_obj(Eigen::seq(0, 2), Eigen::all) = T_tmp.cast<double>();
        simple_patch.sph_coeff.resize((simple_patch.sph_degree + 1) * (simple_patch.sph_degree + 1));
        if (!infile.read(reinterpret_cast<char *>(simple_patch.sph_coeff.data()),
                         simple_patch.sph_coeff.size() * sizeof(double))) {
            return false;
        }
        simple_patch.default_img_iso = direct_method_config_ptr->minimum_img_rso;
        double binary_mask_size = std::pow(simple_patch.default_img_iso, 2);
        int loop_times = std::ceil(binary_mask_size / 64.0);
        for (int loop = 0; loop < loop_times; ++loop) {
            std::bitset<64> mask_bits;
            if (!infile.read(reinterpret_cast<char *>(&mask_bits), sizeof(std::bitset<64>))) {
                std::cout << "Invalid binary file" << std::endl;
                return false;
            }
            for (int i = 0; i < 64; ++i) {
                int mask_idx = loop * 64 + i;
                if (mask_idx >= binary_mask_size) {
                    break;
                }
                if (mask_bits[63 - i] == 1) {
                    simple_patch.mask_idx_vec.push_back(mask_idx);
                }
            }
        }
        simple_patches.push_back(simple_patch);
    }
}

template <typename BasicType>
void CurlContinuousRecons<BasicType>::save_curlmap(const std::string &filename, const int degree) {

    std::ofstream debug_binary_file(filename, std::ios::binary | std::ios::out);
    for (auto &simple_patch : simple_patches) {
        // save is_ground
        debug_binary_file.write(reinterpret_cast<const char *>(&simple_patch.is_ground), sizeof(bool));
        Eigen::MatrixXf T_w_obj = simple_patch.T_w_obj(Eigen::seq(0, 2), Eigen::all).template cast<float>();
        // save T_w_obj
        debug_binary_file.write(reinterpret_cast<const char *>(T_w_obj.data()), T_w_obj.size() * sizeof(float));
        // save sph_coeff
        int sph_size = std::pow(degree + 1, 2);
        if (degree > simple_patch.sph_degree) {
            sph_size = simple_patch.sph_coeff.size();
        }
        debug_binary_file.write(reinterpret_cast<const char *>(simple_patch.sph_coeff.data()),
                                sph_size * sizeof(double));
        // save binary_mask
        Eigen::VectorXi binary_mask = simple_patch.get_binary_mask();
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

template class CurlContinuousRecons<BT>; // This is very important
