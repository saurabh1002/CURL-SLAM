//
// Created by zkc on 19/06/23.
//

#include "curl_slam/PatchProcession.h"

#include <algorithm>
#include <cstdlib>
#include <ros/ros.h>

// constructor for spatial without conformal-mapping odometry
template <typename BasicType>
PatchProcession<BasicType>::PatchProcession(const Eigen::Vector3d &_last_update_position,
                                            const std::shared_ptr<DIRECT_METHOD_CONFIG> _direct_method_config_ptr,
                                            const std::shared_ptr<SH_TABLE_CONFIG<BasicType>> _SH_table_config_ptr,
                                            std::shared_ptr<DEBUG_CONFIG> _debug_config_ptr,
                                            const Eigen::MatrixX<BasicType> &_v_obj,
                                            const Eigen::VectorX<BasicType> &_v_range_squared,
                                            const BasicType half_diag_side, int _SH_degree, bool _is_ground,
                                            bool _is_cleared, int _status)
    : last_update_position(_last_update_position), direct_method_config_ptr(_direct_method_config_ptr),
      SH_table_config_ptr(_SH_table_config_ptr), debug_config_ptr(_debug_config_ptr), SH_degree(_SH_degree),
      is_ground(_is_ground), is_cleared(_is_cleared), status(_status) {
    warping_SH_degree = std::min(SH_degree, SH_table_config_ptr->warping_SH_degree);
    update_step_counter = 0;
    initialization_counter = 0;
    is_initialization = true;
    is_sph_coeff_lock = false;
    is_set_BA_sph_coeff = false;
    is_build_mask_F = false;
    x_size = direct_method_config_ptr->minimum_img_rso;
    y_size = direct_method_config_ptr->minimum_img_rso;

    // fine_grid_x_size = (x_size - 1) * direct_method_config_ptr->fine_grid_times + 1;
    // fine_grid_y_size = (x_size - 1) * direct_method_config_ptr->fine_grid_times + 1;
    min_x = -half_diag_side;
    max_x = half_diag_side;
    min_y = -half_diag_side;
    max_y = half_diag_side;
    curl::mesh_grid<BasicType>(min_x, max_x, x_size, max_y, min_y, y_size, xy_grid);
    I_mask = Eigen::MatrixX<BasicType>::Ones(y_size, x_size) * T_INVALID;
    I_mask_times.setZero(y_size, x_size);
    I_mask_counters.setZero(y_size, x_size);
    //        I_mask_weights = Eigen::MatrixX<BasicType>::Ones(y_size, x_size) * T_INVALID;
    I_mask_weights.setZero(y_size, x_size);
    recons_v_local.resize(y_size * x_size, 3);
    BA_I_mask = Eigen::MatrixX<BasicType>::Ones(y_size, x_size) * T_INVALID;
    BA_I_mask_all = Eigen::MatrixX<BasicType>::Ones(y_size, x_size) * T_INVALID;
    BA_I_mask_times.setZero(y_size, x_size);
    // initialize the pyramid
    // FIXME: precalculate this outside
    pyramids.depth = 1;
    double n = double(min(x_size, y_size));
    while (n >= direct_method_config_ptr->pyramid_minimum_img_rso * 2) {
        ++pyramids.depth;
        n /= 2;
    }
    pyramids.I_vec.resize(pyramids.depth);
    pyramids.x_grid_len_vec.resize(pyramids.depth);
    pyramids.y_grid_len_vec.resize(pyramids.depth);
    pyramids.I_Gx_vec.resize(pyramids.depth);
    pyramids.I_Gy_vec.resize(pyramids.depth);
    pyramids.x_grid_len_vec[0] = (max_x - min_x) / static_cast<BasicType>(x_size - 1);
    pyramids.y_grid_len_vec[0] = (max_y - min_y) / static_cast<BasicType>(y_size - 1);
    for (int idx = 1; idx < pyramids.depth; ++idx) {
        pyramids.x_grid_len_vec[idx] = pyramids.x_grid_len_vec[idx - 1] * 2;
        pyramids.y_grid_len_vec[idx] = pyramids.y_grid_len_vec[idx - 1] * 2;
    }
    G_theta = M_PI * SH_table_config_ptr->SH_scale / (max_y - min_y);
    G_phi = 2 * M_PI * SH_table_config_ptr->SH_scale / (max_x - min_x);
    mask_idx.clear();
    if (update_mask(_v_obj, _v_range_squared)) {
        extract_sph_coeff();
        calculate_I_without_conformal_mapping();
        calculate_delta_without_conformal_mapping();
        calculate_recons_v_local_without_conformal_mapping();
    }
}

template <typename BasicType>
PatchProcession<BasicType>::PatchProcession(const Eigen::Vector3d &_last_update_position,
                                            const std::shared_ptr<DIRECT_METHOD_CONFIG> _direct_method_config_ptr,
                                            const std::shared_ptr<SH_TABLE_CONFIG<BasicType>> _SH_table_config_ptr,
                                            std::shared_ptr<DEBUG_CONFIG> _debug_config_ptr,
                                            const Eigen::MatrixX<BasicType> &_v_obj,
                                            const Eigen::VectorX<BasicType> &_v_range_squared,
                                            const BasicType half_diag_side, int _SH_degree, bool _is_ground,
                                            int is_low_RAM, bool _is_cleared, int _status)
    : last_update_position(_last_update_position), direct_method_config_ptr(_direct_method_config_ptr),
      SH_table_config_ptr(_SH_table_config_ptr), debug_config_ptr(_debug_config_ptr), SH_degree(_SH_degree),
      is_ground(_is_ground), is_cleared(_is_cleared), status(_status),
      x_size(_direct_method_config_ptr->minimum_img_rso), y_size(_direct_method_config_ptr->minimum_img_rso),
      AT_A_block((_SH_degree + 1) * (_SH_degree + 1), (_SH_degree + 1) * (_SH_degree + 1)),
      AT_b_block((_SH_degree + 1) * (_SH_degree + 1)) {
    warping_SH_degree = std::min(SH_degree, SH_table_config_ptr->warping_SH_degree);
    update_step_counter = 0;
    initialization_counter = 0;
    is_initialization = true;
    is_sph_coeff_lock = false;
    is_set_BA_sph_coeff = false;
    is_build_mask_F = false;
    // x_size = direct_method_config_ptr->minimum_img_rso;
    // y_size = direct_method_config_ptr->minimum_img_rso;

    min_x = -half_diag_side;
    max_x = half_diag_side;
    min_y = -half_diag_side;
    max_y = half_diag_side;
    // initialize the pyramid
    pyramids.depth = 1;

    pyramids.x_grid_len_vec.resize(pyramids.depth);
    pyramids.y_grid_len_vec.resize(pyramids.depth);
    pyramids.x_grid_len_vec[0] = (max_x - min_x) / static_cast<BasicType>(x_size - 1);
    pyramids.y_grid_len_vec[0] = (max_y - min_y) / static_cast<BasicType>(y_size - 1);
    for (int idx = 1; idx < pyramids.depth; ++idx) {
        pyramids.x_grid_len_vec[idx] = pyramids.x_grid_len_vec[idx - 1] * 2;
        pyramids.y_grid_len_vec[idx] = pyramids.y_grid_len_vec[idx - 1] * 2;
    }
    G_theta = M_PI * SH_table_config_ptr->SH_scale / (max_y - min_y);
    G_phi = 2 * M_PI * SH_table_config_ptr->SH_scale / (max_x - min_x);
    AT_A_block.setZero();
    AT_b_block.setZero();
    if (update_mask_low_RAM(_v_obj)) {
        extract_sph_coeff_low_RAM();
        // calculate_I_without_conformal_mapping();
        // calculate_delta_without_conformal_mapping();
        // calculate_recons_v_local_without_conformal_mapping_low_RAM();
    }
}

template <typename BasicType>
PatchProcession<BasicType>::PatchProcession(const std::shared_ptr<DIRECT_METHOD_CONFIG> _direct_method_config_ptr,
                                            const std::shared_ptr<SH_TABLE_CONFIG<BasicType>> _SH_table_config_ptr,
                                            std::shared_ptr<DEBUG_CONFIG> _debug_config_ptr,
                                            const BasicType half_diag_side, int _SH_degree, bool _is_ground,
                                            const Eigen::VectorXd _sph_coeff, std::vector<BasicType> binary_mask)
    : direct_method_config_ptr(_direct_method_config_ptr), SH_table_config_ptr(_SH_table_config_ptr),
      debug_config_ptr(_debug_config_ptr), SH_degree(_SH_degree), is_ground(_is_ground), sph_coeff(_sph_coeff) {

    is_cleared = false;

    warping_SH_degree = std::min(SH_degree, SH_table_config_ptr->warping_SH_degree);

    x_size = direct_method_config_ptr->minimum_img_rso;
    y_size = direct_method_config_ptr->minimum_img_rso;

    min_x = -half_diag_side;
    max_x = half_diag_side;
    min_y = -half_diag_side;
    max_y = half_diag_side;

    // Process binary mask to convert non-1 values to T_INVALID
    for (auto &value : binary_mask) {
        if (value != 1) {
            value = T_INVALID;
        }
    }

    for (int i = 0; i < binary_mask.size(); ++i) {
        if (binary_mask[i] != 1) {
            binary_mask[i] = T_INVALID;
        } else {
            mask_idx.push_back(i);
        }
    }

    mask_idx_set = std::set<int>(mask_idx.begin(), mask_idx.end());
    mask_idx_flags.assign(x_size * y_size, 0);
    for (const int idx : mask_idx) {
        mask_idx_flags[idx] = 1;
    }

    assert(y_size * x_size == binary_mask.size());

    if (y_size * x_size != binary_mask.size()) {
        ROS_ERROR("Binary mask size mismatch: expected %d, got %zu", y_size * x_size, binary_mask.size());
        throw std::invalid_argument("Binary mask size does not match expected dimensions");
    }

    I_mask = Eigen::Map<const Eigen::MatrixX<BasicType>>(binary_mask.data(), y_size, x_size);

    pyramids.x_grid_len_vec.resize(1);
    pyramids.y_grid_len_vec.resize(1);
    pyramids.x_grid_len_vec[0] = (max_x - min_x) / static_cast<BasicType>(x_size - 1);
    pyramids.y_grid_len_vec[0] = (max_y - min_y) / static_cast<BasicType>(y_size - 1);

    G_theta = M_PI * SH_table_config_ptr->SH_scale / (max_y - min_y);
    G_phi = 2 * M_PI * SH_table_config_ptr->SH_scale / (max_x - min_x);

    // calculate I

    if (I.rows() != y_size || I.cols() != x_size) {
        I = Eigen::MatrixX<BasicType>::Ones(y_size, x_size) * T_INVALID;
    }
    if (SH_table_config_ptr->is_speed_up) {
        Eigen::Map<Eigen::VectorX<BasicType>>(I.data(), y_size * x_size)(mask_idx) =
            curl::invLeastSquaresSHT_table<BasicType>(sph_coeff.cast<BasicType>(),
                                                      SH_table_config_ptr->I_dirs(mask_idx, Eigen::all),
                                                      *SH_table_config_ptr, SH_degree);
    } else {
        Eigen::Map<Eigen::VectorX<BasicType>>(I.data(), y_size * x_size)(mask_idx) =
            curl::invLeastSquaresSHT<BasicType>(sph_coeff.cast<BasicType>(),
                                                SH_table_config_ptr->I_dirs(mask_idx, Eigen::all), SH_degree);
    }

    if (xy_grid.rows() != x_size * y_size) {
        curl::mesh_grid<BasicType>(min_x, max_x, x_size, max_y, min_y, y_size, xy_grid);
    }
    recons_v_local.resize(y_size * x_size, 3);
    recons_v_local(Eigen::all, Eigen::seq(0, 1)) = xy_grid;
    recons_v_local.col(2) = Eigen::Map<Eigen::VectorX<BasicType>>(I.data(), I.size());
    recons_v_local_dirty.store(false);
}

template <typename BasicType> PatchProcession<BasicType>::~PatchProcession() {}

// member functions
// member functions
template <typename BasicType>
bool PatchProcession<BasicType>::update_mask(const Eigen::MatrixX<BasicType> &_v_obj,
                                             const Eigen::VectorX<BasicType> &_v_range_squared) {
    // FIXME: this can only be used as initialization, update one should be changed, otherwise, ground patches are hard
    // to be updated
    // TODO: remove invalid points
    // TODO: use interpolation function
    // this mask has the same resolution of the image
    if (is_cleared) {
        recover_data();
    }
    std::unique_lock<std::shared_mutex> ul_mask_idx(mask_idx_lock);

    for (int i = 0; i < _v_obj.cols(); ++i) {
        // notice the row_idx is max_y - _v_obj(1, i)
        BasicType row_idx_float = (max_y - _v_obj(1, i)) / (max_y - min_y) * (static_cast<BasicType>(y_size) - 1);
        BasicType col_idx_float = (_v_obj(0, i) - min_x) / (max_x - min_x) * (static_cast<BasicType>(x_size) - 1);
        // int row_up_idx = std::ceil(row_idx_float);
        // int row_down_idx = std::floor(row_idx_float);
        // int col_up_idx = std::ceil(col_idx_float);
        // int col_down_idx = std::floor(col_idx_float);
        // for (int row : {row_down_idx, row_up_idx}) {
        //     for (int col : {col_down_idx, col_up_idx}) {
        // Check bounds for the current combination of row and col
        int row = std::round(row_idx_float);
        int col = std::round(col_idx_float);
        if (row >= 0 && row < y_size && col >= 0 && col < x_size) {
            // Calculate distance for the current combination
            BasicType dis_squared = std::pow(row_idx_float - static_cast<BasicType>(row), 2) +
                                    std::pow(col_idx_float - static_cast<BasicType>(col), 2);
            // BasicType weight = curl::gaussian_weight_function(dis_squared);
            BasicType weight = curl::gaussian_range_weight_function<BasicType>(dis_squared, _v_range_squared(i));

            if (I_mask(row, col) == T_INVALID) {
                // If the position is invalid, set it with the weighted value
                I_mask(row, col) = weight * _v_obj(2, i);
                mask_idx.push_back(col * y_size + row); // Adjusted indexing
            } else {
                // If the position is already valid, add the weighted value
                I_mask(row, col) += weight * _v_obj(2, i);
            }
            // Accumulate the weight for normalization or further processing
            I_mask_times(row, col) += weight;
            I_mask_counters(row, col) += 1;
            // }
            // }
        }
    }
    I_mask_dense = I_mask;
    Eigen::Map<Eigen::VectorX<BasicType>>(I_mask_dense.data(), y_size * x_size)(mask_idx) =
        Eigen::Map<Eigen::VectorX<BasicType>>(I_mask.data(), y_size * x_size)(mask_idx).array() /
        Eigen::Map<Eigen::VectorX<BasicType>>(I_mask_times.data(), y_size * x_size)(mask_idx).array();
    Eigen::Map<Eigen::VectorX<BasicType>>(I_mask_weights.data(), y_size * x_size)(mask_idx) =
        Eigen::Map<Eigen::VectorX<BasicType>>(I_mask_times.data(), y_size * x_size)(mask_idx).array() /
        Eigen::Map<Eigen::VectorXi>(I_mask_counters.data(), y_size * x_size)(mask_idx).array().cast<BasicType>();
    return true;

    // if (false) {
    //     Eigen::MatrixX<BasicType> I_mask_dis = Eigen::MatrixX<BasicType>::Ones(y_size, x_size) * T_INVALID;
    //     Eigen::MatrixX<BasicType> I_mask_tmp_value = Eigen::MatrixX<BasicType>::Ones(y_size, x_size) * T_INVALID;
    //     std::vector<int> idx_vec_init, idx_vec_update;
    //     std::unordered_map<std::vector<int>, std::vector<BasicType>, Voxel2DHashFuncPrime> fine_grid;
    //     std::vector<int> key_2d(2);
    //     for (int i = 0; i < _v_obj.cols(); ++i) {
    //         key_2d[0] = std::round((max_y - _v_obj(1, i)) / (max_y - min_y) * (fine_grid_y_size - 1));
    //         key_2d[1] = std::round((_v_obj(0, i) - min_x) / (max_x - min_x) * (fine_grid_x_size - 1));
    //         if (key_2d[0] >= 0 && key_2d[0] < fine_grid_y_size && key_2d[1] >= 0 && key_2d[1] < fine_grid_x_size) {
    //             fine_grid[key_2d].push_back(_v_obj(2, i));
    //         }
    //     }
    //     std::unique_lock<std::shared_mutex> ul_mask_idx(mask_idx_lock);
    //     for (const auto &n : fine_grid) {
    //         if (n.first[0] % direct_method_config_ptr->fine_grid_times == 0 &&
    //             n.first[1] % direct_method_config_ptr->fine_grid_times == 0) {
    //             int row_idx = n.first[0] / direct_method_config_ptr->fine_grid_times;
    //             int col_idx = n.first[1] / direct_method_config_ptr->fine_grid_times;
    //             if (I_mask(row_idx, col_idx) == T_INVALID) {
    //                 I_mask(row_idx, col_idx) =
    //                     std::accumulate(n.second.begin(), n.second.end(), 0) /
    //                     static_cast<BasicType>(n.second.size());
    //                 mask_idx.push_back(col_idx * y_size + row_idx);
    //             } else {
    //                 I_mask(row_idx, col_idx) +=
    //                     std::accumulate(n.second.begin(), n.second.end(), 0) /
    //                     static_cast<BasicType>(n.second.size());
    //             }
    //             ++I_mask_times(row_idx, col_idx);
    //         }
    //     }
    // }
}

template <typename BasicType>
bool PatchProcession<BasicType>::update_mask_low_RAM(const Eigen::MatrixX<BasicType> &_v_obj) {
    std::unique_lock<std::shared_mutex> ul_mask_idx(mask_idx_lock);
    // std::vector<int> valid_idx;
    std::vector<BasicType> new_v_obj_vec;
    new_v_obj_vec.reserve(_v_obj.size() * 4);
    if (mask_idx_flags.size() != static_cast<size_t>(x_size * y_size)) {
        mask_idx_flags.assign(x_size * y_size, 0);
        for (const int idx : mask_idx_set) {
            mask_idx_flags[idx] = 1;
        }
    }
    for (int i = 0; i < _v_obj.cols(); ++i) {
        // notice the row_idx is max_y - _v_obj(1, i)
        BasicType row_idx_float = (max_y - _v_obj(1, i)) / (max_y - min_y) * (static_cast<BasicType>(y_size) - 1.0);
        BasicType col_idx_float = (_v_obj(0, i) - min_x) / (max_x - min_x) * (static_cast<BasicType>(x_size) - 1.0);
        int row_up_idx = std::ceil(row_idx_float);
        int row_down_idx = std::floor(row_idx_float);
        int col_up_idx = std::ceil(col_idx_float);
        int col_down_idx = std::floor(col_idx_float);
        // Check bounds for the current combination of row and col
        for (int row : {row_down_idx, row_up_idx}) {
            for (int col : {col_down_idx, col_up_idx}) {
                if (row >= 0 && row < y_size && col >= 0 && col < x_size) {
                    new_v_obj_vec.push_back(
                        static_cast<BasicType>(col) / (static_cast<BasicType>(x_size) - 1.0) * (max_x - min_x) + min_x);
                    new_v_obj_vec.push_back(max_y - (static_cast<BasicType>(row) /
                                                     (static_cast<BasicType>(y_size) - 1.0) * (max_y - min_y)));
                    new_v_obj_vec.push_back(_v_obj(2, i));
                    const int idx = col * y_size + row;
                    mask_idx_set.insert(idx);
                    if (!mask_idx_flags.empty()) {
                        mask_idx_flags[idx] = 1;
                    }
                    // valid_idx.push_back(i);
                }
            }
        }
    }
    // if (!valid_idx.empty()) {
    if (!new_v_obj_vec.empty()) {
        mask_idx = std::vector<int>(mask_idx_set.begin(), mask_idx_set.end());
        Eigen::Map<Eigen::MatrixX<BasicType>> new_v_obj(new_v_obj_vec.data(), 3, new_v_obj_vec.size() / 3);
        // update the AT_A_block
        // 1. get dirs
        Eigen::MatrixX<BasicType> dirs(new_v_obj.cols(), 2);
        dirs.col(0) =
            (new_v_obj(0, Eigen::all).array() - min_x) / (max_x - min_x) * 2 * M_PI * SH_table_config_ptr->SH_scale +
            SH_table_config_ptr->get_azi_low();
        dirs.col(1) =
            (new_v_obj(1, Eigen::all).array() - min_y) / (max_y - min_y) * M_PI * SH_table_config_ptr->SH_scale +
            SH_table_config_ptr->get_elev_low();
        // dirs << (x - min_x) / (max_x - min_x) * 2 * M_PI * SH_table_config_ptr->SH_scale +
        //             SH_table_config_ptr->get_azi_low(),
        //     (y - min_y) / (max_y - min_y) * M_PI * SH_table_config_ptr->SH_scale +
        //     SH_table_config_ptr->get_elev_low();
        // 2. get the Y_N for these points
        Eigen::VectorXi azi_idx = ((dirs.col(0).array() - SH_table_config_ptr->get_azi_low()) /
                                   (SH_table_config_ptr->get_azi_high() - SH_table_config_ptr->get_azi_low()) *
                                   (SH_table_config_ptr->azi_rso - 1))
                                      .array()
                                      .round()
                                      .template cast<int>();
        Eigen::VectorXi elev_idx = ((dirs.col(1).array() - SH_table_config_ptr->get_elev_low()) /
                                    (SH_table_config_ptr->get_elev_high() - SH_table_config_ptr->get_elev_low()) *
                                    (SH_table_config_ptr->elev_rso - 1))
                                       .array()
                                       .round()
                                       .template cast<int>();
        Eigen::MatrixX<BasicType> Y_N = SH_table_config_ptr->SH_table(
            azi_idx * SH_table_config_ptr->elev_rso + elev_idx, Eigen::seq(0, pow(SH_degree + 1, 2) - 1));
        AT_A_block = (AT_A_block + Y_N.transpose() * Y_N).eval();
        AT_b_block = (AT_b_block + Y_N.transpose() * new_v_obj(2, Eigen::all).transpose()).eval();
        return true;
    } else {
        return false;
    }
}

template <typename BasicType> void PatchProcession<BasicType>::extract_sph_coeff() {
    std::unique_lock<std::shared_mutex> ul_sph_coeff(sph_coeff_lock);
    std::shared_lock<std::shared_mutex> sl_mask_idx(mask_idx_lock);
    // TODO: write the new spherical harmonics extraction function and compare the result with this
    sph_coeff = (curl::IRF_least_square_table_faster<BasicType>(
                     SH_degree, Eigen::Map<Eigen::VectorX<BasicType>>(I_mask_dense.data(), x_size * y_size)(mask_idx),
                     SH_table_config_ptr->I_dirs(mask_idx, Eigen::all), SH_table_config_ptr->init_SPH_pass,
                     SH_table_config_ptr->init_SPH_granularity, *SH_table_config_ptr, SH_table_config_ptr->medium_thres,
                     SH_table_config_ptr->bad_thres, status))
                    .template cast<double>();
    recons_v_local_dirty.store(true);
    // clean memory for useless members
    I_mask_dense = Eigen::MatrixX<BasicType>();
}

template <typename BasicType> void PatchProcession<BasicType>::extract_sph_coeff_low_RAM() {
    std::unique_lock<std::shared_mutex> ul_sph_coeff(sph_coeff_lock);
    std::shared_lock<std::shared_mutex> sl_mask_idx(mask_idx_lock);

    // TODO: write the new spherical harmonics extraction function and compare the result with this
    sph_coeff = (curl::least_square_table_block_matrix_faster<BasicType>(SH_degree, AT_b_block, AT_A_block,
                                                                         *SH_table_config_ptr))
                    .template cast<double>();
    recons_v_local_dirty.store(true);
}

template <typename BasicType> void PatchProcession<BasicType>::calculate_I_without_conformal_mapping() {
    std::shared_lock<std::shared_mutex> sl_sph_coeff(sph_coeff_lock);
    std::shared_lock<std::shared_mutex> sl_mask_idx(mask_idx_lock);
    // 3 step: use sph_coeff to recons the height value
    if (I.rows() != y_size || I.cols() != x_size) {
        I = Eigen::MatrixX<BasicType>::Ones(y_size, x_size) * T_INVALID;
    }
    if (SH_table_config_ptr->is_speed_up) {
        Eigen::Map<Eigen::VectorX<BasicType>>(I.data(), y_size * x_size)(mask_idx) =
            curl::invLeastSquaresSHT_table<BasicType>(sph_coeff.cast<BasicType>(),
                                                      SH_table_config_ptr->I_dirs(mask_idx, Eigen::all),
                                                      *SH_table_config_ptr, SH_degree);
    } else {
        Eigen::Map<Eigen::VectorX<BasicType>>(I.data(), y_size * x_size)(mask_idx) =
            curl::invLeastSquaresSHT<BasicType>(sph_coeff.cast<BasicType>(),
                                                SH_table_config_ptr->I_dirs(mask_idx, Eigen::all), SH_degree);
    }

    // for pyramid
    std::unique_lock<std::shared_mutex> ul_I(I_lock);
    if (pyramids.I_vec.size() != pyramids.depth) {
        pyramids.I_vec.resize(pyramids.depth);
    }
    if (pyramids.I_Gx_vec.size() != pyramids.depth) {
        pyramids.I_Gx_vec.resize(pyramids.depth);
    }
    if (pyramids.I_Gy_vec.size() != pyramids.depth) {
        pyramids.I_Gy_vec.resize(pyramids.depth);
    }
    pyramids.I_vec[0] = I;
    for (int i = 0; i < pyramids.depth - 1; ++i) {
        // down sample
        // cv::pyrDown(img_interp_valid, img_interp_valid_down);
        pyramids.I_vec[i + 1] = curl::pyr_down<BasicType>(
            pyramids.I_vec[i], SH_table_config_ptr->gaussian_blur_kernel_5.template cast<BasicType>(), T_INVALID);
    }
}

template <typename BasicType> void PatchProcession<BasicType>::calculate_delta_without_conformal_mapping() {
    // calculate gradient for I
    std::unique_lock<std::shared_mutex> ul_I(I_lock);
    for (int i = 0; i < pyramids.depth; ++i) {
        // TODO: Try sobel gradient method
        pyramids.I_Gx_vec[i] =
            curl::eigen_central_gradient<BasicType>(pyramids.I_vec[i], "x", pyramids.x_grid_len_vec[i]);
        pyramids.I_Gy_vec[i] =
            curl::eigen_central_gradient<BasicType>(pyramids.I_vec[i], "y", pyramids.y_grid_len_vec[i]);
    }

}

template <typename BasicType> void PatchProcession<BasicType>::calculate_recons_v_local_without_conformal_mapping() {
    std::shared_lock<std::shared_mutex> sl_I(I_lock);
    std::unique_lock<std::shared_mutex> ul(v_recons_lock);
    // this is all points contains in this patch (invalid & valid)
    if (recons_v_local.rows() != y_size * x_size && recons_v_local.cols() != 3) {
        recons_v_local.resize(y_size * x_size, 3);
    }
    if (xy_grid.rows() != x_size * y_size) {
        std::unique_lock<std::shared_mutex> ul_xy_grid(xy_grid_lock);
        curl::mesh_grid<BasicType>(min_x, max_x, x_size, max_y, min_y, y_size, xy_grid);
    }
    std::shared_lock<std::shared_mutex> sl_xy_grid(xy_grid_lock);
    recons_v_local(Eigen::all, Eigen::seq(0, 1)) = xy_grid;
    recons_v_local.col(2) = Eigen::Map<Eigen::VectorX<BasicType>>(pyramids.I_vec[0].data(), pyramids.I_vec[0].size());
    recons_v_local_dirty.store(false);
}

template <typename BasicType>
void PatchProcession<BasicType>::calculate_recons_v_local_without_conformal_mapping_low_RAM() {
    std::shared_lock<std::shared_mutex> sl_sph_coeff(sph_coeff_lock);
    std::shared_lock<std::shared_mutex> sl_mask_idx(mask_idx_lock);
    std::unique_lock<std::shared_mutex> ul(v_recons_lock);
    // this is all points contains in this patch (invalid & valid)
    if (recons_v_local.rows() != y_size * x_size && recons_v_local.cols() != 3) {
        recons_v_local.resize(y_size * x_size, 3);
    }
    std::shared_lock<std::shared_mutex> sl_xy_grid(xy_grid_lock);
    // TODO: this RAM can be further reduced
    recons_v_local(Eigen::all, Eigen::seq(0, 1)) = SH_table_config_ptr->xy_grid.template cast<BasicType>();

    recons_v_local(mask_idx, 2) = curl::invLeastSquaresSHT_table<BasicType>(
        sph_coeff.cast<BasicType>(), SH_table_config_ptr->I_dirs(mask_idx, Eigen::all), *SH_table_config_ptr,
        SH_degree);
    recons_v_local_dirty.store(false);
}

template <typename BasicType>
BasicType PatchProcession<BasicType>::fetch_weights(const BasicType x, const BasicType y) {
    BasicType result = 0;
    if ((x >= min_x) && (x <= max_x) && (y >= min_y) && (y <= max_y)) {
        BasicType row_idx_float = (max_y - y) / (max_y - min_y) * (static_cast<BasicType>(y_size) - 1);
        BasicType col_idx_float = (x - min_x) / (max_x - min_x) * (static_cast<BasicType>(x_size) - 1);
        BasicType dis_squared = std::pow(row_idx_float - std::round(row_idx_float), 2) +
                                std::pow(col_idx_float - std::round(col_idx_float), 2);
        return curl::gaussian_weight_function(dis_squared);
        //        BasicType u = (x - min_x) / pyramids.x_grid_len_vec[0];
        //        // origin of the coordinate is at the left bottom corner
        //        BasicType v = (max_y - y) / pyramids.y_grid_len_vec[0];
        //        if (std::ceil(u) >= I_mask_times.cols() || std::ceil(v) >= I_mask_times.rows()) {
        //            return result;
        //        }
        //        BasicType ceil_v = std::ceil(v);
        //        BasicType floor_v = std::floor(v);
        //        BasicType ceil_u = std::ceil(u);
        //        BasicType floor_u = std::floor(u);
        //        if (ceil_v != floor_v && ceil_u != floor_u) {
        //            // bilinear interpolation https://en.wikipedia.org/wiki/Bilinear_interpolation
        //            BasicType Q11 = (I_mask_times)(static_cast<int>(ceil_v), static_cast<int>(floor_u));
        //            BasicType Q21 = (I_mask_times)(static_cast<int>(ceil_v), static_cast<int>(ceil_u));
        //            BasicType Q12 = (I_mask_times)(static_cast<int>(floor_v), static_cast<int>(floor_u));
        //            BasicType Q22 = (I_mask_times)(static_cast<int>(floor_v), static_cast<int>(ceil_u));
        //            Eigen::Vector2<BasicType> x_vec, y_vec;
        //            x_vec << ceil_u - u, u - floor_u;
        //            y_vec << v - floor_v, ceil_v - v;
        //            Eigen::Matrix2<BasicType> Q_matrix;
        //            Q_matrix << Q11, Q12, Q21, Q22;
        //            result = x_vec.transpose() * Q_matrix * y_vec;
        //            result /= ((ceil_u - floor_u) * (ceil_v - floor_v));
        //
        //        } else if (ceil_v != floor_v && ceil_u == floor_u) {
        //            BasicType Q1 = (I_mask_times)(static_cast<int>(ceil_v), static_cast<int>(u));
        //            BasicType Q2 = (I_mask_times)(static_cast<int>(floor_v), static_cast<int>(u));
        //            result = (v - floor_v) / (ceil_v - floor_v) * Q1 + (ceil_v - v) / (ceil_v - floor_v) * Q2;
        //        } else if (ceil_v == floor_v && ceil_u != floor_u) {
        //            BasicType Q1 = (I_mask_times)(static_cast<int>(v), static_cast<int>(ceil_u));
        //            BasicType Q2 = (I_mask_times)(static_cast<int>(v), static_cast<int>(floor_u));
        //            result = (u - floor_u) / (ceil_u - floor_u) * Q1 + (ceil_u - u) / (ceil_u - floor_u) * Q2;
        //        } else {
        //            result = (I_mask_times)(static_cast<int>(v), static_cast<int>(u));
        //        }
    } else {
        return result;
    }
}

template <typename BasicType>
BasicType PatchProcession<BasicType>::get_pixel_weights(const BasicType x, const BasicType y) {
    BasicType result = 0;
    if ((x >= min_x) && (x <= max_x) && (y >= min_y) && (y <= max_y)) {
        int row_idx = std::round((max_y - y) / (max_y - min_y) * (static_cast<BasicType>(y_size) - 1));
        int col_idx = std::round((x - min_x) / (max_x - min_x) * (static_cast<BasicType>(x_size) - 1));
        return I_mask_weights(row_idx, col_idx);
    } else {
        return result;
    }
}

template <typename BasicType>
BasicType PatchProcession<BasicType>::fetch_I_without_conformal_mapping(const std::string &matrix_name,
                                                                        const BasicType x, const BasicType y,
                                                                        const int pyramid_idx) {
    MatrixType matrix_type;
    if (matrix_name == "I") {
        matrix_type = MatrixType::I;
    } else if (matrix_name == "Gx") {
        matrix_type = MatrixType::Gx;
    } else if (matrix_name == "Gy") {
        matrix_type = MatrixType::Gy;
    } else {
        std::cerr << "Invalid matrix name!" << std::endl;
        return T_INVALID;
    }
    return fetch_I_without_conformal_mapping(matrix_type, x, y, pyramid_idx);
}

template <typename BasicType>
BasicType PatchProcession<BasicType>::fetch_I_without_conformal_mapping(MatrixType matrix_type, const BasicType x,
                                                                        const BasicType y, const int pyramid_idx) {
    if (is_cleared) {
        recover_data();
    }
    std::shared_lock<std::shared_mutex> sl_I(I_lock);
    const Eigen::MatrixX<BasicType> *I_matrix_ptr = nullptr;
    switch (matrix_type) {
    case MatrixType::I:
        I_matrix_ptr = &pyramids.I_vec[pyramid_idx];
        break;
    case MatrixType::Gx:
        I_matrix_ptr = &pyramids.I_Gx_vec[pyramid_idx];
        break;
    case MatrixType::Gy:
        I_matrix_ptr = &pyramids.I_Gy_vec[pyramid_idx];
        break;
    default:
        return T_INVALID;
    }

    BasicType result = T_INVALID;
    if ((x >= min_x) && (x <= max_x) && (y >= min_y) && (y <= max_y)) {
        const BasicType u = (x - min_x) / pyramids.x_grid_len_vec[pyramid_idx];
        // origin of the coordinate is at the left bottom corner
        const BasicType v = (max_y - y) / pyramids.y_grid_len_vec[pyramid_idx];
        const BasicType ceil_v = std::ceil(v);
        const BasicType floor_v = std::floor(v);
        const BasicType ceil_u = std::ceil(u);
        const BasicType floor_u = std::floor(u);

        if (ceil_u >= I_matrix_ptr->cols() || ceil_v >= I_matrix_ptr->rows() || floor_u < 0 || floor_v < 0) {
            return result;
        }

        const int cu = static_cast<int>(ceil_u);
        const int fu = static_cast<int>(floor_u);
        const int cv = static_cast<int>(ceil_v);
        const int fv = static_cast<int>(floor_v);

        if (cv != fv && cu != fu) {
            const BasicType Q11 = (*I_matrix_ptr)(cv, fu);
            const BasicType Q21 = (*I_matrix_ptr)(cv, cu);
            const BasicType Q12 = (*I_matrix_ptr)(fv, fu);
            const BasicType Q22 = (*I_matrix_ptr)(fv, cu);
            if ((Q11 != T_INVALID) && (Q21 != T_INVALID) && (Q12 != T_INVALID) && (Q22 != T_INVALID)) {
                const BasicType du = u - floor_u;
                const BasicType dv = v - floor_v;
                result = (static_cast<BasicType>(1) - du) * dv * Q11 + du * dv * Q21 +
                         (static_cast<BasicType>(1) - du) * (static_cast<BasicType>(1) - dv) * Q12 +
                         du * (static_cast<BasicType>(1) - dv) * Q22;
            }
        } else if (cv != fv && cu == fu) {
            const BasicType Q1 = (*I_matrix_ptr)(cv, fu);
            const BasicType Q2 = (*I_matrix_ptr)(fv, fu);
            if (Q1 != T_INVALID && Q2 != T_INVALID) {
                const BasicType dv = v - floor_v;
                result = dv * Q1 + (static_cast<BasicType>(1) - dv) * Q2;
            }
        } else if (cv == fv && cu != fu) {
            const BasicType Q1 = (*I_matrix_ptr)(fv, cu);
            const BasicType Q2 = (*I_matrix_ptr)(fv, fu);
            if (Q1 != T_INVALID && Q2 != T_INVALID) {
                const BasicType du = u - floor_u;
                result = du * Q1 + (static_cast<BasicType>(1) - du) * Q2;
            }
        } else {
            result = (*I_matrix_ptr)(fv, fu);
        }
    }
    return result;
}

template <typename BasicType>
void PatchProcession<BasicType>::update_projection_plane(const Eigen::MatrixX<BasicType> &v_obj,
                                                         const Eigen::VectorX<BasicType> &_v_range_squared) {
    reset();
    if (update_mask(v_obj, _v_range_squared)) {
        extract_sph_coeff();
        calculate_I_without_conformal_mapping();
        calculate_delta_without_conformal_mapping();
        calculate_recons_v_local_without_conformal_mapping();
        update_step_counter = 0;
        ++initialization_counter;
    }
}

template <typename BasicType>
void PatchProcession<BasicType>::update_projection_plane_low_RAM(const Eigen::MatrixX<BasicType> &v_obj) {
    reset_low_RAM();
    if (update_mask_low_RAM(v_obj)) {
        extract_sph_coeff_low_RAM();
        // calculate_recons_v_local_without_conformal_mapping_low_RAM();
        update_step_counter = 0;
        ++initialization_counter;
    }
}

template <typename BasicType>
bool PatchProcession<BasicType>::update_sph_residuals_without_conformal_mapping(
    const Eigen::MatrixX<BasicType> &v_obj, const Eigen::VectorX<BasicType> &_v_range_squared,
    const Eigen::Vector3d &current_postition) {
    if (is_cleared) {
        recover_data();
    }
    // this mask has the same resolution of the image
    //    if (initialize_mask(v_obj)) {
    if (update_mask(v_obj, _v_range_squared)) {
        if (is_initialization) {
            ++update_step_counter;
            if (update_step_counter > direct_method_config_ptr->update_steps) {
                extract_sph_coeff();
                calculate_I_without_conformal_mapping();
                calculate_delta_without_conformal_mapping();
                calculate_recons_v_local_without_conformal_mapping();
                update_step_counter = 0;
                return true;
            }
            if (initialization_counter > direct_method_config_ptr->update_initialization_times &&
                direct_method_config_ptr->update_initialization_times != -1) {
                is_initialization = false;
            }
            ++initialization_counter;
        } else {
            if ((current_postition - last_update_position).norm() > direct_method_config_ptr->update_distance) {
                extract_sph_coeff();
                calculate_I_without_conformal_mapping();
                calculate_delta_without_conformal_mapping();
                calculate_recons_v_local_without_conformal_mapping();
                last_update_position = current_postition;
                return true;
            }
        }
    }
    return false;
}

template <typename BasicType>
bool PatchProcession<BasicType>::update_sph_residuals_without_conformal_mapping_low_RAM(
    const Eigen::MatrixX<BasicType> &v_obj, const Eigen::Vector3d &current_postition) {
    if (update_mask_low_RAM(v_obj)) {
        if (is_initialization) {
            ++update_step_counter;
            if (update_step_counter > direct_method_config_ptr->update_steps) {
                extract_sph_coeff_low_RAM();
                // calculate_recons_v_local_without_conformal_mapping_low_RAM();
                update_step_counter = 0;
                return true;
            }
            if (initialization_counter > direct_method_config_ptr->update_initialization_times &&
                direct_method_config_ptr->update_initialization_times != -1) {
                is_initialization = false;
            }
            ++initialization_counter;
        } else {
            if ((current_postition - last_update_position).norm() > direct_method_config_ptr->update_distance) {
                extract_sph_coeff_low_RAM();
                // calculate_recons_v_local_without_conformal_mapping_low_RAM();
                last_update_position = current_postition;
                return true;
            }
        }
    }
    return false;
}

template <typename BasicType>
bool PatchProcession<BasicType>::icp_update_recons_v(const Eigen::MatrixX<BasicType> &v_obj) {
    if (v_obj.cols() > 0) {
        pcl::PointCloud<pcl::PointXYZINormal> new_cloud;
        for (int i = 0; i < v_obj.cols(); ++i) {
            pcl::PointXYZINormal pt;
            pt.x = v_obj(0, i);
            pt.y = v_obj(1, i);
            pt.z = v_obj(2, i);
            new_cloud.push_back(pt);
        }
        *recons_v_local_normal_ptr += new_cloud;
        curl::subSampleFrame(*recons_v_local_normal_ptr, direct_method_config_ptr->minimum_rso);
        // estimate
        pcl::NormalEstimation<pcl::PointXYZINormal, pcl::PointXYZINormal> norm_est;
        pcl::search::KdTree<pcl::PointXYZINormal>::Ptr normal_kdtree(new pcl::search::KdTree<pcl::PointXYZINormal>);
        norm_est.setSearchMethod(normal_kdtree);
        // Specify the size of the local neighborhood to use when
        // computing the surface normals
        if (is_ground) {
            norm_est.setRadiusSearch(3);
        } else {
            norm_est.setRadiusSearch(0.3);
        }
        norm_est.setInputCloud(recons_v_local_normal_ptr);
        norm_est.compute(*recons_v_local_normal_ptr);
        kd_tree_ptr->setInputCloud(recons_v_local_normal_ptr);
        std::unique_lock<std::shared_mutex> ul_mask_idx(mask_idx_lock);
        std::unique_lock<std::shared_mutex> ul_recons(v_recons_lock);
        mask_idx.resize(recons_v_local_normal_ptr->size());
        recons_v_local.resize(recons_v_local_normal_ptr->size(), 3);
        for (int i = 0; i < recons_v_local_normal_ptr->size(); ++i) {
            recons_v_local(i, 0) = recons_v_local_normal_ptr->at(i).x;
            recons_v_local(i, 1) = recons_v_local_normal_ptr->at(i).y;
            recons_v_local(i, 2) = recons_v_local_normal_ptr->at(i).z;
        }
        std::iota(mask_idx.begin(), mask_idx.end(), 0);
        recons_v_local_dirty.store(false);
        return true;
    } else {
        return false;
    }
}

template <typename BasicType> void PatchProcession<BasicType>::recover_data() {
    if (is_cleared) {
        calculate_I_without_conformal_mapping();
        calculate_delta_without_conformal_mapping();
        calculate_recons_v_local_without_conformal_mapping();
        Eigen::Map<Eigen::VectorX<BasicType>>(I_mask_weights.data(), y_size * x_size)(mask_idx) =
            Eigen::Map<Eigen::VectorX<BasicType>>(I_mask_times.data(), y_size * x_size)(mask_idx).array() /
            Eigen::Map<Eigen::VectorXi>(I_mask_counters.data(), y_size * x_size)(mask_idx).array().cast<BasicType>();
        is_cleared = false;
    }
}

template <typename BasicType> void PatchProcession<BasicType>::reset() {
    std::unique_lock<std::shared_mutex> ul_sph_coeff(sph_coeff_lock);
    std::unique_lock<std::shared_mutex> ul_mask_idx(mask_idx_lock);
    I_mask = Eigen::MatrixX<BasicType>::Ones(y_size, x_size) * T_INVALID;
    I_mask_times.setZero(y_size, x_size);
    I_mask_weights.setZero(y_size, x_size);
    I_mask_counters.setZero(y_size, x_size);
    sph_coeff = Eigen::VectorXd();
    recons_v_local_dirty.store(true);
    mask_idx.clear();
    if (x_size > 0 && y_size > 0) {
        mask_idx_flags.assign(x_size * y_size, 0);
    } else {
        mask_idx_flags.clear();
    }

    //        I_mask_weights = Eigen::MatrixX<BasicType>::Ones(y_size, x_size) * T_INVALID;
    I_mask_weights.setZero(y_size, x_size);
    clear_content();
    is_cleared = false;
}

template <typename BasicType> void PatchProcession<BasicType>::reset_low_RAM() {
    std::unique_lock<std::shared_mutex> ul_sph_coeff(sph_coeff_lock);
    std::unique_lock<std::shared_mutex> ul_mask_idx(mask_idx_lock);
    AT_A_block.setZero();
    AT_b_block.setZero();
    sph_coeff = Eigen::VectorXd();
    recons_v_local_dirty.store(true);
    mask_idx_set.clear();
    mask_idx.clear();
    if (x_size > 0 && y_size > 0) {
        mask_idx_flags.assign(x_size * y_size, 0);
    } else {
        mask_idx_flags.clear();
    }
    is_cleared = false;
}

template <typename BasicType> void PatchProcession<BasicType>::clear_content() {
    // I_vec I_Gx_vec I_Gy_vec I_mask_dense
    std::unique_lock<std::shared_mutex> ul_I(I_lock);
    std::unique_lock<std::shared_mutex> ul_v_recons(v_recons_lock);
    std::vector<Eigen::MatrixX<BasicType>> empty;
    pyramids.I_vec.swap(empty);
    pyramids.I_Gx_vec.swap(empty);
    pyramids.I_Gy_vec.swap(empty);
    I_mask_dense = Eigen::MatrixX<BasicType>();
    I = Eigen::MatrixX<BasicType>();
    std::vector<int> empty_vec;
    recons_v_local = Eigen::MatrixX<BasicType>();
    recons_v_local_dirty.store(true);
    {
        std::unique_lock<std::shared_mutex> ul_xy_grid(xy_grid_lock);
        xy_grid = Eigen::MatrixX<BasicType>();
    }
    is_cleared = true;
}

template <typename BasicType>
std::pair<BasicType, int> PatchProcession<BasicType>::hist_fall_thres(std::vector<BasicType> &vec,
                                                                      BasicType percentage) {
    std::sort(vec.begin(), vec.end());
    int n = 0;
    if ((vec.size() % 2) == 0) {
        n = static_cast<int>(vec.size() / 2);
    } else {
        n = static_cast<int>((vec.size() - 1) / 2);
    }
    BasicType Q1, Q3;
    if ((n % 2) != 0) {
        Q1 = vec[n / 2];
        Q3 = vec[(vec.size() * 2 - n) / 2];
    } else {
        Q1 = (vec[n / 2 - 1] + vec[n / 2]) / 2;
        Q3 = (vec[(vec.size() - n + vec.size() - 1) / 2] + vec[(vec.size() - n + vec.size() - 1) / 2 + 1]) / 2;
    }
    BasicType IQR = Q3 - Q1;
    BasicType bin_width = 2 * IQR * pow(static_cast<BasicType>(vec.size()), -1 / 3);
    int last_bin_idx = 0;
    int current_bin_idx = 0;
    BasicType last_max_counter = 0.0;
    BasicType current_counter = 0.0;
    percentage = 1 - percentage;
    //    for (auto &item : vec) {
    int i;
    for (i = 0; i < vec.size(); ++i) {
        current_bin_idx = ceil(vec[i] / bin_width);
        if (last_bin_idx == current_bin_idx) {
            ++current_counter;
        } else {
            // always compare with the bin idx contain the largest number of elements in the past
            if ((current_counter / last_max_counter) < percentage) {
                break;
            } else if (current_counter > last_max_counter) {
                last_max_counter = current_counter;
            }
            current_counter = 0;
            last_bin_idx = current_bin_idx;
        }
    }
    BasicType thres = static_cast<BasicType>(current_bin_idx) * bin_width;
    return std::make_pair(thres, i);
}

template <typename BasicType> std::vector<Eigen::MatrixX<BasicType>> PatchProcession<BasicType>::get_I_vec() {
    if (is_cleared) {
        recover_data();
    }
    std::shared_lock<std::shared_mutex> sl_I(I_lock);
    return pyramids.I_vec;
}

template <typename BasicType> std::vector<Eigen::MatrixX<BasicType>> PatchProcession<BasicType>::get_I_Gx_vec() {
    if (is_cleared) {
        recover_data();
    }
    std::shared_lock<std::shared_mutex> sl_I(I_lock);
    return pyramids.I_Gx_vec;
}

template <typename BasicType> std::vector<Eigen::MatrixX<BasicType>> PatchProcession<BasicType>::get_I_Gy_vec() {
    if (is_cleared) {
        recover_data();
    }
    std::shared_lock<std::shared_mutex> sl_I(I_lock);
    return pyramids.I_Gy_vec;
}

template <typename BasicType> Eigen::MatrixX<BasicType> PatchProcession<BasicType>::get_I(const int pyramid_idx) {
    if (pyramids.I_vec.size() != pyramids.depth) {
        if (is_cleared) {
            recover_data();
        } else {
            return Eigen::MatrixX<BasicType>();
        }
    }
    std::shared_lock<std::shared_mutex> sl_I(I_lock);
    return pyramids.I_vec[pyramid_idx];
}

template <typename BasicType> Eigen::MatrixX<BasicType> PatchProcession<BasicType>::get_I_Gx(const int pyramid_idx) {
    if (pyramids.I_Gx_vec.size() != pyramids.depth) {
        if (is_cleared) {
            recover_data();
        } else {
            return Eigen::MatrixX<BasicType>();
        }
    }
    std::shared_lock<std::shared_mutex> sl_I(I_lock);
    return pyramids.I_Gx_vec[pyramid_idx];
}

template <typename BasicType> Eigen::MatrixX<BasicType> PatchProcession<BasicType>::get_I_Gy(const int pyramid_idx) {
    if (pyramids.I_Gy_vec.size() != pyramids.depth) {
        if (is_cleared) {
            recover_data();
        } else {
            return Eigen::MatrixX<BasicType>();
        }
    }
    std::shared_lock<std::shared_mutex> sl_I(I_lock);
    return pyramids.I_Gy_vec[pyramid_idx];
}

template <typename BasicType> Eigen::MatrixX<BasicType> PatchProcession<BasicType>::get_recons_v_obj() {
    if (is_cleared) {
        recover_data();
    }
    Eigen::MatrixX<BasicType> recons_v_local_tmp(mask_idx.size(), 3);
    recons_v_local_tmp(Eigen::all, Eigen::seq(0, 1)) =
        SH_table_config_ptr->xy_grid(mask_idx, Eigen::all).template cast<BasicType>();

    recons_v_local_tmp(Eigen::all, 2) = curl::invLeastSquaresSHT_table<BasicType>(
        sph_coeff.cast<BasicType>(), SH_table_config_ptr->I_dirs(mask_idx, Eigen::all), *SH_table_config_ptr,
        SH_degree);
    return recons_v_local_tmp;
}

template <typename BasicType> Eigen::MatrixX<BasicType> PatchProcession<BasicType>::get_recons_v_obj_cached() {
    update_recons_v_local_if_dirty();
    std::shared_lock<std::shared_mutex> sl_mask_idx(mask_idx_lock);
    std::shared_lock<std::shared_mutex> sl_recons(v_recons_lock);
    if (recons_v_local.rows() == 0 || mask_idx.empty()) {
        return Eigen::MatrixX<BasicType>();
    }
    return recons_v_local(mask_idx, Eigen::all);
}

template <typename BasicType> Eigen::MatrixX<BasicType> PatchProcession<BasicType>::get_recons_v_local_cached() {
    update_recons_v_local_if_dirty();
    std::shared_lock<std::shared_mutex> sl_recons(v_recons_lock);
    return recons_v_local;
}

template <typename BasicType> void PatchProcession<BasicType>::mark_recons_v_local_dirty() {
    recons_v_local_dirty.store(true);
}

template <typename BasicType>
Eigen::MatrixX<BasicType> PatchProcession<BasicType>::get_recons_v_obj_downsample(const int stride) {
    if (stride <= 1) {
        return get_recons_v_obj();
    }
    if (is_cleared) {
        recover_data();
    }
    std::vector<int> mask_idx_downsampled;
    mask_idx_downsampled.reserve(mask_idx.size() / stride + 1);
    const int step = std::max(1, stride);
    for (const int idx : mask_idx) {
        // mask_idx uses column-major indexing: idx = col * y_size + row
        const int row = idx % y_size;
        const int col = idx / y_size;
        if ((row % step) == 0 && (col % step) == 0) {
            mask_idx_downsampled.push_back(idx);
        }
    }
    if (mask_idx_downsampled.empty()) {
        return Eigen::MatrixX<BasicType>();
    }
    Eigen::MatrixX<BasicType> recons_v_local_tmp(mask_idx_downsampled.size(), 3);
    recons_v_local_tmp(Eigen::all, Eigen::seq(0, 1)) =
        SH_table_config_ptr->xy_grid(mask_idx_downsampled, Eigen::all).template cast<BasicType>();
    recons_v_local_tmp(Eigen::all, 2) = curl::invLeastSquaresSHT_table<BasicType>(
        sph_coeff.cast<BasicType>(), SH_table_config_ptr->I_dirs(mask_idx_downsampled, Eigen::all),
        *SH_table_config_ptr, SH_degree);
    return recons_v_local_tmp;
}

template <typename BasicType> void PatchProcession<BasicType>::update_recons_v_local_if_dirty() {
    if (!recons_v_local_dirty.load()) {
        return;
    }
    if (is_cleared) {
        recover_data();
        return;
    }
    if (SH_table_config_ptr && SH_table_config_ptr->is_SH_analytic_jacobian) {
        calculate_recons_v_local_without_conformal_mapping_low_RAM();
    } else {
        calculate_recons_v_local_without_conformal_mapping();
    }
}

template <typename BasicType> bool PatchProcession<BasicType>::is_recons_v_local_dirty() const {
    return recons_v_local_dirty.load();
}

template <typename BasicType>
BasicType PatchProcession<BasicType>::get_recons_height(const Eigen::VectorXd &_sph_coeff, const BasicType x,
                                                        const BasicType y) {
    if (is_cleared) {
        recover_data();
    }
    if (x >= min_x && x <= max_x && y >= min_y && y <= max_y) {
        // Note: check whether this point is on the invalid area using mask
        // double u_d = (x - min_x) / pyramids.x_grid_len_vec[0];
        // double v_d = (max_y - y) / pyramids.y_grid_len_vec[0];
        // int u_floor = std::floor(u_d);
        // int u_ceil = std::ceil(u_d);
        // // origin of the coordinate is at the left bottom corner
        // int v_floor = std::floor(v_d);
        // int v_ceil = std::ceil(v_d);
        // if (I_mask(v_floor, u_floor) == T_INVALID || I_mask(v_floor, u_ceil) == T_INVALID ||
        //     I_mask(v_ceil, u_floor) == T_INVALID || I_mask(v_ceil, u_ceil) == T_INVALID) {
        //     return T_INVALID;
        // }
        Eigen::MatrixX<BasicType> dirs(1, 2);
        dirs << (x - min_x) / (max_x - min_x) * 2 * M_PI * SH_table_config_ptr->SH_scale +
                    SH_table_config_ptr->get_azi_low(),
            (y - min_y) / (max_y - min_y) * M_PI * SH_table_config_ptr->SH_scale + SH_table_config_ptr->get_elev_low();
        Eigen::VectorX<BasicType> value = curl::invLeastSquaresSHT_table<BasicType>(_sph_coeff.cast<BasicType>(), dirs,
                                                                                    *SH_table_config_ptr, SH_degree);
        return value(0);
    } else {
        return T_INVALID;
    }
}

template <typename BasicType> BasicType PatchProcession<BasicType>::get_theta(const BasicType y) {
    if (is_cleared) {
        recover_data();
    }
    if (max_y <= min_y) {
        std::cerr << "[CURL_SLAM] get_theta invalid y range: min_y=" << min_y << " max_y=" << max_y << std::endl;
        std::abort();
    }
    if (y >= min_y && y <= max_y) {
        return (y - min_y) / (max_y - min_y) * M_PI * SH_table_config_ptr->SH_scale +
               SH_table_config_ptr->get_elev_low();
    } else {
        return T_INVALID;
    }
}

template <typename BasicType> BasicType PatchProcession<BasicType>::get_phi(const BasicType x) {
    if (is_cleared) {
        recover_data();
    }
    if (max_x <= min_x) {
        std::cerr << "[CURL_SLAM] get_phi invalid x range: min_x=" << min_x << " max_x=" << max_x << std::endl;
        std::abort();
    }
    if (x >= min_x && x <= max_x) {
        return (x - min_x) / (max_x - min_x) * 2 * M_PI * SH_table_config_ptr->SH_scale +
               SH_table_config_ptr->get_azi_low();
    } else {
        return T_INVALID;
    }
}

template <typename BasicType> std::vector<int> PatchProcession<BasicType>::get_mask_idx() {
    std::shared_lock<std::shared_mutex> sl_mask_idx(mask_idx_lock);
    return mask_idx;
}

template <typename BasicType> std::vector<int> PatchProcession<BasicType>::get_BA_mask_idx() { return BA_mask_idx; }

template <typename BasicType>
bool PatchProcession<BasicType>::is_valid_in_BA_mask(const BasicType x, const BasicType y) {
    if (is_cleared) {
        recover_data();
    }
    if (x >= min_x && x <= max_x && y >= min_y && y <= max_y) {
        int u = std::round((x - min_x) / pyramids.x_grid_len_vec[0]);
        int v = std::round((max_y - y) / pyramids.y_grid_len_vec[0]);
        if (u >= 0 && u < x_size && v >= 0 && v < y_size) {
            if (BA_I_mask(v, u) != T_INVALID) {
                return true;
            }
        }
    }
    return false;
}

template <typename BasicType> Eigen::MatrixX<BasicType> PatchProcession<BasicType>::get_I_mask() { return I_mask; }

template <typename BasicType> bool PatchProcession<BasicType>::is_valid_in_mask(const BasicType x, const BasicType y) {
    if (is_cleared) {
        recover_data();
    }
    if (x >= min_x && x <= max_x && y >= min_y && y <= max_y) {
        int u = std::round((x - min_x) / pyramids.x_grid_len_vec[0]);
        int v = std::round((max_y - y) / pyramids.y_grid_len_vec[0]);
        if (u >= 0 && u < x_size && v >= 0 && v < y_size) {
            if (I_mask(v, u) != T_INVALID) {
                return true;
            }
        }
    }
    return false;
}

template <typename BasicType>
bool PatchProcession<BasicType>::is_valid_in_mask_low_RAM(const BasicType x, const BasicType y) {
    if (x >= min_x && x <= max_x && y >= min_y && y <= max_y) {
        // Note: check whether this point is on the invalid area using mask
        int u = std::round((x - min_x) / pyramids.x_grid_len_vec[0]);
        int v = std::round((max_y - y) / pyramids.y_grid_len_vec[0]);

        int idx = u * y_size + v;

        std::shared_lock<std::shared_mutex> sl_mask_idx(mask_idx_lock);
        if (!mask_idx_flags.empty()) {
            return mask_idx_flags[idx] != 0;
        }
        return mask_idx_set.find(idx) != mask_idx_set.end();
    }
    return false;
}

template <typename BasicType>
BasicType PatchProcession<BasicType>::get_height_value_BA_mask(const BasicType x, const BasicType y) {
    if (x >= min_x && x <= max_x && y >= min_y && y <= max_y) {
        int u = std::round((x - min_x) / pyramids.x_grid_len_vec[0]);
        int v = std::round((max_y - y) / pyramids.y_grid_len_vec[0]);
        if (u >= 0 && u < x_size && v >= 0 && v < y_size) {
            return BA_I_mask(v, u);
        }
    }
    return T_INVALID;
}

template <typename BasicType> int PatchProcession<BasicType>::get_pyramid_depth() const { return pyramids.depth; }

template <typename BasicType> double PatchProcession<BasicType>::get_score() { return mask_idx.size(); }

template <typename BasicType> double *PatchProcession<BasicType>::get_sph_coeff_data() {
    std::shared_lock<std::shared_mutex> sl_sph_coeff(sph_coeff_lock);
    return sph_coeff.data();
}

template <typename BasicType> double *PatchProcession<BasicType>::set_and_get_BA_sph_coeff_data() {
    std::unique_lock<std::shared_mutex> ul_sph_coeff(sph_coeff_lock);
    BA_sph_coeff = sph_coeff;
    is_set_BA_sph_coeff = true;
    return BA_sph_coeff.data();
}

template <typename BasicType> double *PatchProcession<BasicType>::get_BA_sph_coeff_data() {
    if (is_set_BA_sph_coeff) {
        return BA_sph_coeff.data();
    } else {
        std::shared_lock<std::shared_mutex> sl_sph_coeff(sph_coeff_lock);
        BA_sph_coeff = sph_coeff;
        is_set_BA_sph_coeff = true;
        return BA_sph_coeff.data();
    }
}

template <typename BasicType> void PatchProcession<BasicType>::commit_BA_sph_coeff() {
    std::unique_lock<std::shared_mutex> ul_sph_coeff(sph_coeff_lock);
    if (is_set_BA_sph_coeff) {
        sph_coeff = BA_sph_coeff;
    }
}

template <typename BasicType> const Eigen::VectorXd &PatchProcession<BasicType>::get_sph_coeff() {
    std::shared_lock<std::shared_mutex> sl_sph_coeff(sph_coeff_lock);
    return sph_coeff;
}

template <typename BasicType> const Eigen::VectorXd &PatchProcession<BasicType>::get_BA_sph_coeff() {
    if (is_set_BA_sph_coeff) {
        return BA_sph_coeff;
    } else {
        std::shared_lock<std::shared_mutex> sl_sph_coeff(sph_coeff_lock);
        BA_sph_coeff = sph_coeff;
        is_set_BA_sph_coeff = true;
        return BA_sph_coeff;
    }
}

template <typename BasicType> double PatchProcession<BasicType>::get_sph_coeff_sum() {
    std::shared_lock<std::shared_mutex> sl_sph_coeff(sph_coeff_lock);
    return sph_coeff.cwiseAbs().sum();
}

template <typename BasicType> double PatchProcession<BasicType>::get_G_theta() const noexcept { return G_theta; }

template <typename BasicType> double PatchProcession<BasicType>::get_G_phi() const noexcept { return G_phi; }

template <typename BasicType> int PatchProcession<BasicType>::get_SH_degree() const noexcept { return SH_degree; }

template <typename BasicType> void PatchProcession<BasicType>::set_sph_coeff_lock() {
    if (!is_sph_coeff_lock) {
        sph_coeff_lock.lock();
        is_sph_coeff_lock = true;
    }
}

template <typename BasicType> void PatchProcession<BasicType>::unset_sph_coeff_lock() {
    if (is_sph_coeff_lock) {
        sph_coeff_lock.unlock();
        is_sph_coeff_lock = false;
    }
}

template <typename BasicType>
Eigen::MatrixX<BasicType>
PatchProcession<BasicType>::interpolates_xy_grid_points(const Eigen::MatrixX<BasicType> &seed_points) {
    if (xy_grid.rows() != x_size * y_size) {
        std::unique_lock<std::shared_mutex> ul_xy_grid(xy_grid_lock);
        curl::mesh_grid<BasicType>(min_x, max_x, x_size, max_y, min_y, y_size, xy_grid);
    }
    interp_func_pair F = linear_interpolation_2_func<BasicType>(Eigen::VectorX<BasicType>(seed_points.row(0)),
                                                                Eigen::VectorX<BasicType>(seed_points.row(1)),
                                                                Eigen::VectorX<BasicType>(seed_points.row(2)));
    BasicType z_boundary[2] = {seed_points.row(2).minCoeff(), seed_points.row(2).maxCoeff()};
    std::shared_lock<std::shared_mutex> sl_xy_grid(xy_grid_lock);
    std::pair<std::vector<BasicType>, std::vector<int>> interp_result = linear_barycentric_interpolation_2<BasicType>(
        xy_grid, seed_points.row(2), F, z_boundary, direct_method_config_ptr->mask_gap_squared);
    Eigen::MatrixX<BasicType> sampled_points(3, interp_result.second.size());
    sampled_points(Eigen::seq(0, 1), Eigen::all) = xy_grid(interp_result.second, Eigen::all).transpose();
    sampled_points.row(2) = Eigen::Map<Eigen::VectorX<BasicType>>(interp_result.first.data(),
                                                                  interp_result.first.size())(interp_result.second);
    return sampled_points;
}

template <typename BasicType> Eigen::MatrixX<BasicType> PatchProcession<BasicType>::get_xy_grid_dense(const int times) {
    Eigen::MatrixX<BasicType> xy_grid_dense;
    std::unique_lock<std::shared_mutex> ul_xy_grid(xy_grid_lock);
    curl::mesh_grid<BasicType>(min_x, max_x, x_size * times, max_y, min_y, y_size * times, xy_grid_dense);
    return xy_grid_dense;
}

template <typename BasicType>
std::pair<bool, bool> PatchProcession<BasicType>::check_is_on_edge(const BasicType x, const BasicType y) {
    if (is_cleared) {
        recover_data();
    }
    std::pair<bool, bool> is_on_edge(true, true);
    if (x >= min_x && x <= max_x && y >= min_y && y <= max_y) {
        // Note: check whether this point is on the invalid area using mask
        int u = std::round((x - min_x) / pyramids.x_grid_len_vec[0]);
        int v = std::round((max_y - y) / pyramids.y_grid_len_vec[0]);
        std::shared_lock<std::shared_mutex> sl_I(I_lock);
        if (pyramids.I_vec[0](v, u) != T_INVALID) {
            int u_prev = u - 2;
            int u_next = u + 2;
            if (u_prev >= 0 && u_next < pyramids.I_vec[0].cols()) {
                if (pyramids.I_vec[0](v, u_prev) != T_INVALID && pyramids.I_vec[0](v, u_next) != T_INVALID) {
                    is_on_edge.first = false;
                }
            }
            // origin of the coordinate is at the left bottom corner
            int v_prev = v - 2;
            int v_next = v + 2;
            if (v_prev >= 0 && v_next < pyramids.I_vec[0].rows()) {
                if (pyramids.I_vec[0](v_prev, u) != T_INVALID && pyramids.I_vec[0](v_next, u) != T_INVALID) {
                    is_on_edge.second = false;
                }
            }
        }
    }
    return is_on_edge;
}

template <typename BasicType>
std::pair<bool, bool> PatchProcession<BasicType>::check_is_on_edge_low_RAM(const BasicType x, const BasicType y) {
    std::pair<bool, bool> is_on_edge(true, true);
    if (x >= min_x && x <= max_x && y >= min_y && y <= max_y) {
        // Note: check whether this point is on the invalid area using mask
        int u = std::round((x - min_x) / pyramids.x_grid_len_vec[0]);
        int v = std::round((max_y - y) / pyramids.y_grid_len_vec[0]);

        int idx = u * y_size + v;

        std::shared_lock<std::shared_mutex> sl_mask_idx(mask_idx_lock);
        const bool has_mask = (!mask_idx_flags.empty()) ? (mask_idx_flags[idx] != 0)
                                                        : (mask_idx_set.find(idx) != mask_idx_set.end());
        if (has_mask) {
            int u_prev = u - 2;
            int u_next = u + 2;
            if (u_prev >= 0 && u_next < x_size) {
                const int idx_prev = u_prev * y_size + v;
                const int idx_next = u_next * y_size + v;
                const bool has_prev = (!mask_idx_flags.empty())
                                          ? (mask_idx_flags[idx_prev] != 0)
                                          : (mask_idx_set.find(idx_prev) != mask_idx_set.end());
                const bool has_next = (!mask_idx_flags.empty())
                                          ? (mask_idx_flags[idx_next] != 0)
                                          : (mask_idx_set.find(idx_next) != mask_idx_set.end());
                if (has_prev && has_next) {
                    is_on_edge.first = false;
                }
            }
            // origin of the coordinate is at the left bottom corner
            int v_prev = v - 2;
            int v_next = v + 2;
            if (v_prev >= 0 && v_next < y_size) {
                const int idx_prev = u * y_size + v_prev;
                const int idx_next = u * y_size + v_next;
                const bool has_prev = (!mask_idx_flags.empty())
                                          ? (mask_idx_flags[idx_prev] != 0)
                                          : (mask_idx_set.find(idx_prev) != mask_idx_set.end());
                const bool has_next = (!mask_idx_flags.empty())
                                          ? (mask_idx_flags[idx_next] != 0)
                                          : (mask_idx_set.find(idx_next) != mask_idx_set.end());
                if (has_prev && has_next) {
                    is_on_edge.second = false;
                }
            }
        }
    }
    return is_on_edge;
}

template <typename BasicType>
std::pair<bool, bool> PatchProcession<BasicType>::BA_check_is_on_edge_simple(const BasicType x, const BasicType y) {
    std::pair<bool, bool> is_on_edge(true, true);
    if (x >= min_x && x <= max_x && y >= min_y && y <= max_y) {
        // Note: check whether this point is on the invalid area using mask
        int u = std::round((x - min_x) / pyramids.x_grid_len_vec[0]);
        int v = std::round((max_y - y) / pyramids.y_grid_len_vec[0]);
        if (BA_I_mask(v, u) != T_INVALID) {
            int u_prev = u - 2;
            int u_next = u + 2;
            if (u_prev >= 0 && u_next < BA_I_mask.cols()) {
                if (BA_I_mask(v, u_prev) != T_INVALID && BA_I_mask(v, u_next) != T_INVALID) {
                    is_on_edge.first = false;
                }
            }
            // origin of the coordinate is at the left bottom corner
            int v_prev = v - 2;
            int v_next = v + 2;
            if (v_prev >= 0 && v_next < BA_I_mask.rows()) {
                if (BA_I_mask(v_prev, u) != T_INVALID && BA_I_mask(v_next, u) != T_INVALID) {
                    is_on_edge.second = false;
                }
            }
        }
    }
    return is_on_edge;
}

template <typename BasicType>
std::pair<bool, bool> PatchProcession<BasicType>::BA_check_is_on_edge(const BasicType x, const BasicType y) {
    if (is_cleared) {
        recover_data();
    }
    if (x >= min_x && x <= max_x && y >= min_y && y <= max_y) {
        // Note: check whether this point is on the invalid area using mask
        int curr_col_idx = std::round((x - min_x) / pyramids.x_grid_len_vec[0]); // col index
        int curr_row_idx = std::round((max_y - y) / pyramids.y_grid_len_vec[0]); // row index
        bool region_1 = false, region_2 = false, region_3 = false, region_4 = false;
        bool adjacent_region_1 = false, adjacent_region_2 = false, adjacent_region_3 = false, adjacent_region_4 = false;
        int search_region = std::max(
            static_cast<int>(std::round(direct_method_config_ptr->mask_gap_squared / pyramids.x_grid_len_vec[0])), 5);
        // shared region
        for (int row_idx = std::max(0, curr_row_idx - search_region); row_idx < curr_row_idx; ++row_idx) {
            if (BA_I_mask(row_idx, curr_col_idx) != T_INVALID) {
                adjacent_region_1 = true;
                break;
            }
        }
        for (int col_idx = std::min(curr_col_idx + 1, x_size - 1);
             col_idx < std::min(curr_col_idx + search_region + 1, x_size); ++col_idx) {
            if (BA_I_mask(curr_row_idx, col_idx) != T_INVALID) {
                adjacent_region_2 = true;
                break;
            }
        }
        for (int row_idx = std::min(curr_row_idx + 1, y_size - 1);
             row_idx < std::min(curr_row_idx + search_region + 1, y_size); ++row_idx) {
            if (BA_I_mask(row_idx, curr_col_idx) != T_INVALID) {
                adjacent_region_3 = true;
                break;
            }
        }
        for (int col_idx = std::max(0, curr_col_idx - search_region); col_idx < curr_col_idx; ++col_idx) {
            if (BA_I_mask(curr_row_idx, col_idx) != T_INVALID) {
                adjacent_region_4 = true;
                break;
            }
        }
        if (adjacent_region_1 && adjacent_region_2 && adjacent_region_3 && adjacent_region_4) {
            return std::make_pair(false, false);
        }
        // region 1
        for (int row_idx = std::max(0, curr_row_idx - search_region); row_idx < curr_row_idx; ++row_idx) {
            for (int col_idx = std::max(0, curr_col_idx - search_region); col_idx < curr_col_idx; ++col_idx) {
                if (BA_I_mask(row_idx, col_idx) != T_INVALID) {
                    region_1 = true;
                    break;
                }
            }
            if (region_1) {
                break;
            }
        }
        if (region_1 && adjacent_region_2 && adjacent_region_3) {
            return std::make_pair(false, false);
        }
        // region 2
        for (int row_idx = std::max(0, curr_row_idx - search_region); row_idx < curr_row_idx; ++row_idx) {
            for (int col_idx = std::min(curr_col_idx + 1, x_size - 1);
                 col_idx < std::min(curr_col_idx + search_region + 1, x_size); ++col_idx) {
                if (BA_I_mask(row_idx, col_idx) != T_INVALID) {
                    region_2 = true;
                    break;
                }
            }
            if (region_2) {
                break;
            }
        }
        if (region_2 && adjacent_region_3 && adjacent_region_4) {
            return std::make_pair(false, false);
        }
        // region 3
        for (int row_idx = std::min(curr_row_idx + 1, y_size - 1);
             row_idx < std::min(curr_row_idx + search_region + 1, y_size); ++row_idx) {
            for (int col_idx = std::max(0, curr_col_idx - search_region); col_idx < curr_col_idx; ++col_idx) {
                if (BA_I_mask(row_idx, col_idx) != T_INVALID) {
                    region_3 = true;
                    break;
                }
            }
            if (region_3) {
                break;
            }
        }
        if (region_3 && adjacent_region_1 && adjacent_region_2) {
            return std::make_pair(false, false);
        }
        // region 4
        for (int row_idx = std::min(curr_row_idx + 1, y_size - 1);
             row_idx < std::min(curr_row_idx + search_region + 1, y_size); ++row_idx) {
            for (int col_idx = std::min(curr_col_idx + 1, x_size - 1);
                 col_idx < std::min(curr_col_idx + search_region + 1, x_size); ++col_idx) {
                if (BA_I_mask(row_idx, col_idx) != T_INVALID) {
                    region_4 = true;
                    break;
                }
            }
            if (region_4) {
                break;
            }
        }
        if (region_4 && adjacent_region_1 && adjacent_region_2) {
            return std::make_pair(false, false);
        }
        if (region_1 && region_2 && region_3 && region_4) {
            return std::make_pair(false, false);
        } else {
            return std::make_pair(true, true);
        }
    }
    return std::make_pair(true, true);
}

template <typename BasicType> std::vector<BasicType> PatchProcession<BasicType>::get_x_grid_len_vec() {
    return pyramids.x_grid_len_vec;
}

template <typename BasicType> std::vector<BasicType> PatchProcession<BasicType>::get_y_grid_len_vec() {
    return pyramids.y_grid_len_vec;
}

template <typename BasicType>
void PatchProcession<BasicType>::update_BA_mask_idx(const Eigen::MatrixX<BasicType> &update_points_obj,
                                                    const double radius, const int minimum_num) {
    if (BA_I_mask.rows() <= 0 || BA_I_mask.cols() <= 0) {
        std::cerr << "[CURL_SLAM] update_BA_mask_idx invalid BA_I_mask size: rows=" << BA_I_mask.rows()
                  << " cols=" << BA_I_mask.cols() << std::endl;
        std::abort();
    }
    if (pyramids.x_grid_len_vec.empty() || pyramids.y_grid_len_vec.empty()) {
        std::cerr << "[CURL_SLAM] update_BA_mask_idx invalid pyramid grid lengths" << std::endl;
        std::abort();
    }
    if (pyramids.x_grid_len_vec[0] <= 0 || pyramids.y_grid_len_vec[0] <= 0) {
        std::cerr << "[CURL_SLAM] update_BA_mask_idx invalid grid step: x=" << pyramids.x_grid_len_vec[0]
                  << " y=" << pyramids.y_grid_len_vec[0] << std::endl;
        std::abort();
    }
    // get new idx masks
    std::vector<std::pair<int, int>> new_idx_vec;
    for (int idx = 0; idx < update_points_obj.cols(); ++idx) {
        int u = std::round((update_points_obj(0, idx) - min_x) / pyramids.x_grid_len_vec[0]);
        int v = std::round((max_y - update_points_obj(1, idx)) / pyramids.y_grid_len_vec[0]);
        if (u >= 0 && u < x_size && v >= 0 && v < y_size) {
            if (BA_I_mask(v, u) == T_INVALID) {
                BA_I_mask(v, u) = update_points_obj(2, idx);
                new_idx_vec.emplace_back(v, u);
            }
        }
    }
    // check whether newly inserted points are too sparse to be kept according to its neighbous
    int region_patch = std::ceil(radius / pyramids.x_grid_len_vec[0]);
    const int max_row = static_cast<int>(BA_I_mask.rows()) - 1;
    const int max_col = static_cast<int>(BA_I_mask.cols()) - 1;
    for (const auto &new_idx : new_idx_vec) {
        int counter = 0;
        for (int v = std::max(0, new_idx.first - region_patch); v <= std::min(max_row, new_idx.first + region_patch);
             ++v) {
            for (int u = std::max(0, new_idx.second - region_patch);
                 u <= std::min(max_col, new_idx.second + region_patch); ++u) {
                if (BA_I_mask(v, u) != T_INVALID) {
                    ++counter;
                }
            }
        }
        if (counter < minimum_num) {
            BA_I_mask(new_idx.first, new_idx.second) = T_INVALID;
        } else {
            BA_mask_idx.push_back(new_idx.second * y_size + new_idx.first);
        }
    }
}

template <typename BasicType>
void PatchProcession<BasicType>::update_BA_mask_idx(const Eigen::Vector3<BasicType> &pts_obj) {
    if (BA_I_mask.rows() <= 0 || BA_I_mask.cols() <= 0) {
        std::cerr << "[CURL_SLAM] update_BA_mask_idx(pts) invalid BA_I_mask size: rows=" << BA_I_mask.rows()
                  << " cols=" << BA_I_mask.cols() << std::endl;
        std::abort();
    }
    if (pyramids.x_grid_len_vec.empty() || pyramids.y_grid_len_vec.empty()) {
        std::cerr << "[CURL_SLAM] update_BA_mask_idx(pts) invalid pyramid grid lengths" << std::endl;
        std::abort();
    }
    if (pyramids.x_grid_len_vec[0] <= 0 || pyramids.y_grid_len_vec[0] <= 0) {
        std::cerr << "[CURL_SLAM] update_BA_mask_idx(pts) invalid grid step: x=" << pyramids.x_grid_len_vec[0]
                  << " y=" << pyramids.y_grid_len_vec[0] << std::endl;
        std::abort();
    }
    int u = std::round((pts_obj(0) - min_x) / pyramids.x_grid_len_vec[0]);
    int v = std::round((max_y - pts_obj(1)) / pyramids.y_grid_len_vec[0]);
    if (u >= 0 && u < x_size && v >= 0 && v < y_size) {
        if (BA_I_mask(v, u) == T_INVALID) {
            BA_I_mask(v, u) = pts_obj(2);
            BA_mask_idx.push_back(u * y_size + v);
        }
    }
}

template <typename BasicType>
void PatchProcession<BasicType>::update_BA_mask_idx(
    const std::unordered_map<std::vector<int>, Eigen::Vector3<BasicType>, VoxelHashFuncPrime> &hash_map_pts,
    Eigen::Isometry3d _T_obj_lidar) {
    if (BA_I_mask.rows() <= 0 || BA_I_mask.cols() <= 0) {
        std::cerr << "[CURL_SLAM] update_BA_mask_idx(hash) invalid BA_I_mask size: rows=" << BA_I_mask.rows()
                  << " cols=" << BA_I_mask.cols() << std::endl;
        std::abort();
    }
    if (pyramids.x_grid_len_vec.empty() || pyramids.y_grid_len_vec.empty()) {
        std::cerr << "[CURL_SLAM] update_BA_mask_idx(hash) invalid pyramid grid lengths" << std::endl;
        std::abort();
    }
    if (pyramids.x_grid_len_vec[0] <= 0 || pyramids.y_grid_len_vec[0] <= 0) {
        std::cerr << "[CURL_SLAM] update_BA_mask_idx(hash) invalid grid step: x=" << pyramids.x_grid_len_vec[0]
                  << " y=" << pyramids.y_grid_len_vec[0] << std::endl;
        std::abort();
    }
    Eigen::MatrixX<BasicType> points_j(3, hash_map_pts.size());
    int idx = 0;
    for (const auto &pair : hash_map_pts) {
        points_j.col(idx) = pair.second;
        ++idx;
    }
    Eigen::MatrixX<BasicType> points_obj =
        (_T_obj_lidar.matrix()(Eigen::seq(0, 2), Eigen::seq(0, 2)).cast<BasicType>() * points_j).colwise() +
        _T_obj_lidar.matrix()(Eigen::seq(0, 2), 3).cast<BasicType>();
    if (xy_grid.rows() != x_size * y_size) {
        std::unique_lock<std::shared_mutex> ul_xy_grid(xy_grid_lock);
        curl::mesh_grid<BasicType>(min_x, max_x, x_size, max_y, min_y, y_size, xy_grid);
    }
    interp_func_pair F = linear_interpolation_2_func<BasicType>(Eigen::VectorX<BasicType>(points_obj.row(0)),
                                                                Eigen::VectorX<BasicType>(points_obj.row(1)),
                                                                Eigen::VectorX<BasicType>(points_obj.row(2)));
    BasicType z_boundary[2] = {points_obj.row(2).minCoeff(), points_obj.row(2).maxCoeff()};
    if (xy_grid.rows() != x_size * y_size) {
        std::unique_lock<std::shared_mutex> ul_xy_grid(xy_grid_lock);
        curl::mesh_grid<BasicType>(min_x, max_x, x_size, max_y, min_y, y_size, xy_grid);
    }
    std::shared_lock<std::shared_mutex> sl_xy_grid(xy_grid_lock);
    std::pair<std::vector<BasicType>, std::vector<int>> interp_result = linear_barycentric_interpolation_2<BasicType>(
        xy_grid, points_obj.row(2), F, z_boundary, direct_method_config_ptr->mask_gap_squared);
    Eigen::Map<Eigen::VectorX<BasicType>> BA_I_mask_all_vec(BA_I_mask_all.data(), y_size * x_size);
    Eigen::Map<Eigen::VectorX<BasicType>> BA_I_mask_times_vec(BA_I_mask_times.data(), y_size * x_size);
    Eigen::Map<Eigen::VectorX<BasicType>> BA_I_mask_vec(BA_I_mask.data(), y_size * x_size);
    if (direct_method_config_ptr->is_average) {
        for (const auto &i : interp_result.second) {
            assert(interp_result.first[i] != T_INVALID);
            if (BA_I_mask_all_vec(i) == T_INVALID) {
                BA_I_mask_all_vec(i) = interp_result.first[i];
                BA_mask_idx.push_back(i);
            } else {
                BA_I_mask_all_vec(i) += interp_result.first[i];
            }
            BA_I_mask_times_vec(i) += 1;
        }
        BA_I_mask_vec(BA_mask_idx) = BA_I_mask_all_vec(BA_mask_idx).array() / BA_I_mask_times_vec(BA_mask_idx).array();
    } else {
        for (const auto &i : interp_result.second) {
            assert(interp_result.first[i] != T_INVALID);
            if (BA_I_mask_all_vec(i) == T_INVALID) {
                BA_mask_idx.push_back(i);
            }
            BA_I_mask_vec(i) = interp_result.first[i];
        }
    }
}

template <typename BasicType>
void PatchProcession<BasicType>::update_BA_mask_idx_trackingWay(const Eigen::MatrixX<BasicType> &update_points_obj) {
    if (BA_I_mask.rows() <= 0 || BA_I_mask.cols() <= 0) {
        std::cerr << "[CURL_SLAM] update_BA_mask_idx_trackingWay invalid BA_I_mask size: rows=" << BA_I_mask.rows()
                  << " cols=" << BA_I_mask.cols() << std::endl;
        std::abort();
    }
    if (pyramids.x_grid_len_vec.empty() || pyramids.y_grid_len_vec.empty()) {
        std::cerr << "[CURL_SLAM] update_BA_mask_idx_trackingWay invalid pyramid grid lengths" << std::endl;
        std::abort();
    }
    if (pyramids.x_grid_len_vec[0] <= 0 || pyramids.y_grid_len_vec[0] <= 0) {
        std::cerr << "[CURL_SLAM] update_BA_mask_idx_trackingWay invalid grid step: x="
                  << pyramids.x_grid_len_vec[0] << " y=" << pyramids.y_grid_len_vec[0] << std::endl;
        std::abort();
    }
    interp_func_pair F = linear_interpolation_2_func<BasicType>(Eigen::VectorX<BasicType>(update_points_obj.row(0)),
                                                                Eigen::VectorX<BasicType>(update_points_obj.row(1)),
                                                                Eigen::VectorX<BasicType>(update_points_obj.row(2)));
    BasicType z_boundary[2] = {update_points_obj.row(2).minCoeff(), update_points_obj.row(2).maxCoeff()};
    if (xy_grid.rows() != x_size * y_size) {
        std::unique_lock<std::shared_mutex> ul_xy_grid(xy_grid_lock);
        curl::mesh_grid<BasicType>(min_x, max_x, x_size, max_y, min_y, y_size, xy_grid);
    }
    std::shared_lock<std::shared_mutex> sl_xy_grid(xy_grid_lock);
    std::pair<std::vector<BasicType>, std::vector<int>> interp_result = linear_barycentric_interpolation_2<BasicType>(
        xy_grid, update_points_obj.row(2), F, z_boundary, direct_method_config_ptr->mask_gap_squared);
    Eigen::Map<Eigen::VectorX<BasicType>> BA_I_mask_all_vec(BA_I_mask_all.data(), y_size * x_size);
    Eigen::Map<Eigen::VectorX<BasicType>> BA_I_mask_times_vec(BA_I_mask_times.data(), y_size * x_size);
    Eigen::Map<Eigen::VectorX<BasicType>> BA_I_mask_vec(BA_I_mask.data(), y_size * x_size);
    for (const auto &i : interp_result.second) {
        assert(interp_result.first[i] != T_INVALID);
        if (BA_I_mask_all_vec(i) == T_INVALID) {
            BA_I_mask_all_vec(i) = interp_result.first[i];
            BA_mask_idx.push_back(i);
        } else {
            BA_I_mask_all_vec(i) += interp_result.first[i];
        }
        BA_I_mask_times_vec(i) += 1;
    }
    BA_I_mask_vec(BA_mask_idx) = BA_I_mask_all_vec(BA_mask_idx).array() / BA_I_mask_times_vec(BA_mask_idx).array();
}

template <typename BasicType> Eigen::MatrixX<BasicType> PatchProcession<BasicType>::get_BA_prior_points_obj() {
    if (xy_grid.rows() != x_size * y_size) {
        std::unique_lock<std::shared_mutex> ul_xy_grid(xy_grid_lock);
        curl::mesh_grid<BasicType>(min_x, max_x, x_size, max_y, min_y, y_size, xy_grid);
    }
    std::shared_lock<std::shared_mutex> sl_xy_grid(xy_grid_lock);
    Eigen::MatrixX<BasicType> prior_points_obj(3, BA_mask_idx.size());
    prior_points_obj(Eigen::seq(0, 1), Eigen::all) = xy_grid(BA_mask_idx, Eigen::all).transpose();
    Eigen::Map<Eigen::VectorX<BasicType>> BA_I_mask_vec(BA_I_mask.data(), y_size * x_size);
    prior_points_obj.row(2) = BA_I_mask_vec(BA_mask_idx);
    return prior_points_obj;
}

// template <typename BasicType> Eigen::MatrixX<BasicType> PatchProcession<BasicType>::get_recons_v_local() {
//     Eigen::MatrixX<BasicType> xy_grid_dense;
//     std::unique_lock<std::shared_mutex> ul_xy_grid(xy_grid_lock);
//     curl::mesh_grid<BasicType>(min_x, max_x, x_size, max_y, min_y, y_size, xy_grid_dense);
//     Eigen::VectorXd h = curl::invLeastSquaresSHT_table<BasicType>(
//         sph_coeff.cast<BasicType>(), SH_table_config_ptr->I_dirs(mask_idx, Eigen::all), *SH_table_config_ptr,
//         SH_degree);
//     Eigen::MatrixX<BasicType> recons_v(mask_idx.size(), 3);
//     recons_v(Eigen::all, Eigen::seq(0, 1)) = xy_grid_dense(mask_idx, Eigen::all);
//     recons_v(Eigen::all, 2) = h;
//     return recons_v;
// }
template <typename BasicType>
void PatchProcession<BasicType>::set_config_ptr(
    const std::shared_ptr<SH_TABLE_CONFIG<BasicType>> &_SH_table_config_ptr,
    const std::shared_ptr<DIRECT_METHOD_CONFIG> &_direct_method_config_ptr) {
    SH_table_config_ptr = _SH_table_config_ptr;
    direct_method_config_ptr = _direct_method_config_ptr;
}

template <typename BasicType>
Eigen::MatrixX<BasicType>
PatchProcession<BasicType>::calculate_recons_v_local_continuous_recons(const int patch_img_rso,
                                                                       const int SH_degree_CR) {

    Eigen::MatrixX<BasicType> I_CR = Eigen::MatrixX<BasicType>::Ones(patch_img_rso, patch_img_rso) * T_INVALID;

    Eigen::MatrixX<BasicType> I_dirs_CR;
    curl::mesh_grid<BasicType>(SH_table_config_ptr->get_azi_low(), SH_table_config_ptr->get_azi_high(), patch_img_rso,
                               SH_table_config_ptr->get_elev_high(), SH_table_config_ptr->get_elev_low(), patch_img_rso,
                               I_dirs_CR);

    Eigen::MatrixX<BasicType> xy_grid_CR;
    curl::mesh_grid<BasicType>(min_x, max_x, patch_img_rso, max_y, min_y, patch_img_rso, xy_grid_CR);
    if (!is_build_mask_F) {
        // NOTE: Get mask_idx_CR through meshing
        if (xy_grid.rows() != x_size * y_size) {
            std::unique_lock<std::shared_mutex> ul_xy_grid(xy_grid_lock);
            curl::mesh_grid<BasicType>(min_x, max_x, x_size, max_y, min_y, y_size, xy_grid);
        }
        if (I.rows() != y_size || I.cols() != x_size) {
            I = Eigen::MatrixX<BasicType>::Ones(y_size, x_size) * T_INVALID;
        }
        // BUG: This place didn't distringuish non-ground and ground degrees
        if (SH_table_config_ptr->is_speed_up) {
            Eigen::Map<Eigen::VectorX<BasicType>>(I.data(), y_size * x_size)(mask_idx) =
                curl::invLeastSquaresSHT_table<BasicType>(sph_coeff.cast<BasicType>(),
                                                          SH_table_config_ptr->I_dirs(mask_idx, Eigen::all),
                                                          *SH_table_config_ptr, warping_SH_degree);
        } else {
            Eigen::Map<Eigen::VectorX<BasicType>>(I.data(), y_size * x_size)(mask_idx) =
                curl::invLeastSquaresSHT<BasicType>(
                    sph_coeff.cast<BasicType>(), SH_table_config_ptr->I_dirs(mask_idx, Eigen::all), warping_SH_degree);
        }
        mask_F =
            linear_interpolation_2_func<BasicType>(xy_grid(mask_idx, 0), xy_grid(mask_idx, 1),
                                                   Eigen::Map<Eigen::VectorX<BasicType>>(I.data(), I.size())(mask_idx));
        I = Eigen::MatrixX<BasicType>();
        xy_grid = Eigen::MatrixX<BasicType>();
    }
    std::vector<int> mask_idx_CR =
        mask_identification<BasicType>(xy_grid_CR, mask_F, direct_method_config_ptr->mask_gap_squared);

    if (SH_table_config_ptr->is_speed_up) {
        Eigen::Map<Eigen::VectorX<BasicType>>(I_CR.data(), patch_img_rso * patch_img_rso) =
            curl::invLeastSquaresSHT_table<BasicType>(sph_coeff.cast<BasicType>(), I_dirs_CR, *SH_table_config_ptr,
                                                      SH_degree >= SH_degree_CR ? SH_degree_CR : SH_degree);
    } else {
        Eigen::Map<Eigen::VectorX<BasicType>>(I_CR.data(), patch_img_rso * patch_img_rso)(mask_idx_CR) =
            curl::invLeastSquaresSHT<BasicType>(sph_coeff.cast<BasicType>(), I_dirs_CR, SH_degree);
    }

    Eigen::MatrixX<BasicType> recons_v_local_continuous_recons(patch_img_rso * patch_img_rso, 3);

    recons_v_local_continuous_recons(Eigen::all, Eigen::seq(0, 1)) = xy_grid_CR;
    recons_v_local_continuous_recons.col(2) = Eigen::Map<Eigen::VectorX<BasicType>>(I_CR.data(), I_CR.size());
    if (patch_img_rso == x_size) {
        return recons_v_local_continuous_recons(mask_idx, Eigen::all).transpose();
    } else {
        return recons_v_local_continuous_recons.transpose();
    }
}

template <typename BasicType>
Eigen::MatrixX<BasicType> PatchProcession<BasicType>::get_recons_v_local_continuous_recons() {
    return recons_v_local;
}

template <typename BasicType> int PatchProcession<BasicType>::get_height_image_size() { return (y_size * x_size); }

template <typename BasicType> Eigen::VectorXi PatchProcession<BasicType>::get_binary_mask() {
    Eigen::VectorXi binary_mask(y_size * x_size);
    binary_mask.setZero();
    binary_mask(mask_idx).setOnes();
    return binary_mask;
}

template class PatchProcession<BT>; // this is very important
                                    // namespace boost {
                                    // namespace serialization {

// template <class Archive> void serialize(Archive &ar, PatchProcession<BT> &t, const unsigned int version) {
//     t.serialize(ar, version);
// }

// } // namespace serialization
// } // namespace boost

// Export the instantiated template class
// BOOST_CLASS_EXPORT(PatchProcession<BT>)
