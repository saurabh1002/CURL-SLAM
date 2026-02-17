//
// Created by zkc on 06/06/23.
//
#include "curl_slam/CurlLocalBA.h"

#include <utility>

template <typename BasicType>
void CurlLocalBA<BasicType>::load_residuals_point_clouds(
    ceres::Problem &problem, ceres::Manifold *_SE3_manifold, ceres::LossFunctionWrapper *_loss_function,
    ceres::ParameterBlockOrdering *_ordering, const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe_ptr,
    const std::vector<std::tuple<Eigen::MatrixX<BasicType>, std::shared_ptr<PatchInfo<BasicType>>, double>>
        &pose_succeed_associations,
    std::shared_ptr<CURL_TRACKING_CONFIG> _curl_tracking_config_ptr,
    std::shared_ptr<CURL_VOXEL_MAPPING_CONFIG> _curl_voxel_mapping_config_ptr,
    std::shared_ptr<SH_TABLE_CONFIG<BasicType>> _SH_table_config_ptr,
    std::map<std::shared_ptr<KeyframeInfo<BasicType>>,
             std::vector<std::pair<std::shared_ptr<KeyframeInfo<BasicType>>, int>>,
             key_frame_info_comparator<BasicType>> &associated_keyframes,
    std::unordered_set<std::shared_ptr<PatchInfo<BasicType>>> &all_associated_history_patches,
    std::set<std::shared_ptr<KeyframeInfo<BasicType>>, key_frame_info_comparator<BasicType>> &history_keyframes_set) {
    // add into BA residuals
    for (const auto &succeed_asso : pose_succeed_associations) {
        auto patch_info_ptr = std::get<1>(succeed_asso);
        if (!patch_info_ptr) {
            continue;
        }
        std::shared_ptr<KeyframeInfo<BasicType>> history_keyframe_ptr;
        std::shared_ptr<PatchProcession<BasicType>> patch_procession_ptr;
        Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_o_i = Eigen::Matrix4d::Identity();
        bool is_ground = false;
        PatchId patch_id;
        {
            std::lock_guard<std::mutex> patch_lock(patch_info_ptr->patch_update_lock);
            history_keyframe_ptr = patch_info_ptr->keyframe_ptr;
            patch_procession_ptr = patch_info_ptr->patch_procession_ptr;
            T_o_i = patch_info_ptr->T_obj_lidar.matrix();
            is_ground = patch_info_ptr->is_ground;
            patch_id = patch_info_ptr->key;
        }
        if (!history_keyframe_ptr || !history_keyframe_ptr->is_backend_keyframe || !patch_procession_ptr) {
            continue;
        }
        // Double-check pattern: set flag first, then check pointer
        keyframe_ptr->is_being_used_by_ba = true;
        if (!keyframe_ptr->seg_cloud_ptr || !keyframe_ptr->ground_cloud_ptr) {
            keyframe_ptr->is_being_used_by_ba = false;
            continue;
        }
        double *history_loop_pose = history_keyframe_ptr->set_and_get_loopClosure_T_w_lidar_data_from_graph_pose();
        double *current_loop_pose = keyframe_ptr->set_and_get_loopClosure_T_w_lidar_data_from_graph_pose();

        all_associated_history_patches.insert(patch_info_ptr);
        // for historical keyframe debug
        history_keyframes_set.insert(history_keyframe_ptr);
        history_keyframe_ptr->insert_BA_asso_local_patches(patch_id);
        double *ba_sph_coeff = patch_procession_ptr->set_and_get_BA_sph_coeff_data();
        // end
        // TODO: Save all these variable for offline test
        /*
         * observed points + corresponding current keyframe index
         * corresponding historical keyframe + corresponding patch information (coefficients, degree, ground/non-ground)
         */
        for (int k = 0; k < std::get<0>(succeed_asso).cols(); ++k) {
            Eigen::Vector3d p_j = (std::get<0>(succeed_asso).col(k)).template cast<double>();
            ceres::CostFunction *couple_fix_sph_cost_func;
            if (!is_ground) {
                couple_fix_sph_cost_func = CoupleSphModFactor<BasicType, SEG_DEGREE_SIZE>::Create(
                    T_o_i, p_j, patch_procession_ptr,
                    _curl_voxel_mapping_config_ptr->max_residual, _curl_voxel_mapping_config_ptr->opt_status,
                    _SH_table_config_ptr);
                problem.AddResidualBlock(
                    couple_fix_sph_cost_func, _loss_function, history_loop_pose, current_loop_pose, ba_sph_coeff);
            } else {
                couple_fix_sph_cost_func = CoupleSphModFactor<BasicType, GROUND_DEGREE_SIZE>::Create(
                    T_o_i, p_j, patch_procession_ptr,
                    _curl_voxel_mapping_config_ptr->max_residual, _curl_voxel_mapping_config_ptr->opt_status,
                    _SH_table_config_ptr);
                problem.AddResidualBlock(
                    couple_fix_sph_cost_func, _loss_function, history_loop_pose, current_loop_pose, ba_sph_coeff);
            }
        }
        //        associated_keyframes.emplace(keyframe_ptr, std::get<1>(succeed_asso)->keyframe_ptr);
        // NOTE: associated_keyframes is used to find the keyframe which associate with most patches for loop closure
        // constraints
        if (associated_keyframes.find(keyframe_ptr) != associated_keyframes.end()) {
            bool is_found = false;
            for (auto &keyframe_pair : associated_keyframes[keyframe_ptr]) {
                if (keyframe_pair.first->frame_num == history_keyframe_ptr->frame_num) {
                    keyframe_pair.second++;
                    is_found = true;
                    break;
                }
            }
            if (!is_found) {
                associated_keyframes[keyframe_ptr].emplace_back(history_keyframe_ptr, 1);
            }
        } else {
            associated_keyframes[keyframe_ptr].emplace_back(history_keyframe_ptr, 1);
        }
        // get associated current keyframes and history keyframes (for adding loop closure contrains to pose graph)

        // update this patch's checkUpdate_last_update_keyframe_num so this patch can be further updated

        if (!pose_succeed_associations.empty()) {
            {
                std::lock_guard<std::mutex> patch_lock(patch_info_ptr->patch_update_lock);
                patch_info_ptr->checkUpdate_last_update_keyframe_num = keyframe_ptr->frame_num;
            }
            problem.SetManifold(history_loop_pose, _SE3_manifold);
            problem.SetManifold(current_loop_pose, _SE3_manifold);
            //            problem.SetParameterBlockConstant(
            //                std::get<1>(succeed_asso)->keyframe_ptr->set_and_get_loopClosure_T_w_lidar_data());
            //            problem.SetParameterBlockConstant(std::get<1>(succeed_asso)->patch_procession_ptr->get_sph_coeff_data());
            // Set group spherical harmonoics coefficients
            _ordering->AddElementToGroup(ba_sph_coeff, 0);
            // Set group for pose estimation
            _ordering->AddElementToGroup(history_loop_pose, 1);
            _ordering->AddElementToGroup(current_loop_pose, 1);
        }

        //        if (std::get<1>(succeed_asso)->keyframe_ptr->frame_idx == oldest_frame_idx) {

        //        }
        // points regulizers
        const int recons_stride = _curl_voxel_mapping_config_ptr->BA_recons_downsample_stride;
        const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_lidar_obj = T_o_i.inverse();
        Eigen::MatrixX<BasicType> recons_matrix_history_lidar =
            (T_lidar_obj(Eigen::seq(0, 2), Eigen::seq(0, 2)).template cast<BasicType>() *
             patch_procession_ptr->get_recons_v_obj_downsample(recons_stride).transpose())
                .colwise() +
            T_lidar_obj(Eigen::seq(0, 2), 3).template cast<BasicType>();

        for (int k = 0; k < recons_matrix_history_lidar.cols(); ++k) {
            ceres::CostFunction *points_sph_cost_func;
            if (!is_ground) {
                points_sph_cost_func = CoupleSphModRegFactor<BasicType, SEG_DEGREE_SIZE>::Create(
                    T_o_i, recons_matrix_history_lidar.col(k).template cast<double>(),
                    _curl_voxel_mapping_config_ptr->half_diag_cut_threshold,
                    patch_procession_ptr, _SH_table_config_ptr,
                    _curl_voxel_mapping_config_ptr->points_regularizer_weight_square_root);
                problem.AddResidualBlock(points_sph_cost_func, nullptr,
                                         ba_sph_coeff);
            } else {
                points_sph_cost_func = CoupleSphModRegFactor<BasicType, GROUND_DEGREE_SIZE>::Create(
                    T_o_i, recons_matrix_history_lidar.col(k).template cast<double>(),
                    _curl_voxel_mapping_config_ptr->half_diag_cut_threshold,
                    patch_procession_ptr, _SH_table_config_ptr,
                    _curl_voxel_mapping_config_ptr->points_regularizer_weight_square_root);
                problem.AddResidualBlock(points_sph_cost_func, nullptr,
                                         ba_sph_coeff);
            }
            // Set group spherical harmonoics coefficients
            _ordering->AddElementToGroup(ba_sph_coeff, 0);
        }
    }
}

template <typename BasicType>
void CurlLocalBA<BasicType>::set_last_keyframe_constant(
    ceres::Problem &problem, const std::shared_ptr<KeyframeInfo<BasicType>> &oldest_keyframe_ptr) {
    // set the last keyframe constant
    if (oldest_keyframe_ptr != nullptr) {
        problem.SetParameterBlockConstant(
            oldest_keyframe_ptr->set_and_get_loopClosure_T_w_lidar_data_from_graph_pose());
    }
}

template <typename BasicType>
void CurlLocalBA<BasicType>::solve_problem(ceres::Problem &problem,
                                           std::shared_ptr<CURL_VOXEL_MAPPING_CONFIG> _curl_voxel_mapping_config_ptr,
                                           std::shared_ptr<ceres::ParameterBlockOrdering> ordering) {
    // solve the problem
    ceres::Solver::Options options;
    // options.max_num_iterations = curl_tracking_config_ptr->max_num_iterations;
    // options.max_num_iterations = 0;
    options.num_threads = _curl_voxel_mapping_config_ptr->BA_thread_num;
    options.max_num_iterations = _curl_voxel_mapping_config_ptr->max_num_iterations;
    options.function_tolerance = _curl_voxel_mapping_config_ptr->function_tolerance;
    options.minimizer_progress_to_stdout = true;
    options.update_state_every_iteration = true;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.use_nonmonotonic_steps = false;
    // options.num_threads = 16;
    // options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    // options.preconditioner_type = ceres::SCHUR_JACOBI;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    // options.linear_solver_type = ceres::DENSE_SCHUR;
    if (ordering != nullptr) {
        options.linear_solver_ordering = ordering;
    }
    // options.use_explicit_schur_complement = false;
    // options.use_spse_initialization = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
}

template class CurlLocalBA<BT>; // this is very important
