//
// Created by zkc on 06/06/23.
//

#ifndef SRC_CURLLOCALBA_H
#define SRC_CURLLOCALBA_H

#include "curl_slam/CurlVoxelMapping.h"
// #include "DirectMethod.h"
#include "curl_slam/FileReaderBase.h"
#include "curl_slam/Timer.h"
#include "curl_slam/optimization_types.h"
#include "curl_slam/spatial_hashing.h"
#include "curl_slam/types.h"
#include <Eigen/Dense>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <unordered_set>
#include <vector>

template <typename BasicType> class CurlLocalBA {
  private:
    // degree definition
    static constexpr int SEG_DEGREE_SIZE = 36;
    static constexpr int GROUND_DEGREE_SIZE = 9;

  public:
    static void load_residuals_point_clouds(
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
        std::set<std::shared_ptr<KeyframeInfo<BasicType>>, key_frame_info_comparator<BasicType>>
            &history_keyframes_set);

    static void set_last_keyframe_constant(ceres::Problem &problem,
                                           const std::shared_ptr<KeyframeInfo<BasicType>> &oldest_keyframe_ptr);

    static void solve_problem(ceres::Problem &problem,
                              std::shared_ptr<CURL_VOXEL_MAPPING_CONFIG> _curl_voxel_mapping_config_ptr,
                              std::shared_ptr<ceres::ParameterBlockOrdering> ordering = nullptr);
};

#endif // SRC_CURLLOCALBA_H
