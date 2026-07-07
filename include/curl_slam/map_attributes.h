#ifndef SRC_MAP_ATTRIBUTES_H
#define SRC_MAP_ATTRIBUTES_H

#include "curl_slam/PatchProcession.h"
#include "curl_slam/curl_tools_light.h"
#include "curl_slam/optimization_types.h"
#include <Eigen/Dense>
#include <algorithm>
#include <mutex>
#include <shared_mutex>
#include <numeric>
#include <fstream>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// declaration
template <typename BasicType> struct PatchInfo;
template <typename BasicType> struct KeyframeInfo;

// curl point cloud for kd-tree
namespace curl {
template <typename T> struct PointCloud {
    PointCloud() = default;
    std::vector<Eigen::Vector3<T>> pts;
    std::vector<std::weak_ptr<KeyframeInfo<T>>> keyframe_weakptr_vec;
    inline void push_back(const Eigen::Vector3<T> &pt) { pts.push_back(pt); }
    inline void push_back(const Eigen::Vector3<T> &pt, const std::shared_ptr<KeyframeInfo<T>> &keyframe_ptr) {
        pts.push_back(pt);
        keyframe_weakptr_vec.push_back(keyframe_ptr);
    }
    inline void emplace_back(const Eigen::Vector3<T> &pt) { pts.emplace_back(pt); }
    inline void emplace_back(const Eigen::Vector3<T> &pt, const std::shared_ptr<KeyframeInfo<T>> &keyframe_ptr) {
        pts.emplace_back(pt);
        keyframe_weakptr_vec.emplace_back(keyframe_ptr);
    }
    inline auto begin() const { return pts.begin(); }
    inline auto end() const { return pts.end(); }
    inline auto clear() { pts.clear(); }
    inline auto reserve(std::size_t size) { pts.reserve(size); }
    inline std::size_t size() const { return pts.size(); }
    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate
    // value, the
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const { return pts[idx](dim); }

    // Optional bounding-box computation: return false to default to a standard
    // bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned
    //   in "bb" so it can be avoided to redo it again. Look at bb.size() to
    //   find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX> bool kdtree_get_bbox(BBOX & /* bb */) const { return false; }
};
} // namespace curl

// data-association structure

// used to save intersected points in patch
struct MapDataAsso {
    std::vector<std::array<int, 2>> points_idx_vec; // the index is {i,j}
    std::vector<double> IoU_vec;
    //    std::unordered_map<std::array<int, 2>, double, Voxel2DHashFuncPrimeArray> idx_IoU_map;

    int best_index() const {
        if (IoU_vec.empty()) {
            return -1;
        }
        auto it = std::max_element(IoU_vec.begin(), IoU_vec.end());
        return static_cast<int>(std::distance(IoU_vec.begin(), it));
    }

    std::vector<int> sort_according_to_score() {
        std::vector<int> idx_vec(IoU_vec.size());
        std::iota(idx_vec.begin(), idx_vec.end(), 0); // fill with 0, 1, ..., scores.size() - 1
        std::sort(idx_vec.begin(), idx_vec.end(), [&](int a, int b) { return IoU_vec[a] > IoU_vec[b]; });
        return idx_vec;
    }
    void clear() {
        points_idx_vec.clear();
        IoU_vec.clear();
    }
};

// Used to save intersected patch of the current points
template <typename BasicType> struct ScanDataAsso {
    ScanDataAsso() = delete;
    ScanDataAsso(std::array<int, 2> _point_idx) : point_idx(_point_idx) {}
    std::array<int, 2> point_idx;
    std::vector<std::shared_ptr<PatchInfo<BasicType>>> patch_info_ptr_vec;
    std::vector<double> scores;
    // this is not calculate score, this is access the score from MapDataAsso
    void calculate_score() {
        if (scores.empty()) {
            scores.reserve(patch_info_ptr_vec.size());
            for (const auto &patch_info_ptr : patch_info_ptr_vec) {
                assert(patch_info_ptr->data_asso.points_idx_vec.size() > 0);
                for (int i = 0; i < patch_info_ptr->data_asso.points_idx_vec.size(); ++i) {
                    if (patch_info_ptr->data_asso.points_idx_vec[i] == point_idx) {
                        scores.push_back(patch_info_ptr->data_asso.IoU_vec[i]);
                        break;
                    }
                }
            }
        }
    }

    int best_index() {
        calculate_score();
        if (scores.empty()) {
            return -1;
        }
        auto it = std::max_element(scores.begin(), scores.end());
        return static_cast<int>(std::distance(scores.begin(), it));
    }

    std::vector<int> sort_according_to_score() {
        calculate_score();
        std::vector<int> idx_vec(patch_info_ptr_vec.size());
        std::iota(idx_vec.begin(), idx_vec.end(), 0);
        std::sort(idx_vec.begin(), idx_vec.end(), [&](int a, int b) { return scores[a] > scores[b]; });
        return idx_vec;
    }
    void clear() {
        for (auto &patch_info_ptr : patch_info_ptr_vec) {
            patch_info_ptr->data_asso.clear();
        }
        patch_info_ptr_vec.clear();
        scores.clear();
    }
};

// map data structure

// define the hash function rules for std::weak_ptr
template <typename BasicType> struct weak_ptr_hash {
    std::size_t operator()(const std::weak_ptr<PatchInfo<BasicType>> &wp) const {
        auto sp = wp.lock();
        if (!sp) {
            // Handle expired weak_ptr
            return 0;
        }
        return std::hash<std::shared_ptr<PatchInfo<BasicType>>>{}(sp);
    }
};

// define the equality std::weak_ptr
template <typename BasicType> struct weak_ptr_equal {
    bool operator()(const std::weak_ptr<PatchInfo<BasicType>> &wp1,
                    const std::weak_ptr<PatchInfo<BasicType>> &wp2) const {
        auto sp1 = wp1.lock();
        auto sp2 = wp2.lock();
        if (!sp1 || !sp2) {
            return false; // Handle expired weak_ptr
        }
        return sp1 == sp2;
    }
};

struct SCORE {
    double IoU = 0.0;
    double IoU_history = 0.0;
};

struct TrajectoryLabel {
    inline static std::shared_mutex label_mutex;
    TrajectoryLabel(const unsigned int _label_frame_num, const Eigen::MatrixXf &color_map)
        : label_frame_num(_label_frame_num) {
        neighbor_label_frame_num.insert(_label_frame_num); // add itself
        int idx = _label_frame_num % color_map.rows();
        color[0] = color_map(idx, 0);
        color[1] = color_map(idx, 1);
        color[2] = color_map(idx, 2);
    }
    static void insert_neighbor_label(std::shared_ptr<TrajectoryLabel> &label1,
                                      std::shared_ptr<TrajectoryLabel> &label2) {
        std::unique_lock<std::shared_mutex> lock(label_mutex);
        if (label1->label_frame_num != label2->label_frame_num) {
            label1->neighbor_label_frame_num.insert(label2->label_frame_num);
            label2->neighbor_label_frame_num.insert(label1->label_frame_num);
        }
    }
    static bool merge_neighbor_label(std::shared_ptr<TrajectoryLabel> &label1,
                                     std::shared_ptr<TrajectoryLabel> &label2) {
        std::unique_lock<std::shared_mutex> lock(label_mutex);
        for (const auto &label : label1->neighbor_label_frame_num) {
            label2->neighbor_label_frame_num.insert(label);
        }
        label1->neighbor_label_frame_num = label2->neighbor_label_frame_num;
        return true;
    }
    static bool is_connected(const std::shared_ptr<TrajectoryLabel> &label1,
                             const std::shared_ptr<TrajectoryLabel> &label2) {
        std::shared_lock<std::shared_mutex> lock(label_mutex);
        return label1->neighbor_label_frame_num.find(label2->label_frame_num) != label1->neighbor_label_frame_num.end();
    }

    unsigned int label_frame_num;
    std::set<unsigned int> neighbor_label_frame_num;
    std::array<float, 3> color;
};

// use std::weak_ptr rather than std::shared_ptr is trying to avoid the memory leakage
template <typename BasicType> struct KeyframeInfo {
    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_w_lidar_gt;
    KeyframeInfo() = default;
    KeyframeInfo(double _time) : time(_time) {
        is_BA_T_w_lidar_set = false;
        is_IoU_sorted = false;
        is_update_by_pose_graph = false;
        is_backend_keyframe = false;
        lower_bound.resize(3, std::numeric_limits<double>::max());
        upper_bound.resize(3, std::numeric_limits<double>::lowest());
    }
    KeyframeInfo(double _time, const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_w_lidar,
                 const Eigen::Isometry3d &_T_j_1_j, const Eigen::Isometry3d &T_w_lidar_relative,
                 const std::shared_ptr<TrajectoryLabel> _trajectory_label_ptr)
        : time(_time), T_w_lidar(_T_w_lidar), T_j_1_j(_T_j_1_j), trajectory_label_ptr(_trajectory_label_ptr) {
        graph_pose_w_lidar.set_pose(T_w_lidar_relative.matrix());
        graph_obs_j_1_j.set_pose(T_j_1_j.matrix());
        is_BA_T_w_lidar_set = false;
        is_IoU_sorted = false;
        is_update_by_pose_graph = false;
        is_backend_keyframe = false;
        lower_bound.resize(3, std::numeric_limits<double>::max());
        upper_bound.resize(3, std::numeric_limits<double>::lowest());
    }
    unsigned long size() const { return local_patches.size(); };

    template <typename Lookup>
    void clear_patches_content(Lookup &&lookup) {
        for (const auto &patch_id : local_patches) {
            if (auto patch_ptr = lookup(patch_id)) {
                patch_ptr->clear_content();
            }
        }
    }

    template <typename Lookup>
    void maintain_patches_content(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &T_w_j,
                                  const std::pair<std::vector<double>, std::vector<double>> &valid_box_region,
                                  Lookup &&lookup) {
        (void)T_w_j;
        for (const auto &patch_id : local_patches) {
            if (auto patch_ptr = lookup(patch_id)) {
                Eigen::Vector3d pose_w_patch = (patch_ptr->keyframe_ptr->get_T_w_lidar() *
                                                patch_ptr->T_obj_lidar.inverse().matrix())(Eigen::seq(0, 2), 3);
                if (pose_w_patch(0) < valid_box_region.first[0] || pose_w_patch(0) > valid_box_region.second[0] ||
                    pose_w_patch(1) < valid_box_region.first[1] || pose_w_patch(1) > valid_box_region.second[1] ||
                    pose_w_patch(2) < valid_box_region.first[2] || pose_w_patch(2) > valid_box_region.second[2]) {
                    patch_ptr->clear_content();
                } else {
                    patch_ptr->recover_content();
                }
            }
        }
    }

    void set_T_w_lidar(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_w_lidar) {
        std::unique_lock<std::shared_mutex> ul_T_w_lidar(T_w_lidar_lock);
        T_w_lidar = _T_w_lidar;
        graph_pose_w_lidar.set_pose(T_w_lidar);
    }

    void set_T_w_lidar_noLock(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_w_lidar) {
        T_w_lidar = _T_w_lidar;
        graph_pose_w_lidar.set_pose(T_w_lidar);
    }

    void update_T_w_lidar_with_graph_pose() {
        std::unique_lock<std::shared_mutex> ul_T_w_lidar(T_w_lidar_lock);
        T_w_lidar = graph_pose_w_lidar.get_T().matrix();
    }

    void update_T_w_lidar_with_graph_pose_noLock() { T_w_lidar = graph_pose_w_lidar.get_T().matrix(); }

    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> get_T_w_lidar() {
        std::shared_lock<std::shared_mutex> sl_T_w_lidar(T_w_lidar_lock);
        return T_w_lidar;
    }

    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> get_T_w_lidar_noLock() { return T_w_lidar; }

    double *get_T_w_lidar_data() {
        std::shared_lock<std::shared_mutex> sl_w_lidar(T_w_lidar_lock);
        return T_w_lidar.data();
    }

    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> get_loopClosure_T_w_lidar() { return loopClosure_T_w_lidar; }

    double *set_and_get_loopClosure_T_w_lidar_data() {
        if (loopClosure_T_w_lidar != get_T_w_lidar()) {
            loopClosure_T_w_lidar = get_T_w_lidar();
        }
        return loopClosure_T_w_lidar.data();
    }

    double *set_and_get_loopClosure_T_w_lidar_data_from_graph_pose() {
        std::unique_lock<std::shared_mutex> ul_w_lidar(T_w_lidar_lock);
        if (!is_loopClosure_T_w_lidar_set) {
            loopClosure_T_w_lidar = graph_pose_w_lidar.get_T().matrix();
            is_loopClosure_T_w_lidar_set = true;
        }
        return loopClosure_T_w_lidar.data();
    }

    void set_loopClosure_T_w_lidar_from_snapshot(
        const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_loopClosure_T_w_lidar) {
        std::unique_lock<std::shared_mutex> ul_w_lidar(T_w_lidar_lock);
        loopClosure_T_w_lidar = _loopClosure_T_w_lidar;
        is_loopClosure_T_w_lidar_set = true;
    }

    void clear_loopClosure_T_w_lidar_cache() {
        std::unique_lock<std::shared_mutex> ul_w_lidar(T_w_lidar_lock);
        is_loopClosure_T_w_lidar_set = false;
    }

    const Pose3d &get_graph_obs_j_1_j() { return graph_obs_j_1_j; }

    Eigen::Isometry3d get_graph_obs_j_1_j_T() { return graph_obs_j_1_j.get_T(); }

    const Pose3d &get_graph_pose_w_lidar() { return graph_pose_w_lidar; }

    Eigen::Isometry3d get_graph_pose_w_lidar_T() { return graph_pose_w_lidar.get_T(); }

    double *get_graph_pose_w_lidar_Pdata() { return graph_pose_w_lidar.p.data(); }

    double *get_graph_pose_w_lidar_Qdata() { return graph_pose_w_lidar.q.coeffs().data(); }

    void set_BA_T_w_lidar(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_BA_T_w_lidar) {
        std::unique_lock<std::shared_mutex> ul_BA_T_w_lidar(BA_T_w_lidar_lock);
        BA_T_w_lidar = _BA_T_w_lidar;
    }

    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> get_BA_T_w_lidar() {
        std::shared_lock<std::shared_mutex> sl_BA_T_w_lidar(BA_T_w_lidar_lock);
        return BA_T_w_lidar;
    }

    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> get_BA_T_w_lidar_noLock() { return BA_T_w_lidar; }

    double *get_BA_T_w_lidar_data() {
        std::shared_lock<std::shared_mutex> sl_BA_T_w_lidar(BA_T_w_lidar_lock);
        return BA_T_w_lidar.data();
    }

    void set_T_j_1_j(const Eigen::Isometry3d &_T_j_1_j) { T_j_1_j = _T_j_1_j; }

    Eigen::Isometry3d get_T_j_1_j() { return T_j_1_j; };

    void succeed_associations_insert(
        const std::vector<std::tuple<Eigen::MatrixX<BasicType>, std::shared_ptr<PatchInfo<BasicType>>, double>>
            &asso_vec) {
        succeed_associations.insert(succeed_associations.end(), asso_vec.begin(), asso_vec.end());
        is_IoU_sorted = false;
    }
    void succeed_associations_push_back(
        const std::tuple<Eigen::MatrixX<BasicType>, std::shared_ptr<PatchInfo<BasicType>>, double> &ele) {
        succeed_associations.push_back(ele);
        is_IoU_sorted = false;
    }
    // return reference avoid copy
    const std::vector<std::tuple<Eigen::MatrixX<BasicType>, std::shared_ptr<PatchInfo<BasicType>>, double>> &
    get_succeed_associations() const {
        return succeed_associations;
    }
    const std::vector<std::vector<std::tuple<Eigen::MatrixX<BasicType>, std::shared_ptr<PatchInfo<BasicType>>, SCORE>>>
        &get_seg_succeed_associations() const {
        return seg_succeed_associations;
    }
    const std::vector<std::vector<std::tuple<Eigen::MatrixX<BasicType>, std::shared_ptr<PatchInfo<BasicType>>, SCORE>>>
        &get_ground_succeed_associations() const {
        return ground_succeed_associations;
    }

    void merage_succeed_associations(const int region_width_elements, const double outlier_rejection_leaf_size,
                                     const int outlier_rejection_minimum_neighbours) {
        if (succeed_associations.empty()) {
            return;
        }
        calculate_local_bounds();
        std::vector<std::set<unsigned int>> seg_patch_sets_vec(region_width_elements * region_width_elements);
        std::vector<std::set<unsigned int>> ground_patch_sets_vec(region_width_elements * region_width_elements);
        double x_axis_length = upper_bound[0] - lower_bound[0];
        double y_axis_length = upper_bound[1] - lower_bound[1];
        double z_axis_length = upper_bound[2] - lower_bound[2];
        const double min_axis_length = std::min({x_axis_length, y_axis_length, z_axis_length});
        int u_idx, v_idx;
        if (min_axis_length == z_axis_length) {
            u_idx = 0;
            v_idx = 1;
        } else if (min_axis_length == x_axis_length) {
            u_idx = 1;
            v_idx = 2;
        } else {
            u_idx = 0;
            v_idx = 2;
        }
        const double u_rso = (upper_bound[u_idx] - lower_bound[u_idx]) / static_cast<double>(region_width_elements);
        const double v_rso = (upper_bound[v_idx] - lower_bound[v_idx]) / static_cast<double>(region_width_elements);
        for (const auto &tuple : succeed_associations) {
            const double u_grid_pos = std::get<1>(tuple)->T_obj_lidar.matrix()(u_idx, 3) - lower_bound[u_idx];
            const double v_grid_pos = std::get<1>(tuple)->T_obj_lidar.matrix()(v_idx, 3) - lower_bound[v_idx];
            const int u = std::floor(u_grid_pos / u_rso);
            const int v = std::floor(v_grid_pos / v_rso);
            if (!std::get<1>(tuple)->is_ground) {
                seg_patch_sets_vec[v * region_width_elements + u].insert(std::get<1>(tuple)->key);
            } else {
                ground_patch_sets_vec[v * region_width_elements + u].insert(std::get<1>(tuple)->key);
            }
        }
        // get index dictionary
        std::vector<std::unordered_map<unsigned int, unsigned int>> seg_index_map_vec(region_width_elements *
                                                                                      region_width_elements);
        merage_get_index_map_vec(seg_patch_sets_vec, seg_index_map_vec);
        std::vector<std::unordered_map<unsigned int, unsigned int>> ground_index_map_vec(region_width_elements *
                                                                                         region_width_elements);
        merage_get_index_map_vec(ground_patch_sets_vec, ground_index_map_vec);
        // give value to succeed_associations_tmp
        // unsigned int stores total observation points number for each patch
        // NOTE: store points in object frame for getting mean (second try)
        std::vector<std::vector<
            std::tuple<std::vector<Eigen::MatrixX<BasicType>>, std::shared_ptr<PatchInfo<BasicType>>, unsigned int>>>
            seg_succeed_associations_vec_tmp(region_width_elements * region_width_elements);
        std::vector<std::vector<
            std::tuple<std::vector<Eigen::MatrixX<BasicType>>, std::shared_ptr<PatchInfo<BasicType>>, unsigned int>>>
            ground_succeed_associations_vec_tmp(region_width_elements * region_width_elements);
        // initialize values for each region
        for (int i = 0; i < region_width_elements * region_width_elements; ++i) {
            seg_succeed_associations_vec_tmp[i] =
                std::vector<std::tuple<std::vector<Eigen::MatrixX<BasicType>>, std::shared_ptr<PatchInfo<BasicType>>,
                                       unsigned int>>(
                    seg_patch_sets_vec[i].size(),
                    std::make_tuple(std::vector<Eigen::MatrixX<BasicType>>(), nullptr, 0));
            ground_succeed_associations_vec_tmp[i] =
                std::vector<std::tuple<std::vector<Eigen::MatrixX<BasicType>>, std::shared_ptr<PatchInfo<BasicType>>,
                                       unsigned int>>(
                    ground_patch_sets_vec[i].size(),
                    std::make_tuple(std::vector<Eigen::MatrixX<BasicType>>(), nullptr, 0));
        }
        for (const auto tuple : succeed_associations) {
            const double u_grid_pos = std::get<1>(tuple)->T_obj_lidar.matrix()(u_idx, 3) - lower_bound[u_idx];
            const double v_grid_pos = std::get<1>(tuple)->T_obj_lidar.matrix()(v_idx, 3) - lower_bound[v_idx];
            const int u = std::floor(u_grid_pos / u_rso);
            const int v = std::floor(v_grid_pos / v_rso);
            if (!std::get<1>(tuple)->is_ground) {
                unsigned int idx = seg_index_map_vec[v * region_width_elements + u][std::get<1>(tuple)->key];
                // push back the patch points elements
                // TODO: sampling points here and sum to all grid and get mean in the end (second try)
                std::get<0>(seg_succeed_associations_vec_tmp[v * region_width_elements + u][idx])
                    .push_back(std::get<0>(tuple));
                // get shared pointer for patch_info_ptr
                std::get<1>(seg_succeed_associations_vec_tmp[v * region_width_elements + u][idx]) = std::get<1>(tuple);
                // accumulate total number of points for each patch
                std::get<2>(seg_succeed_associations_vec_tmp[v * region_width_elements + u][idx]) +=
                    std::get<0>(tuple).cols();
            } else {
                unsigned int idx = ground_index_map_vec[v * region_width_elements + u][std::get<1>(tuple)->key];
                // push back the patch points elements
                // TODO: sampling points here and sum to all grid and get mean in the end (second try)
                std::get<0>(ground_succeed_associations_vec_tmp[v * region_width_elements + u][idx])
                    .push_back(std::get<0>(tuple));
                // get shared pointer for patch_info_ptr
                std::get<1>(ground_succeed_associations_vec_tmp[v * region_width_elements + u][idx]) =
                    std::get<1>(tuple);
                // accumulate total number of points for each patch
                std::get<2>(ground_succeed_associations_vec_tmp[v * region_width_elements + u][idx]) +=
                    std::get<0>(tuple).cols();
            }
        }
        // merge point cloud and give value from succeed_associations_tmp to succeed_associations
        seg_succeed_associations.resize(region_width_elements * region_width_elements);
        merge_get_IoU(region_width_elements, seg_succeed_associations_vec_tmp, seg_succeed_associations,
                      outlier_rejection_leaf_size, outlier_rejection_minimum_neighbours);
        ground_succeed_associations.resize(region_width_elements * region_width_elements);
        merge_get_IoU(region_width_elements, ground_succeed_associations_vec_tmp, ground_succeed_associations,
                      outlier_rejection_leaf_size, outlier_rejection_minimum_neighbours);
        is_IoU_sorted = false;
    }

    void sort_succeed_associations_IoU_history(
        const std::shared_ptr<KeyframeInfo<BasicType>> &keyframePtr, const int valid_keyframe_idx,
        const int max_patches_region_seg, const int max_patches_region_ground, const double history_weight_factor,
        std::unordered_set<std::shared_ptr<PatchInfo<BasicType>>> &history_patch_ptr_seg,
        std::unordered_set<std::shared_ptr<PatchInfo<BasicType>>> &history_patch_ptr_ground) {
        sort_succ_associations_IoU_history(keyframePtr, valid_keyframe_idx, max_patches_region_seg,
                                           history_weight_factor, history_patch_ptr_seg, seg_succeed_associations);
        sort_succ_associations_IoU_history(keyframePtr, valid_keyframe_idx, max_patches_region_ground,
                                           history_weight_factor, history_patch_ptr_ground,
                                           ground_succeed_associations);
        is_IoU_sorted = false;
    }

    void sort_succeed_associations_IoU() {
        if (!is_IoU_sorted) {
            sort_succ_associations_IoU(seg_succeed_associations);
            sort_succ_associations_IoU(ground_succeed_associations);
        }
        is_IoU_sorted = true;
    }

    void update_patches(const double voxel_size, const double outlier_rejection_leaf_size,
                        const int outlier_rejection_minimum_neighbours) {
        for (auto &tuple : succeed_associations) {
            // only inactivate patches can be updated
            if ((!std::get<1>(tuple)->keyframe_ptr->is_active) && std::get<0>(tuple).cols() != 0) {
                std::get<1>(tuple)->original_points_lidar_vec_update_back(T_w_lidar, std::get<0>(tuple));
            }
        }
        for (auto &tuple : succeed_associations) {
            if ((!std::get<1>(tuple)->keyframe_ptr->is_active) && std::get<0>(tuple).cols() != 0) {
                std::get<1>(tuple)->merge_and_resample_original_points(voxel_size, outlier_rejection_leaf_size,
                                                                       outlier_rejection_minimum_neighbours);
            }
        }
    }

    void succeed_associations_clear() {
        succeed_associations.clear();
        seg_succeed_associations.clear();
        ground_succeed_associations.clear();
    }

    template <typename Lookup>
    pcl::PointCloud<PointT>::Ptr get_all_assco_patch_centroids(Lookup &&lookup) {
        pcl::PointCloud<PointT>::Ptr all_patch_centroids(new pcl::PointCloud<PointT>);
        for (const auto &tuple : succeed_associations) {
            if (std::get<1>(tuple)->is_generated_by_keyframe) {
                PointT pt;
                Eigen::Vector3d centroid_lidar = std::get<1>(tuple)->get_centroid_lidar();
                pt.x = centroid_lidar(0);
                pt.y = centroid_lidar(1);
                pt.z = centroid_lidar(2);
                all_patch_centroids->push_back(pt);
            }
        }
        for (const auto &patch_id : local_patches) {
            if (auto patch_ptr = lookup(patch_id)) {
                if (!patch_ptr->is_generated_by_keyframe) {
                    continue;
                }
                PointT pt;
                Eigen::Vector3d centroid_lidar = patch_ptr->get_centroid_lidar();
                pt.x = centroid_lidar(0);
                pt.y = centroid_lidar(1);
                pt.z = centroid_lidar(2);
                all_patch_centroids->push_back(pt);
            }
        }
        return all_patch_centroids;
    }

    int get_local_patches_size() {
        std::unique_lock<std::mutex> ul_local_patches(local_patches_lock);
        return local_patches.size();
    }

    int get_BA_asso_local_patches_size() {
        std::unique_lock<std::mutex> ul_BA_local_patches(BA_local_patches_lock);
        return BA_asso_local_patches.size();
    }

    void insert_BA_asso_local_patches(PatchId patch_id) {
        std::unique_lock<std::mutex> ul_BA_local_patches(BA_local_patches_lock);
        BA_asso_local_patches.insert(patch_id);
    }

    void clear_BA_asso_local_patches() {
        std::unique_lock<std::mutex> ul_BA_local_patches(BA_local_patches_lock);
        BA_asso_local_patches.clear();
    }

    double time;
    bool is_active; // true in the beginning, and after local BA, the last frame of the window need to be set as false
    std::unordered_set<PatchId>
        local_patches; // inilized in the insert
    std::unordered_set<PatchId>
        BA_asso_local_patches;
    std::mutex local_patches_lock, BA_local_patches_lock;
    std::unordered_set<PatchId>
        associated_patches; // inilized in the insert
    void associated_patches_insert(PatchId patch_id) {
        std::unique_lock<std::shared_mutex> ul(associated_patches_lock);
        associated_patches.insert(patch_id);
    }
    static std::shared_mutex BA_T_w_lidar_lock;
    static std::shared_mutex T_w_lidar_lock;
    std::shared_mutex associated_patches_lock;

    pcl::PointCloud<PointT>::Ptr seg_cloud_ptr;
    pcl::PointCloud<PointT>::Ptr ground_cloud_ptr;

    int frame_idx;
    int frame_num;
    bool is_BA_T_w_lidar_set;
    bool is_loopClosure_T_w_lidar_set = false;
    bool is_update_by_pose_graph;
    bool is_backend_keyframe;
    std::atomic<bool> is_being_used_by_ba{false};

    std::shared_ptr<TrajectoryLabel> trajectory_label_ptr;
    // template <class Archive> void serialize(Archive &ar, const unsigned int version) { ar &T_w_lidar; }

    PoseGraph3dErrorTerm *loop_closure_cost_function = nullptr;

  private:
    // friend class boost::serialization::access;
    void calculate_local_bounds() {
        for (const auto &succeed_asso : succeed_associations) {
            assert(frame_idx != std::get<1>(succeed_asso)->keyframe_ptr->frame_idx);
            for (int i = 0; i < 3; ++i) {
                auto value = std::get<1>(succeed_asso)->T_obj_lidar.matrix()(i, 3);
                if (value < lower_bound[i]) {
                    lower_bound[i] = value;
                }
                if (value > upper_bound[i]) {
                    upper_bound[i] = value;
                }
            }
        }
        // extend a little bit for boundry situation
        for (auto &value : upper_bound) {
            value += 0.001;
        }
        for (auto &value : lower_bound) {
            value -= 0.001;
        }
    }

    void merage_get_index_map_vec(const std::vector<std::set<unsigned int>> &patch_sets_vec,
                                  std::vector<std::unordered_map<unsigned int, unsigned int>> &index_map_vec) {
        for (int i = 0; i < patch_sets_vec.size(); ++i) {
            unsigned int index = 0;
            for (auto iter = patch_sets_vec[i].begin(); iter != patch_sets_vec[i].end(); ++iter, ++index) {
                index_map_vec[i][*iter] = index;
            }
        }
    }

    void merge_get_IoU(
        const int region_width_elements,
        const std::vector<std::vector<
            std::tuple<std::vector<Eigen::MatrixX<BasicType>>, std::shared_ptr<PatchInfo<BasicType>>, unsigned int>>>
            &succ_associations_tmp,
        std::vector<std::vector<std::tuple<Eigen::MatrixX<BasicType>, std::shared_ptr<PatchInfo<BasicType>>, SCORE>>>
            &succ_associations,
        const double outlier_rejection_leaf_size, const int outlier_rejection_minimum_neighbours) {
        for (int i = 0; i < region_width_elements * region_width_elements; ++i) {
            // for seg
            succ_associations[i].resize(succ_associations_tmp[i].size());
            for (int j = 0; j < succ_associations_tmp[i].size(); ++j) {
                // give shared_ptr of patch_info_ptr
                std::get<1>(succ_associations[i][j]) = std::get<1>(succ_associations_tmp[i][j]);
                Eigen::MatrixX<BasicType> matrix(3, std::get<2>(succ_associations_tmp[i][j]));
                // accumulate all matrices into one
                int start_idx = 0;
                for (const auto &sub_matrix : std::get<0>(succ_associations_tmp[i][j])) {
                    matrix(Eigen::all, Eigen::seq(start_idx, start_idx + sub_matrix.cols() - 1)) = sub_matrix;
                    start_idx += sub_matrix.cols();
                }
                std::get<0>(succ_associations[i][j]) = matrix;
            }
            for (auto &tuple : succ_associations[i]) {
                // transfer std::get<0>(tuple) into world coordinate to extract bound to calculate IoU
                // std::get<0>(tuple) = curl::reject_outliers<BasicType>(std::get<0>(tuple),
                // outlier_rejection_leaf_size,
                //                                                       outlier_rejection_minimum_neighbours);
                if (std::get<0>(tuple).cols() != 0) {
                    Eigen::MatrixXd points_w = (get_T_w_lidar()(Eigen::seq(0, 2), Eigen::seq(0, 2)) *
                                                std::get<0>(tuple).template cast<double>())
                                                   .colwise() +
                                               get_T_w_lidar()(Eigen::seq(0, 2), 3);
                    std::vector<double> lower_bound_w{points_w.row(0).minCoeff(), points_w.row(1).minCoeff(),
                                                      points_w.row(2).minCoeff()};
                    std::vector<double> upper_bound_w{points_w.row(0).maxCoeff(), points_w.row(1).maxCoeff(),
                                                      points_w.row(2).maxCoeff()};
                    // calculate IoU
                    std::get<2>(tuple).IoU =
                        curl::IoU_surface_area(lower_bound_w, upper_bound_w, std::get<1>(tuple)->get_lower_bound_w(),
                                               std::get<1>(tuple)->get_upper_bound_w());
                } else {
                    std::get<2>(tuple).IoU = 0;
                }
            }
        }
    }

    void sort_succ_associations_IoU_history(
        const std::shared_ptr<KeyframeInfo<BasicType>> &keyframePtr, const int valid_keyframe_idx,
        const int max_patches_region, const double history_weight_factor,
        std::unordered_set<std::shared_ptr<PatchInfo<BasicType>>> &history_patch_ptr,
        std::vector<std::vector<std::tuple<Eigen::MatrixX<BasicType>, std::shared_ptr<PatchInfo<BasicType>>, SCORE>>>
            &succ_associations) {
        for (auto &vec : succ_associations) {
            for (auto &tuple : vec) {
                if (history_patch_ptr.find(std::get<1>(tuple)) != history_patch_ptr.end()) {
                    // add the score with one if the patch is associated in the history keyframe
                    std::get<2>(tuple).IoU_history =
                        std::get<2>(tuple).IoU + history_weight_factor; // give additional weight for multi-observation
                } else {
                    std::get<2>(tuple).IoU_history =
                        std::get<2>(tuple).IoU; // give additional weight for multi-observation
                }
            }

            std::sort(vec.begin(), vec.end(),
                      [](const std::tuple<Eigen::MatrixX<BasicType>, std::shared_ptr<PatchInfo<BasicType>>, SCORE> &a,
                         const std::tuple<Eigen::MatrixX<BasicType>, std::shared_ptr<PatchInfo<BasicType>>, SCORE> &b) {
                          return std::get<2>(a).IoU_history > std::get<2>(b).IoU_history;
                      });
            int region_counter = 0;
            for (auto &tuple : vec) {
                if (std::get<1>(tuple) != nullptr && !std::get<1>(tuple)->is_generated_by_keyframe &&
                    std::get<2>(tuple).IoU > 0.3) {
                    PatchInfo<BasicType>::update_BA_T_obj_lidar(std::get<1>(tuple), keyframePtr, std::get<0>(tuple));
                    std::get<0>(tuple) = Eigen::MatrixX<BasicType>();
                    std::get<1>(tuple) = nullptr;
                    std::get<2>(tuple).IoU = 0;
                    std::get<2>(tuple).IoU_history = 0;
                }
                if ((std::get<1>(tuple) == nullptr) || (!std::get<1>(tuple)->is_generated_by_keyframe) ||
                    std::get<1>(tuple)->patch_procession_ptr->get_sph_coeff_data() == nullptr ||
                    std::get<1>(tuple)->patch_procession_ptr->get_sph_coeff_sum() == 0 || std::get<2>(tuple).IoU == 0 ||
                    std::get<1>(tuple)->keyframe_ptr->frame_idx < valid_keyframe_idx) {
                    continue;
                }
                history_patch_ptr.insert(std::get<1>(tuple));
                ++region_counter;
                if (max_patches_region != -1 && region_counter >= max_patches_region) {
                    break;
                }
            }
        }
    }

    void sort_succ_associations_IoU(
        std::vector<std::vector<std::tuple<Eigen::MatrixX<BasicType>, std::shared_ptr<PatchInfo<BasicType>>, SCORE>>>
            &succ_associations) {
        for (auto &vec : succ_associations) {
            for (auto &tuple : vec) {
                // TODO: calculate IoU here
                Eigen::MatrixX<BasicType> observed_pts_w =
                    (std::get<1>(tuple)->keyframe_ptr->get_BA_T_w_lidar()(Eigen::seq(0, 2), Eigen::seq(0, 2)) *
                     std::get<0>(tuple))
                        .colwise() +
                    std::get<1>(tuple)->keyframe_ptr->get_BA_T_w_lidar()(Eigen::seq(0, 2), 3);
                std::vector<double> observed_pts_lower_bound_w(3);
                std::vector<double> observed_pts_upper_bound_w(3);
                // extend a little bit
                Eigen::Map<Eigen::Vector3d>(observed_pts_lower_bound_w.data()) =
                    observed_pts_w.rowwise().minCoeff().array() - 0.2;
                Eigen::Map<Eigen::Vector3d>(observed_pts_upper_bound_w.data()) =
                    observed_pts_w.rowwise().maxCoeff().array() + 0.2;
                std::pair<std::vector<double>, std::vector<double>> prior_bound_pair =
                    std::get<1>(tuple)->get_bound_w_for_BA_association();
                std::get<2>(tuple).IoU =
                    curl::intersected_surface_area(observed_pts_lower_bound_w, observed_pts_upper_bound_w,
                                                   prior_bound_pair.first, prior_bound_pair.second);
            }
            std::sort(vec.begin(), vec.end(),
                      [](const std::tuple<Eigen::MatrixX<BasicType>, std::shared_ptr<PatchInfo<BasicType>>, SCORE> &a,
                         const std::tuple<Eigen::MatrixX<BasicType>, std::shared_ptr<PatchInfo<BasicType>>, SCORE> &b) {
                          return std::get<2>(a).IoU > std::get<2>(b).IoU;
                      });
        }
    }

    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_w_lidar;
    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> loopClosure_T_w_lidar;
    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> BA_T_w_lidar;
    Eigen::Isometry3d T_j_1_j;
    Pose3d graph_pose_w_lidar;
    Pose3d graph_obs_j_1_j;
    std::vector<double> lower_bound;
    std::vector<double> upper_bound;
    std::vector<std::tuple<Eigen::MatrixX<BasicType>, std::shared_ptr<PatchInfo<BasicType>>, double>>
        succeed_associations; // point cloud, corresponding patch, weight
    std::vector<std::vector<std::tuple<Eigen::MatrixX<BasicType>, std::shared_ptr<PatchInfo<BasicType>>, SCORE>>>
        seg_succeed_associations; // point cloud, corresponding patch, weight
    std::vector<std::vector<std::tuple<Eigen::MatrixX<BasicType>, std::shared_ptr<PatchInfo<BasicType>>, SCORE>>>
        ground_succeed_associations; // point cloud, corresponding patch, weight
    bool is_IoU_sorted;
};
template <typename BasicType> std::shared_mutex KeyframeInfo<BasicType>::BA_T_w_lidar_lock;
template <typename BasicType> std::shared_mutex KeyframeInfo<BasicType>::T_w_lidar_lock;

template <typename BasicType> struct key_frame_info_comparator {
    bool operator()(const std::shared_ptr<KeyframeInfo<BasicType>> &lhs,
                    const std::shared_ptr<KeyframeInfo<BasicType>> &rhs) const {
        return lhs->time < rhs->time;
    }
};
// BOOST_CLASS_EXPORT(KeyframeInfo<BT>)

template <typename BasicType> struct PatchInfo {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PatchInfo() = default;
    PatchInfo(const std::shared_ptr<KeyframeInfo<BasicType>> &_keyframe_ptr, Eigen::Isometry3d _T_obj_lidar,
              std::shared_ptr<PatchProcession<BasicType>> _patch_procession_ptr, int _status = curl::GOOD)
        : keyframe_ptr(_keyframe_ptr), T_obj_lidar(std::move(_T_obj_lidar)),
          patch_procession_ptr(_patch_procession_ptr), status(_status) {
        BA_pts_lower_bound_w.resize(3);
        BA_pts_upper_bound_w.resize(3);
        Eigen::Map<Eigen::Vector3d>(BA_pts_lower_bound_w.data()) = Eigen::Vector3d::Zero();
        Eigen::Map<Eigen::Vector3d>(BA_pts_upper_bound_w.data()) = Eigen::Vector3d::Zero();
        pts_number = 0;
        is_map_gen = false;
        is_ground = false;
        color = std::array<float, 3>{1, 0, 0};
        data_asso_counter = 0;
    }
    // for odometry
    PatchInfo(const std::shared_ptr<KeyframeInfo<BasicType>> &_keyframe_ptr, Eigen::Isometry3d _T_obj_lidar,
              std::shared_ptr<PatchProcession<BasicType>> _patch_procession_ptr, bool _is_ground,
              bool _is_generated_by_keyframe, Eigen::Matrix4<BasicType> _cov_4, PROJECTION_AXIS _projection_axis,
              int _pts_number, std::array<float, 3> _color = std::array<float, 3>{1, 0, 0}, int _status = curl::GOOD)
        : keyframe_ptr(_keyframe_ptr), T_obj_lidar(std::move(_T_obj_lidar)),
          patch_procession_ptr(_patch_procession_ptr), pts_number(_pts_number), is_ground(_is_ground),
          is_generated_by_keyframe(_is_generated_by_keyframe), cov_4(_cov_4), projection_axis(_projection_axis),
          color(_color), status(_status) {
        create_keyframe_num = keyframe_ptr->frame_num;
        last_update_keyframe_num = keyframe_ptr->frame_num;
        last_update_keyframe_idx = keyframe_ptr->frame_idx;
        BA_pts_lower_bound_w.resize(3);
        BA_pts_upper_bound_w.resize(3);
        Eigen::Map<Eigen::Vector3d>(BA_pts_lower_bound_w.data()) = Eigen::Vector3d::Zero();
        Eigen::Map<Eigen::Vector3d>(BA_pts_upper_bound_w.data()) = Eigen::Vector3d::Zero();
        is_map_gen = false;
        if (is_ground) {
            color = {0.804, 0.522, 0.247};
        }
        data_asso_counter = 0;
    }
    // for map generation
    PatchInfo(Eigen::Isometry3d _map_gen_T_w_obj, std::shared_ptr<PatchProcession<BasicType>> _patch_procession_ptr,
              std::array<float, 3> _color = std::array<float, 3>{1, 0, 0}, int _status = curl::GOOD)
        : map_gen_T_w_obj(std::move(_map_gen_T_w_obj)), patch_procession_ptr(_patch_procession_ptr), color(_color),
          status(_status) {
        BA_pts_lower_bound_w.resize(3);
        BA_pts_upper_bound_w.resize(3);
        Eigen::Map<Eigen::Vector3d>(BA_pts_lower_bound_w.data()) = Eigen::Vector3d::Zero();
        Eigen::Map<Eigen::Vector3d>(BA_pts_upper_bound_w.data()) = Eigen::Vector3d::Zero();
        pts_number = 0;
        is_ground = false;
        is_map_gen = true;
        if (is_ground) {
            color = {0.804, 0.522, 0.247};
        }
        data_asso_counter = 0;
    }

    PatchInfo(Eigen::Isometry3d _map_gen_T_w_obj, std::shared_ptr<PatchProcession<BasicType>> _patch_procession_ptr,
              bool _is_ground, std::array<float, 3> _color = std::array<float, 3>{1, 0, 0}, int _status = curl::GOOD)
        : map_gen_T_w_obj(std::move(_map_gen_T_w_obj)), patch_procession_ptr(_patch_procession_ptr),
          is_ground(_is_ground), color(_color), status(_status) {
        BA_pts_lower_bound_w.resize(3);
        BA_pts_upper_bound_w.resize(3);
        Eigen::Map<Eigen::Vector3d>(BA_pts_lower_bound_w.data()) = Eigen::Vector3d::Zero();
        Eigen::Map<Eigen::Vector3d>(BA_pts_upper_bound_w.data()) = Eigen::Vector3d::Zero();
        pts_number = 0;
        is_map_gen = true;
        if (is_ground) {
            color = {0.804, 0.522, 0.247};
        }
        data_asso_counter = 0;
    }

    void set_enlarged_lower_bound_w(const std::vector<double> &enlarged_lower_bound_w) {
        enlarged_lower_bound_w_ptr = std::make_unique<std::vector<double>>(enlarged_lower_bound_w);
    }

    std::vector<double> get_enlarged_lower_bound_w() const { return *enlarged_lower_bound_w_ptr; }

    void set_enlarged_upper_bound_w(const std::vector<double> &enlarged_upper_bound_w) {
        enlarged_upper_bound_w_ptr = std::make_unique<std::vector<double>>(enlarged_upper_bound_w);
    }

    std::vector<double> get_enlarged_upper_bound_w() const { return *enlarged_upper_bound_w_ptr; }

    void set_lower_bound_w(const std::vector<double> &lower_bound_w) {
        lower_bound_w_ptr = std::make_unique<std::vector<double>>(lower_bound_w);
    }

    std::vector<double> get_lower_bound_w() { return *lower_bound_w_ptr; }

    void set_upper_bound_w(const std::vector<double> &upper_bound_w) {
        upper_bound_w_ptr = std::make_unique<std::vector<double>>(upper_bound_w);
    }

    std::vector<double> get_upper_bound_w() { return *upper_bound_w_ptr; }

    void clear_content() { patch_procession_ptr->clear_content(); }

    void recover_content() { patch_procession_ptr->recover_data(); }

    void merge_and_resample_original_points(double voxel_size, double outlier_rejection_leaf_size,
                                            int outlier_rejection_minimum_neighbours) {
        if (!original_points_lidar_vec.empty()) {
            // std::unique_lock<std::shared_mutex> ul_original_points_lidar(original_points_lidar_vec_lock);
            std::vector<int> key(3);
            for (const auto &eigen_pts : original_points_lidar_vec) {
                for (int i = 0; i < eigen_pts.cols(); ++i) {
                    key[0] = static_cast<int>(eigen_pts(0, i) / voxel_size);
                    key[1] = static_cast<int>(eigen_pts(1, i) / voxel_size);
                    key[2] = static_cast<int>(eigen_pts(2, i) / voxel_size);
                    if (original_points_hash_table.find(key) == original_points_hash_table.end()) {
                        original_points_hash_table[key] = eigen_pts.col(i);
                        Eigen::Vector3<BasicType> pts_obj =
                            T_obj_lidar.matrix()(Eigen::seq(0, 2), Eigen::seq(0, 2)).cast<BasicType>() *
                                eigen_pts.col(i) +
                            T_obj_lidar.matrix()(Eigen::seq(0, 2), 3).cast<BasicType>();
                        patch_procession_ptr->update_BA_mask_idx(pts_obj);
                    }
                }
            }
            original_points_lidar_vec.clear();
            // Eigen::MatrixX<BasicType> original_points_obj = patch_procession_ptr->get_BA_prior_points_obj();
            // Eigen::Matrix4d T_lidar_obj = T_obj_lidar.inverse().matrix();
            // original_points_lidar =
            //     (T_lidar_obj(Eigen::seq(0, 2), Eigen::seq(0, 2)).cast<BasicType>() * original_points_obj).colwise() +
            //     T_lidar_obj(Eigen::seq(0, 2), 3).cast<BasicType>();
            original_points_lidar.resize(3, original_points_hash_table.size());
            int idx = 0;
            for (const auto &it : original_points_hash_table) {
                original_points_lidar.col(idx) = it.second;
                ++idx;
            }
        }
    }

    void original_points_lidar_vec_push_back(const Eigen::MatrixX<BasicType> &_original_points_lidar) {
        // std::unique_lock<std::shared_mutex> ul_original_points_lidar(original_points_lidar_vec_lock);
        original_points_lidar_vec.push_back(_original_points_lidar);
    }

    void original_points_lidar_vec_emplace_back(const Eigen::MatrixX<BasicType> &_original_points_lidar) {
        // std::unique_lock<std::shared_mutex> ul_original_points_lidar(original_points_lidar_vec_lock);
        original_points_lidar_vec.emplace_back(_original_points_lidar);
    }

    void original_points_lidar_vec_update_back(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &T_w_j,
                                               const Eigen::MatrixX<BasicType> &pts_j) {
        Eigen::Matrix4d T_i_j = Eigen::Isometry3d(keyframe_ptr->get_T_w_lidar()).inverse().matrix() * T_w_j;
        Eigen::MatrixX<BasicType> points_i =
            (T_i_j(Eigen::seq(0, 2), Eigen::seq(0, 2)).template cast<BasicType>() * pts_j).colwise() +
            T_i_j(Eigen::seq(0, 2), 3).template cast<BasicType>();
        original_points_lidar_vec_emplace_back(points_i);
    }

    void original_points_lidar_vec_clear() {
        // std::unique_lock<std::shared_mutex> ul_original_points_lidar(original_points_lidar_vec_lock);
        original_points_lidar_vec.clear();
    }

    const std::vector<Eigen::MatrixX<BasicType>> &get_original_points_lidar_vec() {
        // std::shared_lock<std::shared_mutex> sl_original_points_lidar(original_points_lidar_vec_lock);
        return original_points_lidar_vec;
    }

    const std::unordered_map<std::vector<int>, Eigen::Vector3<BasicType>, VoxelHashFuncPrime> &
    get_original_points_hash_table() {
        return original_points_hash_table;
    }

    const Eigen::MatrixX<BasicType> &get_original_points_lidar() { return original_points_lidar; }

    static void update_BA_T_obj_lidar(const std::shared_ptr<PatchInfo<BasicType>> &self_patch_ptr,
                                      const std::shared_ptr<KeyframeInfo<BasicType>> &new_keyframe_ptr,
                                      const Eigen::MatrixX<BasicType> &points_newLidar) {
        assert(!self_patch_ptr->is_generated_by_keyframe);
        Eigen::Matrix4d T_w_lidarNew = new_keyframe_ptr->get_T_w_lidar();
        self_patch_ptr->T_obj_lidar = self_patch_ptr->T_obj_lidar *
                                      Eigen::Isometry3d(self_patch_ptr->keyframe_ptr->get_T_w_lidar()).inverse() *
                                      T_w_lidarNew;
        PatchId patch_id = self_patch_ptr->key;
        self_patch_ptr->keyframe_ptr->local_patches.erase(patch_id);
        self_patch_ptr->keyframe_ptr = new_keyframe_ptr;
        self_patch_ptr->keyframe_ptr->local_patches.insert(patch_id);
        self_patch_ptr->original_points_lidar_vec_clear();
        self_patch_ptr->original_points_lidar_vec_emplace_back(points_newLidar);
        self_patch_ptr->is_generated_by_keyframe = true;
    }

    // Eigen::Isometry3d T_obj_lidar_gt;
    // Eigen::Isometry3d T_obj_j;
    // Eigen::Isometry3d T_w_j;
    // Eigen::Isometry3d T_w_j_gt;
    std::shared_ptr<KeyframeInfo<BasicType>> keyframe_ptr;
    int create_keyframe_num;
    int last_update_keyframe_num;
    int last_update_keyframe_idx;
    Eigen::Isometry3d T_obj_lidar;
    Eigen::Isometry3d map_gen_T_w_obj;
    PatchId key;
    bool is_ground;
    bool is_generated_by_keyframe;
    bool is_map_gen;
    int status;
    std::array<float, 3> color;
    std::shared_ptr<PatchProcession<BasicType>> patch_procession_ptr;
    unsigned int data_asso_counter;
    PROJECTION_AXIS projection_axis;
    Eigen::Matrix4<BasicType> cov_4;
    int pts_number;
    Eigen::Vector3d centroid_lidar;

    std::unordered_set<int> label_frame_num_set;

    std::mutex patch_update_lock;

    bool is_contain_label_frame_num(int frame_num) {
        return (label_frame_num_set.find(frame_num) != label_frame_num_set.end());
    }

    // need to update this after loop closure pose graph optimization
    std::vector<Eigen::MatrixX<BasicType>> initial_points_w_vec;
    std::vector<BasicType> range_squared_distance_vec;

    bool update_patch_projection_plane(const Eigen::MatrixX<BasicType> &pts_lidar,
                                       const Eigen::Matrix4<BasicType> &_T_w_j,
                                       int minimum_points_to_fix_projection_direction) {
        if (is_ground || pts_number == -1) { // don't update if updated once
            initial_points_w_vec = std::vector<Eigen::MatrixX<BasicType>>();
            range_squared_distance_vec = std::vector<BasicType>();
            return false;
        }
        std::vector<double> lower_bound_w = get_lower_bound_w();
        std::vector<double> upper_bound_w = get_upper_bound_w();
        Eigen::MatrixX<BasicType> pts_w =
            (_T_w_j(Eigen::seq(0, 2), Eigen::seq(0, 2)) * pts_lidar).colwise() + _T_w_j(Eigen::seq(0, 2), 3);
        initial_points_w_vec.push_back(pts_w);
        Eigen::VectorX<BasicType> range_squared_lidar = pts_lidar.colwise().squaredNorm();
        range_squared_distance_vec.insert(range_squared_distance_vec.end(), range_squared_lidar.data(),
                                          range_squared_lidar.data() + range_squared_lidar.size());
        pts_number += pts_w.cols();
        if (pts_number < minimum_points_to_fix_projection_direction) {
            return false;
        }
        Eigen::Matrix4<BasicType> T_w_lidar = (keyframe_ptr->get_T_w_lidar()).template cast<BasicType>();
        Eigen::Matrix4<BasicType> T_lidar_w =
            Eigen::Isometry3d(keyframe_ptr->get_T_w_lidar()).inverse().matrix().template cast<BasicType>();
        Eigen::MatrixX<BasicType> patch_cloud_lidar(3, pts_number);
        for (int i = 0, start_idx = 0; i < initial_points_w_vec.size(); ++i) {
            patch_cloud_lidar(Eigen::all, Eigen::seq(start_idx, start_idx + initial_points_w_vec[i].cols() - 1)) =
                (T_lidar_w(Eigen::seq(0, 2), Eigen::seq(0, 2)) * initial_points_w_vec[i]).colwise() +
                T_lidar_w(Eigen::seq(0, 2), 3);
            start_idx += initial_points_w_vec[i].cols();
        }

        Eigen::MatrixX<BasicType> patch_cloud_w =
            (T_w_lidar(Eigen::seq(0, 2), Eigen::seq(0, 2)) * patch_cloud_lidar).colwise() +
            T_w_lidar(Eigen::seq(0, 2), 3);
        bool is_proj_axis_changed = curl::is_update_local_coordinate_with_eig(
            Eigen::Isometry3d(keyframe_ptr->get_T_w_lidar()), patch_cloud_w, pts_number, lower_bound_w, upper_bound_w,
            T_obj_lidar, cov_4, projection_axis);
        if (is_proj_axis_changed) {
            Eigen::Map<Eigen::VectorX<BasicType>> range_squared_distance_eigen(range_squared_distance_vec.data(),
                                                                               range_squared_distance_vec.size());
            Eigen::MatrixX<BasicType> patch_cloud_obj =
                (T_obj_lidar.matrix()(Eigen::seq(0, 2), Eigen::seq(0, 2)).cast<BasicType>() * patch_cloud_lidar)
                    .colwise() +
                T_obj_lidar.matrix()(Eigen::seq(0, 2), 3).cast<BasicType>();
            patch_procession_ptr->update_projection_plane(patch_cloud_obj, range_squared_distance_eigen);
            // save patch_cloud_obj
        }

        initial_points_w_vec = std::vector<Eigen::MatrixX<BasicType>>();
        range_squared_distance_vec = std::vector<BasicType>();
        pts_number = -1;
        return is_proj_axis_changed;
    }

    bool update_patch_projection_plane_low_RAM(const Eigen::MatrixX<BasicType> &pts_lidar,
                                               const Eigen::Matrix4<BasicType> &_T_w_j,
                                               int minimum_points_to_fix_projection_direction) {
        if (is_ground || pts_number == -1) { // don't update if updated once
            initial_points_w_vec = std::vector<Eigen::MatrixX<BasicType>>();
            range_squared_distance_vec = std::vector<BasicType>();
            return false;
        }
        std::vector<double> lower_bound_w = get_lower_bound_w();
        std::vector<double> upper_bound_w = get_upper_bound_w();
        Eigen::MatrixX<BasicType> pts_w =
            (_T_w_j(Eigen::seq(0, 2), Eigen::seq(0, 2)) * pts_lidar).colwise() + _T_w_j(Eigen::seq(0, 2), 3);
        initial_points_w_vec.push_back(pts_w);
        Eigen::VectorX<BasicType> range_squared_lidar = pts_lidar.colwise().squaredNorm();
        range_squared_distance_vec.insert(range_squared_distance_vec.end(), range_squared_lidar.data(),
                                          range_squared_lidar.data() + range_squared_lidar.size());
        pts_number += pts_w.cols();
        if (pts_number < minimum_points_to_fix_projection_direction) {
            return false;
        }
        Eigen::Matrix4<BasicType> T_w_lidar = (keyframe_ptr->get_T_w_lidar()).template cast<BasicType>();
        Eigen::Matrix4<BasicType> T_lidar_w =
            Eigen::Isometry3d(keyframe_ptr->get_T_w_lidar()).inverse().matrix().template cast<BasicType>();
        Eigen::MatrixX<BasicType> patch_cloud_lidar(3, pts_number);
        for (int i = 0, start_idx = 0; i < initial_points_w_vec.size(); ++i) {
            patch_cloud_lidar(Eigen::all, Eigen::seq(start_idx, start_idx + initial_points_w_vec[i].cols() - 1)) =
                (T_lidar_w(Eigen::seq(0, 2), Eigen::seq(0, 2)) * initial_points_w_vec[i]).colwise() +
                T_lidar_w(Eigen::seq(0, 2), 3);
            start_idx += initial_points_w_vec[i].cols();
        }

        Eigen::MatrixX<BasicType> patch_cloud_w =
            (T_w_lidar(Eigen::seq(0, 2), Eigen::seq(0, 2)) * patch_cloud_lidar).colwise() +
            T_w_lidar(Eigen::seq(0, 2), 3);
        bool is_proj_axis_changed = curl::is_update_local_coordinate_with_eig(
            Eigen::Isometry3d(keyframe_ptr->get_T_w_lidar()), patch_cloud_w, pts_number, lower_bound_w, upper_bound_w,
            T_obj_lidar, cov_4, projection_axis);
        if (is_proj_axis_changed) {
            Eigen::MatrixX<BasicType> patch_cloud_obj =
                (T_obj_lidar.matrix()(Eigen::seq(0, 2), Eigen::seq(0, 2)).cast<BasicType>() * patch_cloud_lidar).colwise() +
                T_obj_lidar.matrix()(Eigen::seq(0, 2), 3).cast<BasicType>();
            patch_procession_ptr->update_projection_plane_low_RAM(patch_cloud_obj);
            // save patch_cloud_obj
        }

        initial_points_w_vec = std::vector<Eigen::MatrixX<BasicType>>();
        range_squared_distance_vec = std::vector<BasicType>();
        pts_number = -1;
        return is_proj_axis_changed;
    }

    bool update_cov_4(const Eigen::Matrix4<BasicType> &_cov_4, int _pts_number) {
        cov_4 += _cov_4;
        pts_number += _pts_number;
        bool is_proj_axis_changed = false;
        if (!is_ground) {
            Eigen::Matrix3<BasicType> cov =
                (cov_4(Eigen::seq(0, 2), Eigen::seq(0, 2)) -
                 (1.0 / static_cast<double>(pts_number)) * cov_4(Eigen::seq(0, 2), 3) * cov_4(3, Eigen::seq(0, 2)))
                    .array() /
                static_cast<double>(pts_number);
            //        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3<BasicType>> es;
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3<BasicType>> es;
            es.compute(cov);
            Eigen::Vector3<BasicType> shortest_axis = es.eigenvectors().col(0);
            BasicType dot_with_x = std::abs(shortest_axis.dot(Eigen::Vector3<BasicType>::UnitX()));
            BasicType dot_with_y = std::abs(shortest_axis.dot(Eigen::Vector3<BasicType>::UnitY()));
            BasicType dot_with_z = std::abs(shortest_axis.dot(Eigen::Vector3<BasicType>::UnitZ()));
            Eigen::Matrix3d R_rot;
            if (dot_with_x > dot_with_y && dot_with_x > dot_with_z) {
                if (projection_axis == PROJECTION_AXIS::Z_AXIS ||
                    (projection_axis != PROJECTION_AXIS::X_AXIS && dot_with_x > 0.8)) {
                    R_rot = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitX()).toRotationMatrix() *
                            Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitY()).toRotationMatrix();
                    projection_axis = PROJECTION_AXIS::X_AXIS;
                    is_proj_axis_changed = true;
                }
            } else if (dot_with_y > dot_with_x && dot_with_y > dot_with_z ||
                       (projection_axis != PROJECTION_AXIS::Y_AXIS && dot_with_y > 0.8)) {
                if (projection_axis == PROJECTION_AXIS::Z_AXIS) {
                    R_rot = Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d::UnitX()).toRotationMatrix() *
                            Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
                    projection_axis = PROJECTION_AXIS::Y_AXIS;
                    is_proj_axis_changed = true;
                }
            } else {
                if (projection_axis != PROJECTION_AXIS::Z_AXIS) {
                    R_rot.setIdentity();
                    projection_axis = PROJECTION_AXIS::Z_AXIS;
                    is_proj_axis_changed = true;
                }
            }
            if (is_proj_axis_changed) {
                Eigen::Isometry3d T_w_obj = Eigen::Isometry3d(keyframe_ptr->get_T_w_lidar()) * T_obj_lidar.inverse();
                T_w_obj.matrix()(Eigen::seq(0, 2), Eigen::seq(0, 2)) = R_rot;
                T_obj_lidar = T_w_obj.inverse() * Eigen::Isometry3d(keyframe_ptr->get_T_w_lidar());
            }
        }
        return is_proj_axis_changed;
    }

    Eigen::Vector3d get_centroid_w() {
        if (!is_map_gen) {
            //            return ((Eigen::Transform<double, 3, Eigen::Isometry,
            //            Eigen::DontAlign>(keyframe_ptr->get_T_w_lidar()) * T_obj_lidar.inverse()).translation());
            return (1.0 / static_cast<double>(pts_number)) * cov_4(Eigen::seq(0, 2), 3).template cast<double>();
        } else {
            return (map_gen_T_w_obj.translation());
        }
    }

    Eigen::Vector3d get_centroid_lidar() {
        Eigen::Isometry3d T_lidar_w = Eigen::Isometry3d(keyframe_ptr->get_T_w_lidar()).inverse();
        Eigen::Vector3d centroid_w =
            (1.0 / static_cast<double>(pts_number)) * cov_4(Eigen::seq(0, 2), 3).template cast<double>();
        centroid_lidar = T_lidar_w.matrix()(Eigen::seq(0, 2), Eigen::seq(0, 2)) * centroid_w +
                         T_lidar_w.matrix()(Eigen::seq(0, 2), 3);
        return centroid_lidar;
    }

    MapDataAsso data_asso;

    std::pair<std::vector<double>, std::vector<double>> get_bound_w_for_BA_association() {
        if (keyframe_ptr->is_active || !original_points_lidar_vec.empty()) {
            calculate_BA_pts_bound_w();
        }
        return std::make_pair(BA_pts_lower_bound_w, BA_pts_upper_bound_w);
    }

    // void set_original_points_lidar_shared_lock() { original_points_lidar_vec_lock.lock_shared(); }
    // template <class Archive> void serialize(Archive &ar, const unsigned int version) {
    //     ar &patch_procession_ptr;
    //     ar &keyframe_ptr;
    //     ar &is_ground;
    //     ar &T_obj_lidar;
    // }

  private:
    // void unset_original_points_lidar_shared_lock() { original_points_lidar_vec_lock.unlock_shared(); }
    void calculate_BA_pts_bound_w() {
        if (original_points_lidar_vec[0].cols() > 0) {
            // assume the original_points_lidar_vec has been merged
            Eigen::MatrixXd original_pts_w = (keyframe_ptr->get_BA_T_w_lidar()(Eigen::seq(0, 2), Eigen::seq(0, 2)) *
                                              original_points_lidar_vec[0].template cast<double>())
                                                 .colwise() +
                                             keyframe_ptr->get_BA_T_w_lidar()(Eigen::seq(0, 2), 3);
            BA_pts_lower_bound_w.resize(3);
            BA_pts_upper_bound_w.resize(3);
            Eigen::Map<Eigen::Vector3d>(BA_pts_lower_bound_w.data()) =
                original_pts_w.rowwise().minCoeff().array() - 0.2;
            Eigen::Map<Eigen::Vector3d>(BA_pts_upper_bound_w.data()) =
                original_pts_w.rowwise().maxCoeff().array() + 0.2;
        }
    }
    std::vector<double> BA_pts_lower_bound_w;
    std::vector<double> BA_pts_upper_bound_w;
    std::vector<Eigen::MatrixX<BasicType>> original_points_lidar_vec;
    std::unordered_map<std::vector<int>, Eigen::Vector3<BasicType>, VoxelHashFuncPrime> original_points_hash_table;
    double IoU;
    Eigen::MatrixX<BasicType> original_points_lidar;
    // std::shared_mutex original_points_lidar_vec_lock;
    std::unique_ptr<std::vector<double>> enlarged_lower_bound_w_ptr;
    std::unique_ptr<std::vector<double>> enlarged_upper_bound_w_ptr;
    std::unique_ptr<std::vector<double>> lower_bound_w_ptr;
    std::unique_ptr<std::vector<double>> upper_bound_w_ptr;

    // friend class boost::serialization::access;
};

#endif // SRC_MAP_ATTRIBUTES_H
