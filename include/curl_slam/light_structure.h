//
// Created by zkc on 19/06/23.
//

#ifndef SRC_LIGHT_STRUCTURE_H
#define SRC_LIGHT_STRUCTURE_H
// #include "linear_interpolation_2.h"
// #include "linear_interpolation_2_func.h"
#include "curl_slam/map_attributes.h"
#include <array>
#include <memory>
#include <tuple>
#include <vector>

template <typename T = double> struct Gamma {
    std::pair<interp_func_pair, interp_func_pair> interp_kernel;
    T min_x;
    T max_x;
    T min_y;
    T max_y;

    Gamma(const std::pair<interp_func_pair, interp_func_pair> &_interp_kernel, const Eigen::MatrixX<T> &points)
        : interp_kernel(_interp_kernel) {
        min_x = points.col(0).minCoeff();
        max_x = points.col(0).maxCoeff();
        min_y = points.col(1).minCoeff();
        max_y = points.col(1).maxCoeff();
    }
};

struct DataInfo {
    DataInfo(double _score, double _IoU) : score(_score), IoU(_IoU) {}
    double score;
    double IoU;
};

template <typename BasicType> struct PointCloudInfo {
    std::vector<std::vector<Eigen::Isometry3d>> T_obj_lidar_vec;
    double time;
    std::vector<std::vector<bool>> is_ground_cloud_vec;
    std::vector<std::vector<std::pair<std::vector<double>, std::vector<double>>>> bounding_box_pair_vec;
    std::vector<std::vector<std::vector<std::pair<std::shared_ptr<PatchInfo<BasicType>>, double>>>>
        intersected_score_pairs_vec; // region; raw cloud patch; associated patches and scores
    pcl::PointCloud<PointT>::Ptr seg_cloud_ptr;
    pcl::PointCloud<PointT>::Ptr ground_cloud_ptr;
};

template <typename BasicType> struct TupleCompare {
    bool operator()(const std::tuple<std::array<int, 2>, double, std::shared_ptr<PatchInfo<BasicType>>> &a,
                    const std::tuple<std::array<int, 2>, double, std::shared_ptr<PatchInfo<BasicType>>> &b) const {
        if (std::get<1>(a) != std::get<1>(b)) {
            return std::get<1>(a) < std::get<1>(b);
        }
        if (std::get<0>(a) != std::get<0>(b)) {
            return std::get<0>(a) < std::get<0>(b);
        }
        return std::get<2>(a) < std::get<2>(b);
    }
};

struct PairCompare {
    bool operator()(const std::pair<int, double> &a, const std::pair<int, double> &b) const {
        if (a.first == b.first)
            return false;
        else
            return a.second <= b.second; // the equality of second value is allowed
    }
};

struct LoopClosureConstrain {
    LoopClosureConstrain(int _history_frame_idx, int _current_frame_idx, const Pose3d &_T_his_curr)
        : history_frame_idx(_history_frame_idx), current_frame_idx(_current_frame_idx), T_his_curr(_T_his_curr) {}
    int history_frame_idx;
    int current_frame_idx;
    Pose3d T_his_curr;
};

template <typename T> struct KeyframePairEqual {
    bool operator()(const std::pair<std::shared_ptr<KeyframeInfo<T>>, std::shared_ptr<KeyframeInfo<T>>> &lhs,
                    const std::pair<std::shared_ptr<KeyframeInfo<T>>, std::shared_ptr<KeyframeInfo<T>>> &rhs) const {
        return (lhs.first == rhs.first && lhs.second == rhs.second) ||
               (lhs.first == rhs.second && lhs.second == rhs.first);
    }
};

template <typename T> struct KeyframePairHash {
    size_t operator()(const std::pair<std::shared_ptr<KeyframeInfo<T>>, std::shared_ptr<KeyframeInfo<T>>> &p) const {
        std::hash<typename std::shared_ptr<KeyframeInfo<T>>::element_type *> ptr_hasher;
        auto hash1 = ptr_hasher(p.first.get());
        auto hash2 = ptr_hasher(p.second.get());
        return hash1 ^ (hash2 << 1); // Shift hash2 to avoid colliding hashes from hash1
    }
};

template <typename T> struct keyframe_associated_num_comparator {
    bool operator()(const std::shared_ptr<KeyframeInfo<T>> &lhs, const std::shared_ptr<KeyframeInfo<T>> &rhs) const {
        return lhs->get_loop_closure_associated_times() > rhs->get_loop_closure_associated_times();
    }
};

struct COUNTER {
    int counter;
    COUNTER() : counter(0) {}
    void operator++(int) { counter++; }
    void operator--(int) { counter--; }
    void operator=(int _counter) { counter = _counter; }
    int operator()() { return counter; }
    bool operator<(COUNTER Counter) { return counter < Counter.counter; }
    bool operator>(COUNTER Counter) { return counter > Counter.counter; }
};

#endif // SRC_LIGHT_STRUCTURE_H
