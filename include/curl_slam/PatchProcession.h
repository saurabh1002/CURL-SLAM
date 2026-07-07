//
// Created by zkc on 19/06/23.
//

#ifndef SRC_PatchProcessing_H
#define SRC_PatchProcessing_H
// #include <opencv2/opencv.hpp>
#include "curl_slam/curl_tools_light.h"
// #include "curl_slam/eigen_serialization.h"
#include "curl_slam/linear_interpolation_2.h"
#include "curl_slam/linear_interpolation_2_func.h"
#include "curl_slam/load_config.h"
// #include "opencv2/imgproc.hpp"
#include "curl_slam/nanoflann.hpp"
#include "types.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cmath>
#include <memory>
// #include <opencv2/core/eigen.hpp>
// #include <opencv2/core/mat.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <shared_mutex>
#include <vector>

// TODO: initialize the spherical harmonics coefficients, initialize the mask, the size of the I_mask and I should be
// fixed
// TODO: need to maintain a height mean image grid and a mask grid (maybe they can be the same, invalid height value can
// be used)
// TODO: add adaptive changing spherical harmonics degree
// TODO: update of the mask and spherical harmonics coefficients
// TODO: Figure out hole filling function (Use kd-tree for the first filling (find the nearest neighbour))
// TODO: For update, each new points get its nearest 8 points updated to its value

template <typename BasicType> class PatchProcession {
  private:
    // friend class boost::serialization::access;

    // begin low RAM elements
    Eigen::MatrixX<BasicType> AT_A_block;
    Eigen::VectorX<BasicType> AT_b_block;
    // end low RAM elements

    struct PYRAMID {
        std::vector<int> x_size_vec, y_size_vec;
        std::vector<BasicType> x_grid_len_vec, y_grid_len_vec;
        std::vector<Eigen::MatrixX<BasicType>> I_vec, I_Gx_vec, I_Gy_vec;
        int depth;
    };

    bool is_cleared;

    std::shared_ptr<DIRECT_METHOD_CONFIG> direct_method_config_ptr;
    std::shared_ptr<SH_TABLE_CONFIG<BasicType>> SH_table_config_ptr;
    std::shared_ptr<DEBUG_CONFIG> debug_config_ptr;

    // these two information can be abondoned after generating the image
    std::pair<interp_func_pair, interp_func_pair> conformal_mapping;
    std::pair<interp_func_pair, interp_func_pair> inv_conformal_mapping;

    interp_func_pair I_interp;
    interp_func_pair mask_F;
    bool is_build_mask_F;
    BasicType min_X, max_X, min_Y, max_Y;
    BasicType min_x, max_x, min_y, max_y;
    Eigen::VectorXd sph_coeff, BA_sph_coeff;
    // images
    int x_size, y_size;
    int X_size, Y_size;
    Eigen::MatrixX<BasicType> xy_grid;

    Eigen::MatrixX<BasicType> recons_v_local;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr recons_v_local_normal_ptr;
    pcl::search::KdTree<pcl::PointXYZINormal>::Ptr kd_tree_ptr;
    std::vector<int> mask_idx;  // store valid indices for I_mask in 1-dimension
    std::set<int> mask_idx_set; // store valid indices for I_mask in 1-dimension
    std::vector<uint8_t> mask_idx_flags; // low-RAM fast lookup for valid mask indices
    Eigen::MatrixX<BasicType> I, I_mask, I_mask_dense;
    Eigen::MatrixX<BasicType> I_mask_times, I_mask_weights;
    Eigen::MatrixXi I_mask_counters;
    std::atomic<bool> recons_v_local_dirty{true};
    // lock for image I relative write and read
    mutable std::shared_mutex I_lock;
    mutable std::shared_mutex xy_grid_lock;
    mutable std::shared_mutex v_recons_lock;
    mutable std::shared_mutex sph_coeff_lock;
    mutable std::shared_mutex mask_idx_lock;
    bool is_sph_coeff_lock;
    // residual
    bool is_initialization;
    int initialization_counter;
    int update_step_counter;

    // status for this patch; decide whether icp or direct method is going to be use
    int status;
    // degree
    int SH_degree, warping_SH_degree;
    // pyramid
    PYRAMID pyramids;

    bool is_ground;
    // derivative of theta and phi
    double G_theta;
    double G_phi;

    Eigen::MatrixX<BasicType> BA_I_mask_all;
    Eigen::MatrixX<BasicType> BA_I_mask;
    Eigen::MatrixX<BasicType> BA_I_mask_times;
    std::vector<int> BA_mask_idx;

    bool update_mask(const Eigen::MatrixX<BasicType> &_v_obj, const Eigen::VectorX<BasicType> &_v_range_squared);
    bool update_mask_low_RAM(const Eigen::MatrixX<BasicType> &_v_obj);
    void extract_sph_coeff();
    void extract_sph_coeff_low_RAM();
    void calculate_I_without_conformal_mapping();
    void calculate_delta_without_conformal_mapping();
    void calculate_recons_v_local_without_conformal_mapping();
    void calculate_recons_v_local_without_conformal_mapping_low_RAM();

    std::pair<BasicType, int> hist_fall_thres(std::vector<BasicType> &vec, BasicType percentage);
  public:
    enum class MatrixType { I, Gx, Gy };

    Eigen::Vector3d last_update_position;
    // template <class Archive> void serialize(Archive &ar, const unsigned int version) {
    //     // ar &is_ground;
    //     ar &min_x;
    //     ar &max_x;
    //     ar &max_y;
    //     ar &min_y;
    //     ar &x_size;
    //     ar &y_size;
    //     ar &mask_idx;
    //     ar &sph_coeff;
    //     ar &SH_degree;
    //     // ar &recons_v_local;
    // }
    void recover_data();
    bool is_set_BA_sph_coeff;
    static constexpr BasicType T_INVALID = std::numeric_limits<BasicType>::max();

    // static constexpr double DOUBLE_INVALID = std::numeric_limits<double>::max();
    // double DOUBLE_INVALID;
    // constructor for spatial without conformal-mapping odometry
    PatchProcession() = default;
    PatchProcession(const Eigen::Vector3d &_last_update_position,
                    const std::shared_ptr<DIRECT_METHOD_CONFIG> _direct_method_config_ptr,
                    const std::shared_ptr<SH_TABLE_CONFIG<BasicType>> _SH_table_config_ptr,
                    std::shared_ptr<DEBUG_CONFIG> _debug_config_ptr, const Eigen::MatrixX<BasicType> &_v_obj,
                    const Eigen::VectorX<BasicType> &_v_range_squared, const BasicType half_diag_side, int _SH_degree,
                    bool is_ground, bool _is_cleared = false, int _status = curl::GOOD);
    PatchProcession(const Eigen::Vector3d &_last_update_position,
                    const std::shared_ptr<DIRECT_METHOD_CONFIG> _direct_method_config_ptr,
                    const std::shared_ptr<SH_TABLE_CONFIG<BasicType>> _SH_table_config_ptr,
                    std::shared_ptr<DEBUG_CONFIG> _debug_config_ptr, const Eigen::MatrixX<BasicType> &_v_obj,
                    const Eigen::VectorX<BasicType> &_v_range_squared, const BasicType half_diag_side, int _SH_degree,
                    bool is_ground, int is_low_ram, bool _is_cleared = false, int _status = curl::GOOD);
    PatchProcession(const std::shared_ptr<DIRECT_METHOD_CONFIG> _direct_method_config_ptr,
                    const std::shared_ptr<SH_TABLE_CONFIG<BasicType>> _SH_table_config_ptr,
                    std::shared_ptr<DEBUG_CONFIG> _debug_config_ptr, const BasicType half_diag_side, int _SH_degree,
                    bool _is_ground, const Eigen::VectorXd _sph_coeff, std::vector<BasicType> binary_mask);
    ~PatchProcession();
    void update_projection_plane(const Eigen::MatrixX<BasicType> &v_obj,
                                 const Eigen::VectorX<BasicType> &_v_range_squared);
    void update_projection_plane_low_RAM(const Eigen::MatrixX<BasicType> &v_obj);
    bool update_sph_residuals_without_conformal_mapping(const Eigen::MatrixX<BasicType> &v_obj,
                                                        const Eigen::VectorX<BasicType> &_v_range_squared,
                                                        const Eigen::Vector3d &current_postition);
    bool update_sph_residuals_without_conformal_mapping_low_RAM(const Eigen::MatrixX<BasicType> &v_obj,
                                                                const Eigen::Vector3d &current_postition);

    bool icp_update_recons_v(const Eigen::MatrixX<BasicType> &v_obj);

    BasicType fetch_weights(const BasicType x, const BasicType y);

    BasicType get_pixel_weights(const BasicType x, const BasicType y);

    BasicType fetch_I_without_conformal_mapping(const std::string &matrix_name, const BasicType x, const BasicType y,
                                                const int pyramid_idx);
    BasicType fetch_I_without_conformal_mapping(MatrixType matrix_type, const BasicType x, const BasicType y,
                                                const int pyramid_idx);
    void reset();
    void reset_low_RAM();
    void clear_content();

    Eigen::MatrixX<BasicType> get_I(const int pyramid_idx);
    Eigen::MatrixX<BasicType> get_I_Gx(const int pyramid_idx);
    Eigen::MatrixX<BasicType> get_I_Gy(const int pyramid_idx);
    Eigen::MatrixX<BasicType> get_recons_v_obj();
    Eigen::MatrixX<BasicType> get_recons_v_obj_downsample(const int stride);
    Eigen::MatrixX<BasicType> get_recons_v_obj_cached();
    Eigen::MatrixX<BasicType> get_recons_v_local_cached();
    void mark_recons_v_local_dirty();
    void update_recons_v_local_if_dirty();
    bool is_recons_v_local_dirty() const;
    BasicType get_recons_height(const Eigen::VectorXd &_sph_coeff, const BasicType x, const BasicType y);
    BasicType get_theta(const BasicType y);
    BasicType get_phi(const BasicType x);
    std::vector<int> get_mask_idx();
    std::vector<int> get_BA_mask_idx();
    bool is_valid_in_BA_mask(const BasicType x, const BasicType y);
    bool is_valid_in_mask(const BasicType x, const BasicType y);
    bool is_valid_in_mask_low_RAM(const BasicType x, const BasicType y);
    BasicType get_height_value_BA_mask(const BasicType x, const BasicType y);
    int get_pyramid_depth() const;

    double get_score();
    double *get_sph_coeff_data();
    double *set_and_get_BA_sph_coeff_data();
    double *get_BA_sph_coeff_data();
    void commit_BA_sph_coeff();
    const Eigen::VectorXd &get_sph_coeff();
    const Eigen::VectorXd &get_BA_sph_coeff();
    double get_sph_coeff_sum();
    double get_G_theta() const noexcept;
    double get_G_phi() const noexcept;
    int get_SH_degree() const noexcept;
    void set_sph_coeff_lock();
    void unset_sph_coeff_lock();
    std::vector<Eigen::MatrixX<BasicType>> get_I_vec();
    std::vector<Eigen::MatrixX<BasicType>> get_I_Gx_vec();
    std::vector<Eigen::MatrixX<BasicType>> get_I_Gy_vec();
    std::vector<BasicType> get_x_grid_len_vec();
    std::vector<BasicType> get_y_grid_len_vec();
    Eigen::MatrixX<BasicType> get_xy_grid_dense(const int times);
    std::pair<bool, bool> check_is_on_edge(const BasicType x, const BasicType y);
    std::pair<bool, bool> check_is_on_edge_low_RAM(const BasicType x, const BasicType y);
    std::pair<bool, bool> BA_check_is_on_edge_simple(const BasicType x, const BasicType y);
    std::pair<bool, bool> BA_check_is_on_edge(const BasicType x, const BasicType y);
    // get regularization points using interpolation
    Eigen::MatrixX<BasicType> interpolates_xy_grid_points(const Eigen::MatrixX<BasicType> &seed_points);
    // Eigen::MatrixX<BasicType> get_recons_v_local();
    void update_BA_mask_idx(const Eigen::MatrixX<BasicType> &update_points_obj, const double radius,
                            const int minimum_num);
    void update_BA_mask_idx(const Eigen::Vector3<BasicType> &pts_obj);
    void update_BA_mask_idx(
        const std::unordered_map<std::vector<int>, Eigen::Vector3<BasicType>, VoxelHashFuncPrime> &hash_map_pts,
        Eigen::Isometry3d _T_obj_lidar);
    void update_BA_mask_idx_trackingWay(const Eigen::MatrixX<BasicType> &update_points_obj);
    Eigen::MatrixX<BasicType> get_BA_prior_points_obj();
    void set_config_ptr(const std::shared_ptr<SH_TABLE_CONFIG<BasicType>> &_SH_table_config_ptr,
                        const std::shared_ptr<DIRECT_METHOD_CONFIG> &_direct_method_config_ptr);
    Eigen::MatrixX<BasicType> calculate_recons_v_local_continuous_recons(const int patch_img_rso,
                                                                         const int SH_degree_CR);
    Eigen::MatrixX<BasicType> get_recons_v_local_continuous_recons();
    Eigen::VectorXi get_binary_mask();
    Eigen::MatrixX<BasicType> get_I_mask();
    int get_height_image_size();
};

#endif // SRC_PatchProcessing_H
