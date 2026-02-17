#ifndef CURLCONTINUOUSRECONS_H
#define CURLCONTINUOUSRECONS_H
#include "curl_slam/CurlVoxelMapping.h"
#include "curl_slam/triangluation_mask_identification.h"
#include <curl_slam/ContinuousReconsConfig.h>
#include <dynamic_reconfigure/server.h>
#include <open3d/Open3D.h>
#include <ros/ros.h>

struct SimplePatch {
    double D_INVALID = std::numeric_limits<double>::max();

    int default_img_iso;
    bool is_default_mesh_calculated = false;
    curl::Delaunay_triangulation_linear default_mesh;
    std::vector<double> lengths, areas;

    template <typename BasicType>
    void calculate_default_recons_v_mesh(const std::shared_ptr<SH_TABLE_CONFIG<BasicType>> SH_table_config_ptr);

    double half_diag_side;
    double min_x, max_x, min_y, max_y;
    bool is_ground;
    int sph_degree;
    Eigen::Matrix4d T_w_obj;
    Eigen::VectorXd sph_coeff;
    std::vector<int> mask_idx_vec;

    template <typename BasicType>
    Eigen::MatrixXd
    calculate_recons_v_local_continuous_recons(const int patch_img_rso, int sph_degree_CR, double IQR_factor,
                                               const bool is_fix_length_threshold, const double _length_thres,
                                               const std::shared_ptr<SH_TABLE_CONFIG<BasicType>> SH_table_config_ptr);
    void set_boundary(const double _half_diag_side);
    Eigen::VectorXi get_binary_mask() const;
};

template <typename BasicType> class CurlContinuousRecons {
  private:
    ros::NodeHandle *nh;
    open3d::visualization::Visualizer open3d_vis;
    std::shared_ptr<open3d::geometry::PointCloud> open3d_display_pcd_ptr;
    Eigen::Vector3d open3d_display_pcd_center;
    std::vector<SimplePatch> simple_patches;

  public:
    dynamic_reconfigure::Server<curl_slam::ContinuousReconsConfig> server;
    dynamic_reconfigure::Server<curl_slam::ContinuousReconsConfig>::CallbackType f;
    std::shared_ptr<CURL_CONTINUOUS_RECONS_CONFIG> curl_continuous_recons_config_ptr;
    CURL_CONTINUOUS_RECONS_CONFIG current_curl_continuous_recons_config;
    std::shared_ptr<CURL_VOXEL_MAPPING_CONFIG> curl_voxel_mapping_config_ptr;
    std::shared_ptr<SH_TABLE_CONFIG<BasicType>> SH_table_config_ptr;
    std::shared_ptr<DIRECT_METHOD_CONFIG> direct_method_config_ptr;
    std::shared_ptr<DEBUG_CONFIG> debug_config_ptr;

    std::shared_ptr<SpatialHashing<BasicType>> spatial_hashing_ptr;
    open3d::geometry::PointCloud open3d_pcd;

  public:
    CurlContinuousRecons() = default;
    CurlContinuousRecons(ros::NodeHandle *nh,
                         const std::shared_ptr<CURL_CONTINUOUS_RECONS_CONFIG> _curl_continuous_recons_config_ptr,
                         const std::shared_ptr<CURL_VOXEL_MAPPING_CONFIG> &_curl_voxel_mapping_config_ptr,
                         const std::shared_ptr<SH_TABLE_CONFIG<BasicType>> &_SH_table_config_ptr,
                         const std::shared_ptr<DIRECT_METHOD_CONFIG> &_direct_method_config_ptr,
                         const std::shared_ptr<DEBUG_CONFIG> &_debug_config_ptr);
    std::unordered_set<std::shared_ptr<PatchInfo<BasicType>>> all_saving_patches_ptr;
    bool load_from_binary(const std::string &filename);
    void run();
    void run_reconstruction();
    void param_callback(curl_slam::ContinuousReconsConfig &config, uint32_t level);

  private:
    void reconstruction();
    void reconstruction_new();
    void save_curlmap(const std::string &filename, const int degree);
};

#endif // CURLCONTINUOUSRECONS_H