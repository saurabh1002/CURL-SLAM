//
// Created by zkc on 04/05/23.
//

#include "curl_slam/CurlPoseGraph.h"
#include "curl_slam/CurlTracking.h"
#include "curl_slam/load_config.h"
#include "curl_slam/types.h"
#include <memory>
#include <ros/ros.h>
#include <thread>

int main(int argc, char **argv) {
    if (argc == 1) {
        std::cerr << "Please provide config.yaml file!" << std::endl;
    }
    ros::init(argc, argv, "curl_odometry");
    ros::NodeHandle nh;
    // load parameters
    std::shared_ptr<DIRECT_METHOD_CONFIG> direct_method_config_ptr = std::make_shared<DIRECT_METHOD_CONFIG>();
    load_direct_method_config(argv[1], *direct_method_config_ptr);

    std::shared_ptr<CURL_VOXEL_MAPPING_CONFIG> curl_voxel_mapping_config_ptr =
        std::make_shared<CURL_VOXEL_MAPPING_CONFIG>();
    load_curl_voxel_mapping_config(argv[1], *curl_voxel_mapping_config_ptr);
    direct_method_config_ptr->minimum_img_rso =
        (curl_voxel_mapping_config_ptr->half_diag_cut_threshold * 2 / direct_method_config_ptr->minimum_rso) + 1;

    std::shared_ptr<DEBUG_CONFIG> debug_config_ptr = std::make_shared<DEBUG_CONFIG>();
    if (argc == 3) {
        load_debug_config(argv[1], *debug_config_ptr, argv[2]);
    } else {
        load_debug_config(argv[1], *debug_config_ptr);
    }
    debug_config_ptr->color_map.resize(5, 3);
    debug_config_ptr->color_map << 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1;
    ROS_INFO("Sequence: %s", debug_config_ptr->seq.c_str());
    std::shared_ptr<AABB_CONFIG> aabb_config_ptr = std::make_shared<AABB_CONFIG>();
    load_aabb_config(argv[1], *aabb_config_ptr);
    std::shared_ptr<CURL_TRACKING_CONFIG> curl_tracking_config_ptr = std::make_shared<CURL_TRACKING_CONFIG>();
    load_curl_tracking_config(argv[1], *curl_tracking_config_ptr);
    std::shared_ptr<CURL_LOOP_CLOSURE_CONFIG> curl_loop_closure_config_ptr =
        std::make_shared<CURL_LOOP_CLOSURE_CONFIG>();
    load_loop_closure_config(argv[1], *curl_loop_closure_config_ptr);
    // initilize the SPH table
    std::shared_ptr<SH_TABLE_CONFIG<BT>> SH_table_config_ptr = std::make_shared<SH_TABLE_CONFIG<BT>>();
    load_SH_table_config<BT>(argv[1], *SH_table_config_ptr);
    curl::getSH_table(*SH_table_config_ptr, direct_method_config_ptr->minimum_img_rso,
                      direct_method_config_ptr->minimum_img_rso,
                      curl_voxel_mapping_config_ptr->half_diag_cut_threshold);
    curl::get_update_idx(*SH_table_config_ptr);

    // initializing the voxel mapping (database)

    std::shared_ptr<CurlVoxelMapping<BT>> curl_voxel_mapping_ptr = std::make_shared<CurlVoxelMapping<BT>>(
        &nh, curl_voxel_mapping_config_ptr, curl_tracking_config_ptr, SH_table_config_ptr, direct_method_config_ptr,
        aabb_config_ptr, curl_loop_closure_config_ptr, debug_config_ptr);

    // initialize the pose graph
    std::shared_ptr<CurlPoseGraph<BT>> curl_pose_graph_ptr = std::make_shared<CurlPoseGraph<BT>>(
        &nh, curl_loop_closure_config_ptr, curl_tracking_config_ptr, curl_voxel_mapping_config_ptr, SH_table_config_ptr,
        debug_config_ptr, curl_voxel_mapping_ptr);
    // initlize the tracking
    std::shared_ptr<CurlTracking<BT>> curl_tracking_ptr =
        std::make_shared<CurlTracking<BT>>(&nh, curl_tracking_config_ptr, aabb_config_ptr, SH_table_config_ptr,
                                           debug_config_ptr, curl_voxel_mapping_ptr, curl_pose_graph_ptr);
    // initlize the multi-thread
    //    std::thread t_tf_publish(&TfPublisher::run, tf_publisher_ptr);
    std::thread t_tracking(&CurlTracking<BT>::run, curl_tracking_ptr);
    ros::spin();
    //    t_tf_publish.join();
    t_tracking.join();
    //     t_local_ba.join();
    // }
    return 0;
}
