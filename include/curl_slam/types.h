//
// Created by zkc on 07/02/23.
//
#ifndef CURL_SLAM_TYPES_H
#define CURL_SLAM_TYPES_H

#define PCL_NO_PRECOMPILE
#include "curl_slam/FileReader.h"
#include <pcl/io/pcd_io.h>
// #include <pcl/memory.h>
#include <pcl/pcl_macros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cstdint>

#define INVALID_IDX -1
struct PointXYZITL {
    PCL_ADD_POINT4D; // quad-word XYZ
    float intensity; ///< laser intensity reading
    uint32_t t;
    uint16_t label;                 ///< point label
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // ensure proper alignment
};

// Register custom point struct according to PCL
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZITL, (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
                                                   std::uint32_t, t, t)(uint16_t, label, label))

typedef PointXYZITL PointT;
typedef FileReader<float, std::size_t> FR;
typedef FR::Basic_Type BT;
typedef FR::Semantic_Type ST;
using PatchId = std::uint64_t;

#endif // CURL_SLAM_TYPES_H
