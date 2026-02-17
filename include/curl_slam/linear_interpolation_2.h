//
// Created by zkc on 14/03/23.
//

#ifndef CURL_SLAM_LINEAR_INTERPOLATION_2_H
#define CURL_SLAM_LINEAR_INTERPOLATION_2_H
#include "curl_slam/linear_interpolation_2_func.h"
#include "curl_slam/types.h"
#define DOUBLE_INVALID (std::numeric_limits<double>::max())
#define BT_INVALID (std::numeric_limits<BT>::max())

BT default_boundary[2] = {-BT_INVALID, BT_INVALID};
/**
 *
 * @tparam DerivedP
 * @param query_points
 * @param F
 * @param idx
 * @return the interpolation result and valid indices
 */
// template <typename DerivedP>
// std::pair<std::vector<double>, std::vector<int>>
// linear_interpolation_2(const Eigen::MatrixBase<DerivedP> &query_points, const interp_func_pair &F,
//                        const double boundary[2] = default_boundary,
//                        const std::vector<double> &value = std::vector<double>()) {
//     //    assert(query_points.cols() == 2);
//     //    std::cout << "linear interpolation" << std::endl;
//     Delaunay_triangulation_linear::Face_handle fh;
//     Delaunay_triangulation_linear::Face_handle fh_last_valid;
//     std::pair<std::vector<double>, std::vector<int>> interpolate_result;
//     for (int i = 0; i < query_points.rows(); ++i) {
//         if (!value.empty()) {
//             if (value[i] != DOUBLE_INVALID) {
//                 K_linear ::Point_2 p(query_points(i, 0), query_points(i, 1));
//                 std::vector<std::pair<Point_linear, Coord_type_linear>> coords;
//                 fh = F.first.locate(p, fh_last_valid);
//                 Coord_type_linear norm =
//                     CGAL::natural_neighbor_coordinates_2(F.first, p, std::back_inserter(coords), fh)
//                         .second;
//                 if (!coords.empty()) {
//                     Coord_type_linear res = CGAL::linear_interpolation(
//                         coords.begin(), coords.end(), norm, Value_access_linear(F.second));
//                     interpolate_result.first.push_back(res);
//                     if ((res >= boundary[0]) && (res <= boundary[1])) {
//                         interpolate_result.second.push_back(i);
//                     }
//                     fh_last_valid = fh;
//                 } else {
//                     interpolate_result.first.push_back(DOUBLE_INVALID);
//                 }
//             } else {
//                 interpolate_result.first.push_back(DOUBLE_INVALID);
//             }
//         } else {
//             K_linear::Point_2 p(query_points(i, 0), query_points(i, 1));
//             std::vector<std::pair<Point_linear, Coord_type_linear>> coords;
//             fh = F.first.locate(p, fh_last_valid);
//             Coord_type_linear norm =
//                 CGAL::natural_neighbor_coordinates_2(F.first, p, std::back_inserter(coords), fh)
//                     .second;
//             if (!coords.empty()) {
//                 Coord_type_linear res = CGAL::linear_interpolation(
//                     coords.begin(), coords.end(), norm, Value_access_linear(F.second));
//                 interpolate_result.first.push_back(res);
//                 if ((res >= boundary[0]) && (res <= boundary[1])) {
//                     interpolate_result.second.push_back(i);
//                 }
//                 fh_last_valid = fh;
//             } else {
//                 interpolate_result.first.push_back(DOUBLE_INVALID);
//             }
//         }
//     }
//
//     return interpolate_result;
// }

template <typename T>
bool is_nearby(const Delaunay_triangulation_linear &triangulation, K_linear::Point_2 &query_point,
               Delaunay_triangulation_linear::Face_handle &fh, T threshold) {
    fh = triangulation.locate(query_point, fh);
    if (fh == nullptr) {
        return false;
    }

    auto nearest_vertex = triangulation.nearest_vertex(query_point, fh);
    if (CGAL::squared_distance(nearest_vertex->point(), query_point) < threshold * threshold) {
        query_point = nearest_vertex->point();
    }

    return true;
}

template <typename T>
std::pair<std::vector<T>, std::vector<int>> linear_interpolation_2(const Eigen::MatrixX<T> &query_points,
                                                                   const interp_func_pair &F, const T boundary[2],
                                                                   const std::vector<T> &value = std::vector<T>()) {
    //    assert(query_points.cols() == 2);
    //    std::cout << "linear interpolation" << std::endl;
    const T T_INVALID = std::numeric_limits<T>::max();
    Delaunay_triangulation_linear::Face_handle fh;
    //    Delaunay_triangulation_linear::Face_handle fh_last_valid;
    std::pair<std::vector<T>, std::vector<int>> interpolate_result;
    interpolate_result.first.resize(query_points.rows(), T_INVALID);
    interpolate_result.second.reserve(query_points.rows());
    for (int i = 0; i < query_points.rows(); ++i) {
        if (!value.empty()) {
            if (value[i] != T_INVALID) {
                K_linear ::Point_2 p(query_points(i, 0), query_points(i, 1));
                std::vector<std::pair<Point_linear, Coord_type_linear>> coords;
                //                fh = F.first.locate(p, fh);
                is_nearby(F.first, p, fh, 0.001);
                Coord_type_linear norm =
                    CGAL::natural_neighbor_coordinates_2(F.first, p, std::back_inserter(coords), fh).second;
                if (!coords.empty()) {
                    Coord_type_linear res =
                        CGAL::linear_interpolation(coords.begin(), coords.end(), norm, Value_access_linear(F.second));
                    if ((res >= boundary[0]) && (res <= boundary[1])) {
                        interpolate_result.first[i] = res;
                        interpolate_result.second.push_back(i);
                    }
                    //                    fh_last_valid = fh;
                }
            }
        } else {
            K_linear::Point_2 p(query_points(i, 0), query_points(i, 1));
            std::vector<std::pair<Point_linear, Coord_type_linear>> coords;
            //            fh = F.first.locate(p, fh);
            is_nearby(F.first, p, fh, 0.001);
            Coord_type_linear norm =
                CGAL::natural_neighbor_coordinates_2(F.first, p, std::back_inserter(coords), fh).second;
            if (!coords.empty()) {
                Coord_type_linear res =
                    CGAL::linear_interpolation(coords.begin(), coords.end(), norm, Value_access_linear(F.second));
                if ((res >= boundary[0]) && (res <= boundary[1])) {
                    interpolate_result.first[i] = res;
                    interpolate_result.second.push_back(i);
                }
                //                fh_last_valid = fh;
            }
        }
    }

    return interpolate_result;
}

template <typename T> T get_max_length(Delaunay_triangulation_linear::Face_handle &fh) {
    if (fh == nullptr) {
        return std::numeric_limits<T>::max();
    }
    T max_length = 0;
    for (int i = 0; i < 3; ++i) {
        // check whether vertex is nullptr
        if (fh->vertex(i) == nullptr || fh->vertex((i + 1) % 3) == nullptr) {
            return std::numeric_limits<T>::max();
        }
        T length = CGAL::squared_distance(fh->vertex(i)->point(), fh->vertex((i + 1) % 3)->point());
        if (length > max_length) {
            max_length = length;
        }
    }
    return max_length;
}

// BUG: This function is wrong
template <typename T> T get_max_length_3D(Delaunay_triangulation_linear::Face_handle &fh) {
    if (fh == nullptr) {
        return std::numeric_limits<T>::max();
    }
    T max_length = 0;
    for (int i = 0; i < 3; ++i) {
        // check whether vertex is nullptr
        if (fh->vertex(i) == nullptr || fh->vertex((i + 1) % 3) == nullptr) {
            return std::numeric_limits<T>::max();
        }
        Eigen::Vector3<T> pt0(fh->vertex(i)->point().x(), fh->vertex(i)->point().y(), fh->vertex(i)->info());
        Eigen::Vector3<T> pt1(fh->vertex((i + 1) % 3)->point().x(), fh->vertex((i + 1) % 3)->point().y(),
                              fh->vertex((i + 1) % 3)->info());
        // T length = CGAL::squared_distance(fh->vertex(i)->point(), fh->vertex((i + 1) % 3)->point());
        T length = (pt0 - pt1).squaredNorm();
        if (length > max_length) {
            max_length = length;
        }
    }
    return max_length;
}

template <typename T>
std::pair<std::vector<T>, std::vector<int>>
linear_interpolation_2(const int start_idx, const int step, const Eigen::MatrixX<T> &query_points,
                       const interp_func_pair &F, const T boundary[2], const T length_threshold,
                       const std::vector<T> &value = std::vector<T>()) {
    //    assert(query_points.cols() == 2);
    const T T_INVALID = std::numeric_limits<T>::max();
    Delaunay_triangulation_linear::Face_handle fh;
    std::pair<std::vector<T>, std::vector<int>> interpolate_result;
    interpolate_result.first.resize(query_points.rows(), T_INVALID);
    for (int i = start_idx; i < query_points.rows(); i += step) {
        if (!value.empty()) {
            if (value[i] != T_INVALID) {
                K_linear ::Point_2 p(query_points(i, 0), query_points(i, 1));
                std::vector<std::pair<Point_linear, Coord_type_linear>> coords;
                fh = F.first.locate(p, fh);
                //                is_nearby(F.first, p, fh, 0.001);
                T max_length = get_max_length<T>(fh);
                //                std::cout << "1: " << max_length << std::endl;
                if (max_length < length_threshold) {
                    //                    std::cout << "2: " << max_length << std::endl;
                    Coord_type_linear norm =
                        CGAL::natural_neighbor_coordinates_2(F.first, p, std::back_inserter(coords), fh).second;
                    if (!coords.empty()) {
                        Coord_type_linear res = CGAL::linear_interpolation(coords.begin(), coords.end(), norm,
                                                                           Value_access_linear(F.second));
                        if ((res >= boundary[0]) && (res <= boundary[1])) {
                            interpolate_result.first[i] = res;
                            interpolate_result.second.push_back(i);
                        }
                        //                    fh_last_valid = fh;
                    }
                }
            }
        } else {
            K_linear::Point_2 p(query_points(i, 0), query_points(i, 1));
            std::vector<std::pair<Point_linear, Coord_type_linear>> coords;
            fh = F.first.locate(p, fh);
            //            is_nearby(F.first, p, fh, 0.001);
            T max_length = get_max_length<T>(fh);
            //            std::cout << "1: " << max_length << std::endl;
            if (max_length < length_threshold) {
                //                std::cout << "2: " << max_length << std::endl;
                Coord_type_linear norm =
                    CGAL::natural_neighbor_coordinates_2(F.first, p, std::back_inserter(coords), fh).second;
                if (!coords.empty()) {
                    Coord_type_linear res =
                        CGAL::linear_interpolation(coords.begin(), coords.end(), norm, Value_access_linear(F.second));
                    if ((res >= boundary[0]) && (res <= boundary[1])) {
                        interpolate_result.first[i] = res;
                        interpolate_result.second.push_back(i);
                    }
                }
            }
        }
    }
    return interpolate_result;
}

template <typename T>
std::pair<std::vector<T>, std::vector<int>>
linear_interpolation_2_with_idx(const Eigen::MatrixX<T> &query_points, const std::vector<int> &valid_idx,
                                const interp_func_pair &F, const T boundary[2], const T length_threshold) {
    //    assert(query_points.cols() == 2);
    const T T_INVALID = std::numeric_limits<T>::max();
    Delaunay_triangulation_linear::Face_handle fh;
    std::pair<std::vector<T>, std::vector<int>> interpolate_result;
    interpolate_result.first.resize(valid_idx.size(), T_INVALID);
    for (int i = 0; i < valid_idx.size(); ++i) {
        K_linear::Point_2 p(query_points(valid_idx[i], 0), query_points(valid_idx[i], 1));
        std::vector<std::pair<Point_linear, Coord_type_linear>> coords;
        fh = F.first.locate(p, fh);
        T max_length = get_max_length<T>(fh);
        if (max_length < length_threshold) {
            Coord_type_linear norm =
                CGAL::natural_neighbor_coordinates_2(F.first, p, std::back_inserter(coords), fh).second;
            if (!coords.empty()) {
                Coord_type_linear res =
                    CGAL::linear_interpolation(coords.begin(), coords.end(), norm, Value_access_linear(F.second));
                if ((res >= boundary[0]) && (res <= boundary[1])) {
                    interpolate_result.first[i] = res;
                    interpolate_result.second.push_back(i);
                }
            }
        }
    }
    return interpolate_result;
}

template <typename T>
std::pair<std::vector<T>, std::vector<int>>
linear_barycentric_interpolation_2(const Eigen::MatrixX<T> &query_points, const Eigen::VectorX<T> &value,
                                   const interp_func_pair &F, const T boundary[2], const T length_threshold) {
    //    assert(query_points.cols() == 2);
    const T T_INVALID = std::numeric_limits<T>::max();
    Delaunay_triangulation_linear::Face_handle fh;
    std::pair<std::vector<T>, std::vector<int>> interpolate_result;
    interpolate_result.first.resize(query_points.rows(), T_INVALID);
    interpolate_result.second.reserve(query_points.rows());
    for (int i = 0; i < query_points.rows(); ++i) {
        K_linear::Point_2 p(query_points(i, 0), query_points(i, 1));
        std::vector<std::pair<Point_linear, Coord_type_linear>> coords;
        fh = F.first.locate(p, fh);
        if (fh == nullptr) {
            continue;
        }
        T u, v, w;
        if (curl::barycentric_coordinate(fh->vertex(0)->point().x(), fh->vertex(0)->point().y(),
                                         fh->vertex(1)->point().x(), fh->vertex(1)->point().y(),
                                         fh->vertex(2)->point().x(), fh->vertex(2)->point().y(), p.x(), p.y(), u, v,
                                         w)) {
            T max_length = get_max_length<T>(fh);
            if (max_length < length_threshold && fh->vertex(0)->info() < value.size() &&
                fh->vertex(1)->info() < value.size() && fh->vertex(2)->info() < value.size()) {
                Coord_type_linear res = u * value(fh->vertex(0)->info()) + v * value(fh->vertex(1)->info()) +
                                        w * value(fh->vertex(2)->info());
                if ((res >= boundary[0]) && (res <= boundary[1])) {
                    interpolate_result.first[i] = res;
                    interpolate_result.second.push_back(i);
                }
            }
        }
    }
    return interpolate_result;
}

template <typename T>
std::vector<int> mask_identification(const Eigen::MatrixX<T> &query_points, const interp_func_pair &F,
                                     const T length_threshold) {
    //    assert(query_points.cols() == 2);
    const T T_INVALID = std::numeric_limits<T>::max();
    Delaunay_triangulation_linear::Face_handle fh;
    std::vector<int> valid_mask;
    valid_mask.reserve(query_points.rows());
    int counter1 = 0;
    int counter2 = 0;
    for (int i = 0; i < query_points.rows(); ++i) {
        K_linear::Point_2 p(query_points(i, 0), query_points(i, 1));
        std::vector<std::pair<Point_linear, Coord_type_linear>> coords;
        fh = F.first.locate(p, fh);
        if (fh == nullptr) {
            ++counter1;
            continue;
        }
        T u, v, w;
        // if (curl::barycentric_coordinate(fh->vertex(0)->point().x(), fh->vertex(0)->point().y(),
        //                                  fh->vertex(1)->point().x(), fh->vertex(1)->point().y(),
        //                                  fh->vertex(2)->point().x(), fh->vertex(2)->point().y(), p.x(), p.y(), u, v,
        //                                  w)) {
        T max_length = get_max_length<T>(fh);
        if (max_length < length_threshold) {
            valid_mask.push_back(i);
        } else {
            ++counter2;
        }
        // }
    }
    std::cout << "mask_identification: " << counter1 << " " << counter2 << std::endl;
    return valid_mask;
}
#endif // CURL_SLAM_LINEAR_INTERPOLATION_2_H
