#ifndef TRIANGLUATION_MASK_IDENTIFICATION_H
#define TRIANGLUATION_MASK_IDENTIFICATION_H
#include "curl_slam/types.h"
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Interpolation_traits_2.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/interpolation_functions.h>
#include <CGAL/natural_neighbor_coordinates_2.h>
#include <Eigen/Dense>

namespace curl {

struct FaceInfo {
    unsigned int idx;
    double l0;
    double l1;
    double l2;
    double area;
};

typedef CGAL::Exact_predicates_inexact_constructions_kernel K_linear;
typedef CGAL::Triangulation_vertex_base_with_info_2<unsigned int, K_linear> Vb;
typedef CGAL::Triangulation_face_base_with_info_2<FaceInfo, K_linear> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Delaunay_triangulation_2<K_linear, Tds> Delaunay_triangulation_linear;

Delaunay_triangulation_linear meshing_with_idx(const Eigen::VectorXd &x, const Eigen::VectorXd &y) {
    Delaunay_triangulation_linear mesh;
    std::vector<std::pair<K_linear::Point_2, unsigned int>> P;
    for (int i = 0; i < x.size(); ++i) {
        K_linear::Point_2 p(x(i), y(i));
        P.emplace_back(p, i);
    }
    mesh.insert(P.begin(), P.end());
    return mesh;
}

void calculate_lengths_areas_of_mesh(Delaunay_triangulation_linear &mesh, const Eigen::VectorXd &values,
                                     std::vector<double> &lengths, std::vector<double> &areas) {
    unsigned int face_idx = 0;
    for (auto fit = mesh.finite_faces_begin(); fit != mesh.finite_faces_end(); ++fit) {
        Eigen::Vector3d v0(fit->vertex(0)->point().x(), fit->vertex(0)->point().y(), values(fit->vertex(0)->info()));
        Eigen::Vector3d v1(fit->vertex(1)->point().x(), fit->vertex(1)->point().y(), values(fit->vertex(1)->info()));
        Eigen::Vector3d v2(fit->vertex(2)->point().x(), fit->vertex(2)->point().y(), values(fit->vertex(2)->info()));
        double a = (v0 - v1).norm();
        double b = (v1 - v2).norm();
        double c = (v2 - v0).norm();
        lengths.push_back(a);
        lengths.push_back(b);
        lengths.push_back(c);
        double s = (a + b + c) / 2.0;
        double area_square = s * (s - a) * (s - b) * (s - c);
        if (area_square < 0) {
            area_square = 0;
        }
        double area = std::sqrt(area_square);
        areas.push_back(area);
        fit->info().idx = face_idx++;
        fit->info().l0 = a;
        fit->info().l1 = b;
        fit->info().l2 = c;
        fit->info().area = area;
    }
}

typename Delaunay_triangulation_linear::Face_handle locate_point_in_mesh(
    const Delaunay_triangulation_linear &mesh, const K_linear::Point_2 &query_point,
    typename Delaunay_triangulation_linear::Face_handle initial_face = Delaunay_triangulation_linear::Face_handle()) {

    CGAL::Triangulation_2<K_linear, Tds>::Locate_type lt;
    int li;

    // Use the initial_face as a hint for the locate method
    typename Delaunay_triangulation_linear::Face_handle face_handle = mesh.locate(query_point, lt, li, initial_face);

    switch (lt) {
    case CGAL::Triangulation_2<K_linear, Tds>::FACE:
        break;
    case CGAL::Triangulation_2<K_linear, Tds>::EDGE:
        break;
    case CGAL::Triangulation_2<K_linear, Tds>::VERTEX:
        break;
    case CGAL::Triangulation_2<K_linear, Tds>::OUTSIDE_CONVEX_HULL:
        face_handle = nullptr; // Invalidate the face handle
        break;
    case CGAL::Triangulation_2<K_linear, Tds>::OUTSIDE_AFFINE_HULL:
        face_handle = nullptr;
        break;
    default:
        face_handle = nullptr;
        break;
    }

    return face_handle;
}

void mask_generation(const Delaunay_triangulation_linear &mesh, const Eigen::MatrixXd &query_points,
                     const double length_thres, const double area_thres, std::vector<int> &mask_idx_vec) {
    typename Delaunay_triangulation_linear::Face_handle initial_face = nullptr;
    for (int i = 0; i < query_points.rows(); ++i) {
        auto fh = locate_point_in_mesh(mesh, K_linear::Point_2(query_points(i, 0), query_points(i, 1)), initial_face);
        if (fh != nullptr) {
            if (fh->info().l0 < length_thres && fh->info().l1 < length_thres && fh->info().l2 < length_thres &&
                fh->info().area < area_thres) {
                mask_idx_vec.push_back(i);
            }
        }
        initial_face = fh;
    }
}

} // namespace curl

#endif // TRIANGLUATION_MASK_IDENTIFICATION_H