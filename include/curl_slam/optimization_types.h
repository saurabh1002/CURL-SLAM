//
// Created by zkc on 15/03/23.
//

#ifndef CURL_SLAM_OPTIMIZATION_TYPES_H
#define CURL_SLAM_OPTIMIZATION_TYPES_H

#include "curl_slam/FileReader.h"
#include "curl_slam/PatchProcession.h"
#include "curl_slam/RosHandler.h"
#include "curl_slam/curl_tools_light.h"
#include "curl_slam/linear_interpolation_2.h"
#include "curl_slam/linear_interpolation_2_func.h"
#include "curl_slam/load_config.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/autodiff_cost_function.h>
#include <ceres/autodiff_manifold.h>
#include <ceres/ceres.h>
#include <ceres/loss_function.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <sstream>
#include <thread>
#include <visualization_msgs/Marker.h>

inline void abort_if_sph_index_oob(int phi_idx, int theta_idx, int azi_rso, int elev_rso, double phi, double theta,
                                   const char *tag) {
    if (phi_idx < 0 || phi_idx >= azi_rso || theta_idx < 0 || theta_idx >= elev_rso) {
        std::cerr << "[CURL_SLAM] " << tag << " sph idx out of range: phi_idx=" << phi_idx
                  << " theta_idx=" << theta_idx << " azi_rso=" << azi_rso << " elev_rso=" << elev_rso
                  << " phi=" << phi << " theta=" << theta << std::endl;
        std::abort();
    }
}

struct Pose3d {
    Eigen::Vector3d p;
    Eigen::Quaterniond q;
    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T;
    void set_pose(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T) {
        p = _T.block<3, 1>(0, 3);
        q = Eigen::Quaterniond(_T.block<3, 3>(0, 0));
        q.normalize();
        Eigen::Isometry3d T_tmp = Eigen::Isometry3d::Identity();
        T_tmp.matrix()(Eigen::seq(0, 2), 3) = p;
        T_tmp.matrix()(Eigen::seq(0, 2), Eigen::seq(0, 2)) = q.toRotationMatrix();
        T = T_tmp.matrix();
    }
    void set_pose(const Eigen::Vector3d &_p, const Eigen::Quaterniond &_q) {
        p = _p;
        q = _q;
        Eigen::Isometry3d T_tmp = Eigen::Isometry3d::Identity();
        T_tmp.matrix()(Eigen::seq(0, 2), 3) = p;
        T_tmp.matrix()(Eigen::seq(0, 2), Eigen::seq(0, 2)) = q.toRotationMatrix();
        T = T_tmp.matrix();
    }

    Eigen::Isometry3d get_T() {
        Eigen::Isometry3d T_tmp = Eigen::Isometry3d::Identity();
        T_tmp.matrix()(Eigen::seq(0, 2), 3) = p;
        T_tmp.matrix()(Eigen::seq(0, 2), Eigen::seq(0, 2)) = q.toRotationMatrix();
        return T_tmp;
    }

    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> get_T_matrix() { return T; }

    double *get_T_matrix_data() { return T.data(); }

    double *get_p_data() { return p.data(); }

    double *get_q_data() { return q.coeffs().data(); }

    void update_T_from_pq() {
        Eigen::Isometry3d T_tmp = Eigen::Isometry3d::Identity();
        T_tmp.matrix()(Eigen::seq(0, 2), 3) = p;
        T_tmp.matrix()(Eigen::seq(0, 2), Eigen::seq(0, 2)) = q.toRotationMatrix();
        T = T_tmp.matrix();
    }

    void update_pq_from_T() {
        p = T.block<3, 1>(0, 3);
        q = Eigen::Quaterniond(T.block<3, 3>(0, 0));
        q.normalize();
    }
};

// TODO: modify optimization factor since conformal mapping Gamma is no longer needed

/******************* SO3 Manifold *******************/
class SO3Manifold {
  public:
    template <typename T> bool Plus(const T *x_SO3, const T *delta_so3, T *x_plus_delta) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 3>> x(x_SO3);
        Sophus::SO3<T> x_on_manifold(x);

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> delta(delta_so3);
        Sophus::SO3<T> delta_on_manifold = Sophus::SO3<T>::exp(delta);
        Sophus::SO3<T> x_hat = x_on_manifold * delta_on_manifold;
        Eigen::Matrix<T, 3, 3> x_hat_manifold = x_hat.matrix();
        Eigen::Map<Eigen::Matrix<T, 3, 3>> residuals_map(x_plus_delta);
        residuals_map.template block<3, 3>(0, 0) = x_hat_manifold;
        return true;
    }

    template <typename T> bool Minus(const T *y_SO3, const T *x_SO3, T *y_minus_x) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 3>> y(y_SO3);
        Eigen::Map<const Eigen::Matrix<T, 3, 3>> x(x_SO3);
        Sophus::SO3<T> y_on_manifold(y);
        Sophus::SO3<T> x_on_manifold(x);

        Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_map(y_minus_x);
        residuals_map = (x_on_manifold.inverse() * y_on_manifold).log();
        return true;
    }

    static ceres::Manifold *Create() { return new ceres::AutoDiffManifold<SO3Manifold, 9, 3>; }
};

class SE3Manifold : public ceres::Manifold {
  public:
    SE3Manifold() {}

    ~SE3Manifold() override {}

    int AmbientSize() const override { return 16; }

    int TangentSize() const override { return 6; }

    bool Plus(const double *x_SE3, const double *delta_se3, double *x_plus_delta) const override {
        Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> x(x_SE3);
        Sophus::SE3<double> x_on_manifold(x);
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> delta(delta_se3);
        Sophus::SE3<double> delta_on_manifold = Sophus::SE3<double>::exp(delta);
        Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> residuals_map(x_plus_delta);
        residuals_map = (x_on_manifold * delta_on_manifold).matrix();
        return true;
    }

    bool PlusJacobian(const double *x, double *jacobian) const override {
        Eigen::Map<Eigen::Matrix<double, 16, 6, Eigen::RowMajor>> J(jacobian);
        J.setIdentity();
        return true;
    }

    bool Minus(const double *y_SE3, const double *x_SE3, double *y_minus_x) const override {
        Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> y(y_SE3);
        Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> x(x_SE3);
        Sophus::SE3<double> y_on_manifold(y);
        Sophus::SE3<double> x_on_manifold(x);

        Eigen::Map<Eigen::Matrix<double, 6, 1>> residuals_map(y_minus_x);
        residuals_map = (x_on_manifold.inverse() * y_on_manifold).log();
        return true;
    }

    bool MinusJacobian(const double *x, double *jacobian) const override {
        Eigen::Map<Eigen::Matrix<double, 6, 16, Eigen::RowMajor>> J(jacobian);
        J.setIdentity();
        return true;
    }

    static ceres::Manifold *Create() { return new SE3Manifold(); }
};

class SE3ManifoldXYYall : public ceres::Manifold {
  public:
    SE3ManifoldXYYall() {}

    ~SE3ManifoldXYYall() override {}

    int AmbientSize() const override { return 16; }

    int TangentSize() const override { return 6; }

    bool Plus(const double *x_SE3, const double *delta_se3, double *x_plus_delta) const override {
        Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> x_on_manifold(x_SE3);
        //        Eigen::Vector3d euler_original =
        //            Eigen::Matrix3d(x_on_manifold(Eigen::seq(0, 2), Eigen::seq(0, 2))).eulerAngles(0, 1, 2);
        //        Eigen::Matrix<double, 6, 1> x_se3 = Sophus::SE3<double>(x_on_manifold).log();
        //        Eigen::Map<const Eigen::Matrix<double, 6, 1>> delta(delta_se3);
        //        Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> update_x_SE3(x_plus_delta);
        //        update_x_SE3 = x_on_manifold * Sophus::SE3<double>::exp(delta).matrix();
        //        Eigen::Vector3d euler_update =
        //            Eigen::Matrix3d(update_x_SE3(Eigen::seq(0, 2), Eigen::seq(0, 2))).eulerAngles(0, 1, 2);
        //        euler_update(0) = euler_original(0);
        //        euler_update(1) = euler_original(1);
        //        update_x_SE3(Eigen::seq(0, 2), Eigen::seq(0, 2)) =
        //            (Eigen::AngleAxisd(euler_update(0), Eigen::Vector3d::UnitX()) *
        //             Eigen::AngleAxisd(euler_update(1), Eigen::Vector3d::UnitY()) *
        //             Eigen::AngleAxisd(euler_update(2), Eigen::Vector3d::UnitZ()))
        //                .matrix();
        //        update_x_SE3(2, 3) = x_on_manifold(2, 3);
        Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> x(x_SE3);
        Sophus::SE3<double> x_on_manifold_XYYall(x);
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> delta(delta_se3);
        Eigen::Matrix<double, 6, 1> delta_XYYall = delta;
        delta_XYYall(3) = 0;
        delta_XYYall(4) = 0;
        Sophus::SE3<double> delta_on_manifold = Sophus::SE3<double>::exp(delta_XYYall);
        Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> residuals_map(x_plus_delta);
        residuals_map = (x_on_manifold_XYYall * delta_on_manifold).matrix();
        residuals_map(2, 3) = x_on_manifold_XYYall.matrix()(2, 3);
        return true;
    }

    bool PlusJacobian(const double *x, double *jacobian) const override {
        Eigen::Map<Eigen::Matrix<double, 16, 6, Eigen::RowMajor>> J(jacobian);
        J.setIdentity();
        return true;
    }

    bool Minus(const double *y_SE3, const double *x_SE3, double *y_minus_x) const override {
        Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> y(y_SE3);
        Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> x(x_SE3);
        Sophus::SE3<double> y_on_manifold(y);
        Sophus::SE3<double> x_on_manifold(x);

        Eigen::Map<Eigen::Matrix<double, 6, 1>> residuals_map(y_minus_x);
        residuals_map = (x_on_manifold.inverse() * y_on_manifold).log();
        return true;
    }

    bool MinusJacobian(const double *x, double *jacobian) const override {
        Eigen::Map<Eigen::Matrix<double, 6, 16, Eigen::RowMajor>> J(jacobian);
        J.setIdentity();
        return true;
    }

    static ceres::Manifold *Create() { return new SE3ManifoldXYYall(); }
};

class SE3ManifoldZRollPitch : public ceres::Manifold {
  public:
    SE3ManifoldZRollPitch() {}

    ~SE3ManifoldZRollPitch() override {}

    int AmbientSize() const override { return 16; }

    int TangentSize() const override { return 6; }

    bool Plus(const double *x_SE3, const double *delta_se3, double *x_plus_delta) const override {
        Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> x_on_manifold(x_SE3);
        //        Eigen::Vector3d euler_original =
        //            Eigen::Matrix3d(x_on_manifold(Eigen::seq(0, 2), Eigen::seq(0, 2))).eulerAngles(0, 1, 2);
        //        Eigen::Matrix<double, 6, 1> x_se3 = Sophus::SE3<double>(x_on_manifold).log();
        //        Eigen::Map<const Eigen::Matrix<double, 6, 1>> delta(delta_se3);
        //        Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> update_x_SE3(x_plus_delta);
        //        update_x_SE3 = x_on_manifold * Sophus::SE3<double>::exp(delta).matrix();
        //        Eigen::Vector3d euler_update =
        //            Eigen::Matrix3d(update_x_SE3(Eigen::seq(0, 2), Eigen::seq(0, 2))).eulerAngles(0, 1, 2);
        //        euler_update(2) = euler_original(2);
        //        update_x_SE3(Eigen::seq(0, 2), Eigen::seq(0, 2)) =
        //            (Eigen::AngleAxisd(euler_update(0), Eigen::Vector3d::UnitX()) *
        //             Eigen::AngleAxisd(euler_update(1), Eigen::Vector3d::UnitY()) *
        //             Eigen::AngleAxisd(euler_update(2), Eigen::Vector3d::UnitZ()))
        //                .matrix();
        //        update_x_SE3(0, 3) = x_on_manifold(0, 3);
        //        update_x_SE3(1, 3) = x_on_manifold(1, 3);
        Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> x(x_SE3);
        Sophus::SE3<double> x_on_manifold_ZRollPitch(x);
        Eigen::Vector3d euler_original =
            Eigen::Matrix3d(x_on_manifold_ZRollPitch.matrix()(Eigen::seq(0, 2), Eigen::seq(0, 2))).eulerAngles(0, 1, 2);
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> delta(delta_se3);
        Eigen::Matrix<double, 6, 1> delta_ZRollPitch = delta;
        delta_ZRollPitch(5) = 0;
        Sophus::SE3<double> delta_on_manifold = Sophus::SE3<double>::exp(delta_ZRollPitch);
        Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> residuals_map(x_plus_delta);
        residuals_map = (x_on_manifold_ZRollPitch * delta_on_manifold).matrix();
        residuals_map(0, 3) = x_on_manifold_ZRollPitch.matrix()(0, 3);
        residuals_map(1, 3) = x_on_manifold_ZRollPitch.matrix()(1, 3);
        return true;
    }

    bool PlusJacobian(const double *x, double *jacobian) const override {
        Eigen::Map<Eigen::Matrix<double, 16, 6, Eigen::RowMajor>> J(jacobian);
        J.setIdentity();
        return true;
    }

    bool Minus(const double *y_SE3, const double *x_SE3, double *y_minus_x) const override {
        Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> y(y_SE3);
        Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> x(x_SE3);
        Sophus::SE3<double> y_on_manifold(y);
        Sophus::SE3<double> x_on_manifold(x);

        Eigen::Map<Eigen::Matrix<double, 6, 1>> residuals_map(y_minus_x);
        residuals_map = (x_on_manifold.inverse() * y_on_manifold).log();
        return true;
    }

    bool MinusJacobian(const double *x, double *jacobian) const override {
        Eigen::Map<Eigen::Matrix<double, 6, 16, Eigen::RowMajor>> J(jacobian);
        J.setIdentity();
        return true;
    }

    static ceres::Manifold *Create() { return new SE3ManifoldZRollPitch(); }
};

class SE3ManifoldZPitch : public ceres::Manifold {
  public:
    SE3ManifoldZPitch() {}

    ~SE3ManifoldZPitch() override {}

    int AmbientSize() const override { return 16; }

    int TangentSize() const override { return 2; }

    bool Plus(const double *x_SE3, const double *delta_se3, double *x_plus_delta) const override {
        // Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> x_on_manifold(x_SE3);
        // Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> x(x_SE3);
        // Sophus::SE3<double> x_on_manifold_ZPitch(x);
        // Eigen::Map<const Eigen::Matrix<double, 6, 1>> delta(delta_se3);
        // Eigen::Matrix<double, 6, 1> delta_ZPitch = delta;
        // delta_ZPitch(3) = 0;
        // delta_ZPitch(5) = 0;
        // Sophus::SE3<double> delta_on_manifold = Sophus::SE3<double>::exp(delta_ZPitch);
        // Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> residuals_map(x_plus_delta);
        // residuals_map = (x_on_manifold_ZPitch * delta_on_manifold).matrix();
        // residuals_map(0, 3) = x_on_manifold_ZPitch.matrix()(0, 3);
        // residuals_map(1, 3) = x_on_manifold_ZPitch.matrix()(1, 3);
        // return true;

        Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> x_on_manifold(x_SE3);
        Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> x(x_SE3);
        Sophus::SE3<double> x_on_manifold_ZPitch(x);
        Eigen::Map<const Eigen::Matrix<double, 2, 1>> delta(delta_se3);
        Eigen::Matrix<double, 6, 1> delta_ZPitch;
        delta_ZPitch.setZero();
        delta_ZPitch(2) = delta(0);
        delta_ZPitch(4) = delta(1);
        Sophus::SE3<double> delta_on_manifold = Sophus::SE3<double>::exp(delta_ZPitch);
        Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> residuals_map(x_plus_delta);
        residuals_map = (x_on_manifold_ZPitch * delta_on_manifold).matrix();
        return true;
    }

    bool PlusJacobian(const double *x, double *jacobian) const override {
        Eigen::Map<Eigen::Matrix<double, 16, 2, Eigen::RowMajor>> J(jacobian);
        J.setIdentity();
        return true;
    }

    bool Minus(const double *y_SE3, const double *x_SE3, double *y_minus_x) const override {
        Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> y(y_SE3);
        Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> x(x_SE3);
        Sophus::SE3<double> y_on_manifold(y);
        Sophus::SE3<double> x_on_manifold(x);

        Eigen::Matrix<double, 6, 1> tangent_space = (x_on_manifold.inverse() * y_on_manifold).log();
        Eigen::Map<Eigen::Matrix<double, 2, 1>> residuals_map(y_minus_x);
        residuals_map(0) = tangent_space(2);
        residuals_map(1) = tangent_space(4);
        return true;
    }

    bool MinusJacobian(const double *x, double *jacobian) const override {
        Eigen::Map<Eigen::Matrix<double, 2, 16, Eigen::RowMajor>> J(jacobian);
        J.setIdentity();
        return true;
    }

    static ceres::Manifold *Create() { return new SE3ManifoldZPitch(); }
};

class se3Manifold : public ceres::Manifold {
  public:
    se3Manifold() {}

    ~se3Manifold() override {}

    int AmbientSize() const override { return 6; }

    int TangentSize() const override { return 6; }

    bool Plus(const double *x_se3, const double *delta_se3, double *x_plus_delta) const override {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> x(x_se3);
        Sophus::SE3<double> x_on_manifold = Sophus::SE3<double>::exp(x);
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> delta(delta_se3);
        Sophus::SE3<double> delta_on_manifold = Sophus::SE3<double>::exp(delta);
        Eigen::Map<Eigen::Matrix<double, 6, 1>> residuals_map(x_plus_delta);
        residuals_map = (x_on_manifold * delta_on_manifold).log().matrix();
        return true;
    }

    bool PlusJacobian(const double *x, double *jacobian) const override {
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> J(jacobian);
        J.setIdentity();
        return true;
    }

    bool Minus(const double *y_se3, const double *x_se3, double *y_minus_x) const override {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> y(y_se3);
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> x(x_se3);
        Sophus::SE3<double> y_on_manifold = Sophus::SE3<double>::exp(y);
        Sophus::SE3<double> x_on_manifold = Sophus::SE3<double>::exp(x);

        Eigen::Map<Eigen::Matrix<double, 6, 1>> residuals_map(y_minus_x);
        residuals_map = (x_on_manifold.inverse() * y_on_manifold).log();
        return true;
    }

    bool MinusJacobian(const double *x, double *jacobian) const override {
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> J(jacobian);
        J.setIdentity();
        return true;
    }

    static ceres::Manifold *Create() { return new se3Manifold(); }
};

class se3ManifoldXYYall : public ceres::Manifold {
  public:
    se3ManifoldXYYall() {}

    ~se3ManifoldXYYall() override {}

    int AmbientSize() const override { return 6; }

    int TangentSize() const override { return 6; }

    bool Plus(const double *x_se3, const double *delta_se3, double *x_plus_delta) const override {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> x(x_se3);
        Sophus::SE3<double> x_on_manifold = Sophus::SE3<double>::exp(x);
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> delta(delta_se3);
        Sophus::SE3<double> delta_on_manifold = Sophus::SE3<double>::exp(delta);
        Eigen::Map<Eigen::Matrix<double, 6, 1>> residuals_map(x_plus_delta);
        residuals_map = (x_on_manifold * delta_on_manifold).log().matrix();
        residuals_map(2) = x(2);
        residuals_map(3) = x(3);
        residuals_map(4) = x(4);
        return true;
    }

    bool PlusJacobian(const double *x, double *jacobian) const override {
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> J(jacobian);
        J.setIdentity();
        return true;
    }

    bool Minus(const double *y_se3, const double *x_se3, double *y_minus_x) const override {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> y(y_se3);
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> x(x_se3);
        Sophus::SE3<double> y_on_manifold = Sophus::SE3<double>::exp(y);
        Sophus::SE3<double> x_on_manifold = Sophus::SE3<double>::exp(x);

        Eigen::Map<Eigen::Matrix<double, 6, 1>> residuals_map(y_minus_x);
        residuals_map = (x_on_manifold.inverse() * y_on_manifold).log();
        return true;
    }

    bool MinusJacobian(const double *x, double *jacobian) const override {
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> J(jacobian);
        J.setIdentity();
        return true;
    }

    static ceres::Manifold *Create() { return new se3ManifoldXYYall(); }
};

class se3ManifoldZRollPitch : public ceres::Manifold {
  public:
    se3ManifoldZRollPitch() {}

    ~se3ManifoldZRollPitch() override {}

    int AmbientSize() const override { return 6; }

    int TangentSize() const override { return 6; }

    bool Plus(const double *x_se3, const double *delta_se3, double *x_plus_delta) const override {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> x(x_se3);
        Sophus::SE3<double> x_on_manifold = Sophus::SE3<double>::exp(x);
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> delta(delta_se3);
        Sophus::SE3<double> delta_on_manifold = Sophus::SE3<double>::exp(delta);
        Eigen::Map<Eigen::Matrix<double, 6, 1>> residuals_map(x_plus_delta);
        residuals_map = (x_on_manifold * delta_on_manifold).log().matrix();
        residuals_map(0) = x(0);
        residuals_map(1) = x(1);
        residuals_map(5) = x(5);
        return true;
    }

    bool PlusJacobian(const double *x, double *jacobian) const override {
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> J(jacobian);
        J.setIdentity();
        return true;
    }

    bool Minus(const double *y_se3, const double *x_se3, double *y_minus_x) const override {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> y(y_se3);
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> x(x_se3);
        Sophus::SE3<double> y_on_manifold = Sophus::SE3<double>::exp(y);
        Sophus::SE3<double> x_on_manifold = Sophus::SE3<double>::exp(x);

        Eigen::Map<Eigen::Matrix<double, 6, 1>> residuals_map(y_minus_x);
        residuals_map = (x_on_manifold.inverse() * y_on_manifold).log();
        return true;
    }

    bool MinusJacobian(const double *x, double *jacobian) const override {
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> J(jacobian);
        J.setIdentity();
        return true;
    }

    static ceres::Manifold *Create() { return new se3ManifoldZRollPitch(); }
};

/******************* Odometry Factor *******************/
class StopCallback : public ceres::IterationCallback {
  public:
    StopCallback() = default;
    ~StopCallback() override = default;
    ceres::CallbackReturnType operator()(const ceres::IterationSummary &summary) override {
        if (summary.cost_change < 0) {
            return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
        }
        return ceres::SOLVER_CONTINUE;
    }
};

template <typename T> class CoupleFixSphDirectFactor : public ceres::SizedCostFunction<1, 16> {
  public:
    CoupleFixSphDirectFactor(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_o_w, const Eigen::Vector3d &_p_j,
                             std::shared_ptr<PatchProcession<T>> _patch_processtion_ptr,
                             std::shared_ptr<SH_TABLE_CONFIG<T>> _SH_table_config_ptr)
        : T_o_w(_T_o_w), p_j(_p_j), patch_procession_ptr(_patch_processtion_ptr),
          SH_table_config_ptr(_SH_table_config_ptr) {
        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = T_o_w(Eigen::seq(0, 2), Eigen::seq(0, 2));
        T_o_w(Eigen::seq(0, 2), Eigen::seq(0, 2)) = Eigen::Quaterniond(R).normalized().toRotationMatrix();
        invalid_res = patch_procession_ptr->T_INVALID;
        weight = 1;
        pyramid_idx = patch_procession_ptr->get_pyramid_depth();
        sph_coeff = patch_procession_ptr->get_sph_coeff();
        degree = patch_procession_ptr->get_SH_degree();
    }

    CoupleFixSphDirectFactor(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_o_w, const Eigen::Vector3d &_p_j,
                             std::shared_ptr<PatchProcession<T>> _patch_processtion_ptr,
                             std::shared_ptr<SH_TABLE_CONFIG<T>> _SH_table_config_ptr, double _invalid_res)
        : T_o_w(_T_o_w), p_j(_p_j), patch_procession_ptr(_patch_processtion_ptr),
          SH_table_config_ptr(_SH_table_config_ptr), invalid_res(_invalid_res) {
        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = T_o_w(Eigen::seq(0, 2), Eigen::seq(0, 2));
        T_o_w(Eigen::seq(0, 2), Eigen::seq(0, 2)) = Eigen::Quaterniond(R).normalized().toRotationMatrix();
        weight = 1;
        pyramid_idx = patch_procession_ptr->get_pyramid_depth();
        sph_coeff = patch_procession_ptr->get_sph_coeff();
        degree = patch_procession_ptr->get_SH_degree();
    }

    CoupleFixSphDirectFactor(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_o_w, const Eigen::Vector3d &_p_j,
                             std::shared_ptr<PatchProcession<T>> _patch_processtion_ptr,
                             std::shared_ptr<SH_TABLE_CONFIG<T>> _SH_table_config_ptr, double _invalid_res,
                             double _weight)
        : T_o_w(_T_o_w), p_j(_p_j), patch_procession_ptr(_patch_processtion_ptr),
          SH_table_config_ptr(_SH_table_config_ptr), invalid_res(_invalid_res), weight(_weight) {
        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = T_o_w(Eigen::seq(0, 2), Eigen::seq(0, 2));
        T_o_w(Eigen::seq(0, 2), Eigen::seq(0, 2)) = Eigen::Quaterniond(R).normalized().toRotationMatrix();
        pyramid_idx = patch_procession_ptr->get_pyramid_depth();
        sph_coeff = patch_procession_ptr->get_sph_coeff();
        degree = patch_procession_ptr->get_SH_degree();
    }

    ~CoupleFixSphDirectFactor() override {}
    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override {
        Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> T_w_j(parameters[0]);
        Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_o_j = T_o_w * T_w_j;
        Eigen::Vector3d p_o = T_o_j(Eigen::seq(0, 2), Eigen::seq(0, 2)) * p_j + T_o_j(Eigen::seq(0, 2), 3);

        bool is_fault = false;
        double I;
        if (SH_table_config_ptr->is_SH_analytic_jacobian) {
            I = patch_procession_ptr->get_recons_height(sph_coeff, p_o(0), p_o(1));
        } else {
            I = patch_procession_ptr->fetch_I_without_conformal_mapping(PatchProcession<T>::MatrixType::I, p_o(0),
                                                                        p_o(1), pyramid_idx);
        }
        double res = (p_o(2) - I);
        double interpolation_weight = 1;
        if (I == patch_procession_ptr->T_INVALID) {
            is_fault = true;
            residuals[0] = 0;
        } else if (std::abs(res) > invalid_res) {
            is_fault = true;
            residuals[0] = 0;
        } else {
            //            interpolation_weight = patch_procession_ptr->fetch_weights(p_o(0), p_o(1));
            residuals[0] = interpolation_weight * res;
        }
        // Compute the Jacobian if asked for.
        if (jacobians != nullptr && jacobians[0] != nullptr) {
            // Map jacobians
            Eigen::Map<Eigen::Matrix<double, 1, 16, Eigen::RowMajor>> Jac(jacobians[0], 1, 16);
            Jac.setZero();
            if (is_fault) {
                return true;
            }
            // SE3_Jac
            Eigen::Matrix<double, 3, 6> SE3_Jac;
            SE3_Jac(Eigen::seq(0, 2), Eigen::seq(0, 2)) = T_o_j(Eigen::seq(0, 2), Eigen::seq(0, 2));
            SE3_Jac(Eigen::seq(0, 2), Eigen::seq(3, 5)) =
                -T_o_j(Eigen::seq(0, 2), Eigen::seq(0, 2)) * curl::skew<double>(p_j);
            // I_Jac
            Eigen::Matrix<double, 1, 2, Eigen::RowMajor> I_Jac;
            I_Jac.setZero();
            if (SH_table_config_ptr->is_SH_analytic_jacobian) {
                // check whether the current point is on the edge of the valid region
                const std::pair<bool, bool> is_on_edge = patch_procession_ptr->check_is_on_edge_low_RAM(p_o(0), p_o(1));
                // const std::pair<bool, bool> is_on_edge = {false, false};
                double phi = patch_procession_ptr->get_phi(p_o(0));
                double theta = patch_procession_ptr->get_theta(p_o(1));
                int phi_idx = std::round((phi - SH_table_config_ptr->get_azi_low()) /
                                         (SH_table_config_ptr->get_azi_high() - SH_table_config_ptr->get_azi_low()) *
                                         (SH_table_config_ptr->azi_rso - 1));
                int theta_idx =
                    std::round((theta - SH_table_config_ptr->get_elev_low()) /
                               (SH_table_config_ptr->get_elev_high() - SH_table_config_ptr->get_elev_low()) *
                               (SH_table_config_ptr->elev_rso - 1));
                abort_if_sph_index_oob(phi_idx, theta_idx, SH_table_config_ptr->azi_rso, SH_table_config_ptr->elev_rso,
                                       phi, theta, "CoupleFixSphDirectFactor");
                Eigen::Matrix<double, 1, 2> SPH_Jac;
                SPH_Jac.setZero();
                for (int l = 0; l <= degree; ++l) {
                    for (int m = -l; m <= l; ++m) {
                        if (!is_on_edge.first) {
                            SPH_Jac(0, 0) = SH_table_config_ptr->SH_G_phi_table(
                                phi_idx * SH_table_config_ptr->elev_rso + theta_idx, l * l + l + m);
                        }
                        if (!is_on_edge.second) {
                            SPH_Jac(0, 1) = SH_table_config_ptr->SH_G_theta_table(
                                phi_idx * SH_table_config_ptr->elev_rso + theta_idx, l * l + l + m);
                        }
                        I_Jac += sph_coeff(l * l + l + m) * SPH_Jac;
                    }
                }
                I_Jac(0, 0) = patch_procession_ptr->get_G_phi() * I_Jac(0, 0);
                I_Jac(0, 1) = patch_procession_ptr->get_G_theta() * I_Jac(0, 1);
            } else {
                // I_Jac
                I_Jac(0, 0) = patch_procession_ptr->fetch_I_without_conformal_mapping(
                    PatchProcession<T>::MatrixType::Gx, p_o(0), p_o(1), pyramid_idx);
                //            I_Jac(0, 0) = patch_procession_ptr->fetch_I_Gx_interp(x, y);
                if (I_Jac(0, 0) == patch_procession_ptr->T_INVALID) {
                    I_Jac(0, 0) = 0;
                    // return true;
                }
                I_Jac(0, 1) = patch_procession_ptr->fetch_I_without_conformal_mapping(
                    PatchProcession<T>::MatrixType::Gy, p_o(0), p_o(1), pyramid_idx);
                //            I_Jac(0, 1) = patch_procession_ptr->fetch_I_Gy_interp(x, y);
                if (I_Jac(0, 1) == patch_procession_ptr->T_INVALID) {
                    I_Jac(0, 1) = 0;
                    // return true;
                }
            }
            // XY_Jac
            Eigen::Matrix<double, 2, 3, Eigen::RowMajor> XY_Jac;
            XY_Jac << 1, 0, 0, 0, 1, 0;
            // Jac equation
            Jac(0, Eigen::seq(0, 5)) =
                interpolation_weight * (SH_table_config_ptr->Z_Jac * SE3_Jac - I_Jac * XY_Jac * SE3_Jac);
        }
        return true;
    }

    int get_pyramid_idx() const { return pyramid_idx; }

    void set_pyramid_idx(const int _pyramid_idx) {
        if (_pyramid_idx >= 0) {
            pyramid_idx = _pyramid_idx;
        } else {
            pyramid_idx = 0;
        }
    }

    void update_pyramid_idx() {
        --pyramid_idx;
        if (pyramid_idx < 0) {
            pyramid_idx = 0;
        }
    }

    static ceres::CostFunction *Create(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_o_w,
                                       const Eigen::Vector3d &_p_j,
                                       std::shared_ptr<PatchProcession<T>> _patch_processtion_ptr,
                                       std::shared_ptr<SH_TABLE_CONFIG<T>> _SH_table_config_ptr) {
        return (new CoupleFixSphDirectFactor(_T_o_w, _p_j, _patch_processtion_ptr, _SH_table_config_ptr));
    }

    static ceres::CostFunction *Create(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_o_w,
                                       const Eigen::Vector3d &_p_j,
                                       std::shared_ptr<PatchProcession<T>> _patch_processtion_ptr,
                                       std::shared_ptr<SH_TABLE_CONFIG<T>> _SH_table_config_ptr, double _invalid_res) {
        return (new CoupleFixSphDirectFactor(_T_o_w, _p_j, _patch_processtion_ptr, _SH_table_config_ptr, _invalid_res));
    }

    static ceres::CostFunction *Create(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_o_w,
                                       const Eigen::Vector3d &_p_j,
                                       std::shared_ptr<PatchProcession<T>> _patch_processtion_ptr,
                                       std::shared_ptr<SH_TABLE_CONFIG<T>> _SH_table_config_ptr, double _invalid_res,
                                       double _weight) {
        return (new CoupleFixSphDirectFactor(_T_o_w, _p_j, _patch_processtion_ptr, _SH_table_config_ptr, _invalid_res,
                                             _weight));
    }

  private:
    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_o_w;
    Eigen::Vector3d p_j;
    std::shared_ptr<PatchProcession<T>> patch_procession_ptr;
    double invalid_res;
    double weight;
    int pyramid_idx;
    Eigen::VectorXd sph_coeff;
    std::shared_ptr<SH_TABLE_CONFIG<T>> SH_table_config_ptr;
    int degree;
};

template <typename T> class CoupleFixSphDirectZPitchFactor : public ceres::SizedCostFunction<1, 16> {
  public:
    CoupleFixSphDirectZPitchFactor(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_o_w,
                                   const Eigen::Vector3d &_p_j, PatchProcession<T> *_patch_processtion_ptr)
        : T_o_w(_T_o_w), p_j(_p_j), patch_procession_ptr(_patch_processtion_ptr) {
        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = T_o_w(Eigen::seq(0, 2), Eigen::seq(0, 2));
        T_o_w(Eigen::seq(0, 2), Eigen::seq(0, 2)) = Eigen::Quaterniond(R).normalized().toRotationMatrix();
        invalid_res = patch_procession_ptr->T_INVALID;
        weight = 1;
        pyramid_idx = patch_procession_ptr->get_pyramid_depth();
    }

    CoupleFixSphDirectZPitchFactor(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_o_w,
                                   const Eigen::Vector3d &_p_j, PatchProcession<T> *_patch_processtion_ptr,
                                   double _invalid_res)
        : T_o_w(_T_o_w), p_j(_p_j), patch_procession_ptr(_patch_processtion_ptr), invalid_res(_invalid_res) {
        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = T_o_w(Eigen::seq(0, 2), Eigen::seq(0, 2));
        T_o_w(Eigen::seq(0, 2), Eigen::seq(0, 2)) = Eigen::Quaterniond(R).normalized().toRotationMatrix();
        weight = 1;
        pyramid_idx = patch_procession_ptr->get_pyramid_depth();
    }

    CoupleFixSphDirectZPitchFactor(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_o_w,
                                   const Eigen::Vector3d &_p_j, PatchProcession<T> *_patch_processtion_ptr,
                                   double _invalid_res, double _weight)
        : T_o_w(_T_o_w), p_j(_p_j), patch_procession_ptr(_patch_processtion_ptr), invalid_res(_invalid_res),
          weight(_weight) {
        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = T_o_w(Eigen::seq(0, 2), Eigen::seq(0, 2));
        T_o_w(Eigen::seq(0, 2), Eigen::seq(0, 2)) = Eigen::Quaterniond(R).normalized().toRotationMatrix();
        pyramid_idx = patch_procession_ptr->get_pyramid_depth();
    }

    ~CoupleFixSphDirectZPitchFactor() override {}
    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override {
        Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> T_w_j(parameters[0]);
        Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_o_j = T_o_w * T_w_j;
        Eigen::Vector3d p_o = T_o_j(Eigen::seq(0, 2), Eigen::seq(0, 2)) * p_j + T_o_j(Eigen::seq(0, 2), 3);

        bool is_fault = false;
        double I = patch_procession_ptr->fetch_I_without_conformal_mapping(PatchProcession<T>::MatrixType::I, p_o(0),
                                                                           p_o(1), pyramid_idx);
        double res = (p_o(2) - I);
        if (I == patch_procession_ptr->T_INVALID) {
            is_fault = true;
            residuals[0] = 0;
        } else if (std::abs(res) > invalid_res) {
            is_fault = true;
            residuals[0] = 0;
        } else {
            residuals[0] = weight * res;
        }
        // Compute the Jacobian if asked for.
        if (jacobians != nullptr && jacobians[0] != nullptr) {
            // Map jacobians
            Eigen::Map<Eigen::Matrix<double, 1, 16, Eigen::RowMajor>> Jac(jacobians[0], 1, 16);
            Jac.setZero();
            if (is_fault) {
                return true;
            }
            // Z_jac
            Eigen::Matrix<double, 1, 3, Eigen::RowMajor> Z_Jac;
            Z_Jac << 0, 0, 1;
            // SE3_Jac
            Eigen::Matrix<double, 3, 6> SE3_Jac;
            SE3_Jac(Eigen::seq(0, 2), Eigen::seq(0, 2)) = T_o_j(Eigen::seq(0, 2), Eigen::seq(0, 2));
            SE3_Jac(Eigen::seq(0, 2), Eigen::seq(3, 5)) =
                -T_o_j(Eigen::seq(0, 2), Eigen::seq(0, 2)) * curl::skew<double>(p_j);
            // I_Jac
            Eigen::Matrix<double, 1, 2, Eigen::RowMajor> I_Jac;
            I_Jac(0, 0) = patch_procession_ptr->fetch_I_without_conformal_mapping(PatchProcession<T>::MatrixType::Gx,
                                                                                  p_o(0), p_o(1), pyramid_idx);
            //            I_Jac(0, 0) = patch_procession_ptr->fetch_I_Gx_interp(x, y);
            if (I_Jac(0, 0) == patch_procession_ptr->T_INVALID) {
                I_Jac(0, 0) = 0;
                // return true;
            }
            I_Jac(0, 1) = patch_procession_ptr->fetch_I_without_conformal_mapping(PatchProcession<T>::MatrixType::Gy,
                                                                                  p_o(0), p_o(1), pyramid_idx);
            //            I_Jac(0, 1) = patch_procession_ptr->fetch_I_Gy_interp(x, y);
            if (I_Jac(0, 1) == patch_procession_ptr->T_INVALID) {
                I_Jac(0, 1) = 0;
                // return true;
            }
            // XY_Jac
            Eigen::Matrix<double, 2, 3, Eigen::RowMajor> XY_Jac;
            XY_Jac << 1, 0, 0, 0, 1, 0;
            // Jac equation
            Eigen::Matrix<double, 1, 6> Jac_full = weight * (Z_Jac * SE3_Jac - I_Jac * XY_Jac * SE3_Jac);
            Jac(0, 0) = Jac_full(0, 2);
            Jac(0, 1) = Jac_full(0, 4);
        }
        return true;
    }

    int get_pyramid_idx() const { return pyramid_idx; }

    void set_pyramid_idx(const int _pyramid_idx) {
        if (_pyramid_idx >= 0) {
            pyramid_idx = _pyramid_idx;
        } else {
            pyramid_idx = 0;
        }
    }

    void update_pyramid_idx() {
        --pyramid_idx;
        if (pyramid_idx < 0) {
            pyramid_idx = 0;
        }
    }

    static ceres::CostFunction *Create(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_o_w,
                                       const Eigen::Vector3d &_p_j, PatchProcession<T> *_patch_processtion_ptr) {
        return (new CoupleFixSphDirectZPitchFactor(_T_o_w, _p_j, _patch_processtion_ptr));
    }

    static ceres::CostFunction *Create(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_o_w,
                                       const Eigen::Vector3d &_p_j, PatchProcession<T> *_patch_processtion_ptr,
                                       double _invalid_res) {
        return (new CoupleFixSphDirectZPitchFactor(_T_o_w, _p_j, _patch_processtion_ptr, _invalid_res));
    }

    static ceres::CostFunction *Create(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_o_w,
                                       const Eigen::Vector3d &_p_j, PatchProcession<T> *_patch_processtion_ptr,
                                       double _invalid_res, double _weight) {
        return (new CoupleFixSphDirectZPitchFactor(_T_o_w, _p_j, _patch_processtion_ptr, _invalid_res, _weight));
    }

  private:
    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_o_w;
    Eigen::Vector3d p_j;
    PatchProcession<T> *patch_procession_ptr;
    double invalid_res;
    double weight;
    int pyramid_idx;
};

template <typename T> class CoupleFixSphBADirectFactor : public ceres::SizedCostFunction<1, 16, 16> {
  public:
    CoupleFixSphBADirectFactor(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_o_i, const Eigen::Vector3d &_p_j,
                               PatchProcession<T> *_patch_processtion_ptr)
        : T_o_i(_T_o_i), p_j(_p_j), patch_procession_ptr(_patch_processtion_ptr) {
        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = T_o_i(Eigen::seq(0, 2), Eigen::seq(0, 2));
        T_o_i(Eigen::seq(0, 2), Eigen::seq(0, 2)) = Eigen::Quaterniond(R).normalized().toRotationMatrix();
        invalid_res = patch_procession_ptr->T_INVALID;
        weight = 1;
        pyramid_idx = patch_procession_ptr->get_pyramid_depth();
    }

    CoupleFixSphBADirectFactor(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_o_i, const Eigen::Vector3d &_p_j,
                               PatchProcession<T> *_patch_processtion_ptr, double _invalid_res)
        : T_o_i(_T_o_i), p_j(_p_j), patch_procession_ptr(_patch_processtion_ptr), invalid_res(_invalid_res) {
        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = T_o_i(Eigen::seq(0, 2), Eigen::seq(0, 2));
        T_o_i(Eigen::seq(0, 2), Eigen::seq(0, 2)) = Eigen::Quaterniond(R).normalized().toRotationMatrix();
        weight = 1;
        pyramid_idx = patch_procession_ptr->get_pyramid_depth();
    }

    CoupleFixSphBADirectFactor(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_o_i, const Eigen::Vector3d &_p_j,
                               PatchProcession<T> *_patch_processtion_ptr, double _invalid_res, double _weight)
        : T_o_i(_T_o_i), p_j(_p_j), patch_procession_ptr(_patch_processtion_ptr), invalid_res(_invalid_res),
          weight(_weight) {
        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = T_o_i(Eigen::seq(0, 2), Eigen::seq(0, 2));
        T_o_i(Eigen::seq(0, 2), Eigen::seq(0, 2)) = Eigen::Quaterniond(R).normalized().toRotationMatrix();
        pyramid_idx = patch_procession_ptr->get_pyramid_depth();
    }

    ~CoupleFixSphBADirectFactor() override {}

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override {
        Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> T_w_i(parameters[0]);
        Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> T_w_j(parameters[1]);
        Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_i_w;
        T_i_w.setIdentity();
        T_i_w(Eigen::seq(0, 2), Eigen::seq(0, 2)) = T_w_i(Eigen::seq(0, 2), Eigen::seq(0, 2)).transpose();
        T_i_w(Eigen::seq(0, 2), 3) =
            -T_w_i(Eigen::seq(0, 2), Eigen::seq(0, 2)).transpose() * T_w_i(Eigen::seq(0, 2), 3);
        Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_o_w = T_o_i * T_i_w;
        Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_o_j = T_o_w * T_w_j;
        Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_i_j = T_i_w * T_w_j;

        Eigen::Vector3d p_o = T_o_j(Eigen::seq(0, 2), Eigen::seq(0, 2)) * p_j + T_o_j(Eigen::seq(0, 2), 3);

        bool is_fault = false;
        double I = patch_procession_ptr->fetch_I_without_conformal_mapping(PatchProcession<T>::MatrixType::I, p_o(0),
                                                                           p_o(1), pyramid_idx);
        double res = (p_o(2) - I);
        if (I == patch_procession_ptr->T_INVALID) {
            is_fault = true;
            residuals[0] = 0;
        } else if (std::abs(res) > invalid_res) {
            is_fault = true;
            residuals[0] = 0;
        } else {
            residuals[0] = weight * res;
        }
        // jacobian building blocks
        Eigen::Matrix<double, 1, 2> I_Jac;
        Eigen::Matrix<double, 3, 6> SE3_Jac_j;
        Eigen::Matrix<double, 3, 6> SE3_Jac_i;
        // Eigen::Matrix<double, 6, 6> SE3_Jac_i_inv;
        Eigen::Matrix<double, 2, 3> XY_Jac;
        Eigen::Matrix<double, 1, 3> Z_Jac;
        if (!is_fault) {
            // I_Jac
            I_Jac(0, 0) = patch_procession_ptr->fetch_I_without_conformal_mapping(PatchProcession<T>::MatrixType::Gx,
                                                                                  p_o(0), p_o(1), pyramid_idx);
            if (I_Jac(0, 0) == patch_procession_ptr->T_INVALID) {
                I_Jac(0, 0) = 0;
                // is_fault = true;
            }
            I_Jac(0, 1) = patch_procession_ptr->fetch_I_without_conformal_mapping(PatchProcession<T>::MatrixType::Gy,
                                                                                  p_o(0), p_o(1), pyramid_idx);
            if (I_Jac(0, 1) == patch_procession_ptr->T_INVALID) {
                I_Jac(0, 1) = 0;
                // is_fault = true;
            }
            if (!is_fault) {
                if (!is_fault) {
                    // SE3_Jac_i
                    Eigen::Vector3d p_i = T_i_j(Eigen::seq(0, 2), Eigen::seq(0, 2)) * p_j + T_i_j(Eigen::seq(0, 2), 3);
                    SE3_Jac_i(Eigen::seq(0, 2), Eigen::seq(0, 2)) = -T_o_i(Eigen::seq(0, 2), Eigen::seq(0, 2));
                    SE3_Jac_i(Eigen::seq(0, 2), Eigen::seq(3, 5)) =
                        T_o_i(Eigen::seq(0, 2), Eigen::seq(0, 2)) * curl::skew<double>(p_i);
                    // SE3_Jac_j
                    SE3_Jac_j(Eigen::seq(0, 2), Eigen::seq(0, 2)) = T_o_j(Eigen::seq(0, 2), Eigen::seq(0, 2));
                    SE3_Jac_j(Eigen::seq(0, 2), Eigen::seq(3, 5)) =
                        -T_o_j(Eigen::seq(0, 2), Eigen::seq(0, 2)) * curl::skew<double>(p_j);
                    // XY_Jac
                    XY_Jac << 1, 0, 0, 0, 1, 0;
                    // Z_jac
                    Z_Jac << 0, 0, 1;
                }
            }
        }

        // Compute the Jacobian if asked for.
        if (jacobians != nullptr) {
            if (jacobians[0] != nullptr) {
                // Map jacobians for i
                Eigen::Map<Eigen::Matrix<double, 1, 16, Eigen::RowMajor>> Jac_i(jacobians[0], 1, 16);
                Jac_i.setZero();
                if (!is_fault) {
                    // Jac_i equation
                    Jac_i(0, Eigen::seq(0, 5)) = weight * (Z_Jac * SE3_Jac_i - I_Jac * XY_Jac * SE3_Jac_i);
                }
            }
            if (jacobians[1] != nullptr) {
                // Map jacobians for j
                Eigen::Map<Eigen::Matrix<double, 1, 16, Eigen::RowMajor>> Jac_j(jacobians[1], 1, 16);
                Jac_j.setZero();
                if (!is_fault) {
                    // Jac_j equation
                    Jac_j(0, Eigen::seq(0, 5)) = weight * (Z_Jac * SE3_Jac_j - I_Jac * XY_Jac * SE3_Jac_j);
                }
            }
        }
        return true;
    }

    int get_pyramid_idx() { return pyramid_idx; }

    void set_pyramid_idx(int _pyramid_idx) {
        if (_pyramid_idx >= 0) {
            pyramid_idx = _pyramid_idx;
        } else {
            pyramid_idx = 0;
        }
    }

    void update_pyramid_idx() {
        --pyramid_idx;
        if (pyramid_idx < 0) {
            pyramid_idx = 0;
        }
    }

    static ceres::CostFunction *Create(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_o_i,
                                       const Eigen::Vector3d &_p_j, PatchProcession<T> *_patch_processtion_ptr) {
        return (new CoupleFixSphBADirectFactor(_T_o_i, _p_j, _patch_processtion_ptr));
    }

    static ceres::CostFunction *Create(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_o_i,
                                       const Eigen::Vector3d &_p_j, PatchProcession<T> *_patch_processtion_ptr,
                                       double _invalid_res) {
        return (new CoupleFixSphBADirectFactor(_T_o_i, _p_j, _patch_processtion_ptr, _invalid_res));
    }

    static ceres::CostFunction *Create(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_o_i,
                                       const Eigen::Vector3d &_p_j, PatchProcession<T> *_patch_processtion_ptr,
                                       double _invalid_res, double _weight) {
        return (new CoupleFixSphBADirectFactor(_T_o_i, _p_j, _patch_processtion_ptr, _invalid_res, _weight));
    }

  private:
    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_o_i;
    Eigen::Vector3d p_j;
    PatchProcession<T> *patch_procession_ptr;
    double invalid_res;
    double weight;
    int pyramid_idx;
};

template <typename T, int DEGREE_SIZE>
class CoupleSphModFactor : public ceres::SizedCostFunction<1, 16, 16, DEGREE_SIZE> {
  public:
    CoupleSphModFactor(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_o_i, const Eigen::Vector3d &_p_j,
                       std::shared_ptr<PatchProcession<T>> _patch_processtion_ptr, double _invalid_res,
                       std::string _opt_status, std::shared_ptr<SH_TABLE_CONFIG<T>> _SH_table_config_ptr)
        : T_o_i(_T_o_i), p_j(_p_j), patch_procession_ptr(_patch_processtion_ptr), invalid_res(_invalid_res),
          opt_status(_opt_status), SH_table_config_ptr(_SH_table_config_ptr) {
        weight = 1;
        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = T_o_i(Eigen::seq(0, 2), Eigen::seq(0, 2));
        T_o_i(Eigen::seq(0, 2), Eigen::seq(0, 2)) = Eigen::Quaterniond(R).normalized().toRotationMatrix();
        degree = patch_procession_ptr->get_SH_degree();
    }
    ~CoupleSphModFactor() override {}
    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override {
        Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> T_w_i(parameters[0]);
        Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> T_w_j(parameters[1]);
        Eigen::Map<const Eigen::VectorXd> sph_coeff(parameters[2], DEGREE_SIZE);
        Eigen::Vector3d p_o =
            Eigen::Isometry3d(T_o_i) * Eigen::Isometry3d(T_w_i).inverse() * Eigen::Isometry3d(T_w_j) * p_j;
        double phi = patch_procession_ptr->get_phi(p_o(0));
        double theta = patch_procession_ptr->get_theta(p_o(1));
        int phi_idx = -1;
        int theta_idx = -1;
        double I = patch_procession_ptr->T_INVALID;
        if (phi != patch_procession_ptr->T_INVALID && theta != patch_procession_ptr->T_INVALID) {
            phi_idx = std::round((phi - SH_table_config_ptr->get_azi_low()) /
                                 (SH_table_config_ptr->get_azi_high() - SH_table_config_ptr->get_azi_low()) *
                                 (SH_table_config_ptr->azi_rso - 1));
            theta_idx = std::round((theta - SH_table_config_ptr->get_elev_low()) /
                                   (SH_table_config_ptr->get_elev_high() - SH_table_config_ptr->get_elev_low()) *
                                   (SH_table_config_ptr->elev_rso - 1));
            abort_if_sph_index_oob(phi_idx, theta_idx, SH_table_config_ptr->azi_rso, SH_table_config_ptr->elev_rso, phi,
                                   theta, "CoupleSphModFactor");
            const int row = phi_idx * SH_table_config_ptr->elev_rso + theta_idx;
            if (row >= 0 && row < SH_table_config_ptr->SH_table.rows()) {
                I = SH_table_config_ptr->SH_table(row, Eigen::seq(0, DEGREE_SIZE - 1))
                        .template cast<double>()
                        .dot(sph_coeff);
            }
        }
        weight = 1;
        bool is_fault = false;
        bool is_optimize_pose = true;
        bool is_optimize_sph_coeff = true;
        if (opt_status == "totally_separate") {
            if (I != patch_procession_ptr->T_INVALID) {
                is_fault = false;
                const double res = p_o(2) - I;
                bool is_valid_in_mask;
                if (!SH_table_config_ptr->is_SH_analytic_jacobian) {
                    is_valid_in_mask = patch_procession_ptr->is_valid_in_mask(p_o(0), p_o(1));
                } else {
                    is_valid_in_mask = patch_procession_ptr->is_valid_in_mask_low_RAM(p_o(0), p_o(1));
                }
                if (is_valid_in_mask) {
                    weight = prior_weight_square_root;
                    if (std::abs(res) <= invalid_res) {
                        is_optimize_pose = true;
                    } else {
                        is_optimize_pose = false;
                    }
                    is_optimize_sph_coeff = false;
                } else {
                    weight = observation_weight_square_root;
                    is_optimize_pose = false;
                    is_optimize_sph_coeff = true;
                }
                residuals[0] = weight * res;
            } else {
                is_fault = true;
                is_optimize_pose = false;
                is_optimize_sph_coeff = false;
                residuals[0] = 0;
            }
        } else if (opt_status == "half_separate") {
            if (I != patch_procession_ptr->T_INVALID) {
                is_fault = false;
                const double res = p_o(2) - I;

                if (std::abs(res) <= invalid_res) {
                    is_optimize_pose = true;
                } else {
                    is_optimize_pose = false;
                }
                bool is_valid_in_mask;
                if (!SH_table_config_ptr->is_SH_analytic_jacobian) {
                    is_valid_in_mask = patch_procession_ptr->is_valid_in_mask(p_o(0), p_o(1));
                } else {
                    is_valid_in_mask = patch_procession_ptr->is_valid_in_mask_low_RAM(p_o(0), p_o(1));
                }
                if (is_valid_in_mask) {
                    weight = prior_weight_square_root;
                    // is_optimize_sph_coeff = true;
                } else {
                    weight = observation_weight_square_root;
                    // is_optimize_sph_coeff = true;
                }
                residuals[0] = weight * res;
            } else {
                is_fault = true;
                is_optimize_pose = false;
                is_optimize_sph_coeff = false;
                residuals[0] = 0;
            }
        } else if (opt_status == "no_separate") {
            if (I != patch_procession_ptr->T_INVALID) {
                is_fault = false;
                const double res = p_o(2) - I;
                bool is_valid_in_mask;
                if (!SH_table_config_ptr->is_SH_analytic_jacobian) {
                    is_valid_in_mask = patch_procession_ptr->is_valid_in_mask(p_o(0), p_o(1));
                } else {
                    is_valid_in_mask = patch_procession_ptr->is_valid_in_mask_low_RAM(p_o(0), p_o(1));
                }
                if (is_valid_in_mask) {
                    weight = prior_weight_square_root;
                } else {
                    weight = observation_weight_square_root;
                }
                if (std::abs(res) <= invalid_res) {
                    is_optimize_pose = true;
                } else {
                    is_optimize_pose = false;
                }
                is_optimize_sph_coeff = true;
                residuals[0] = weight * res;
            } else {
                is_fault = true;
                is_optimize_pose = false;
                is_optimize_sph_coeff = false;
                residuals[0] = 0;
            }
        }
        // jacobian building blocks
        Eigen::Matrix<double, 3, 6> SE3_Jac_i;
        Eigen::Matrix<double, 3, 6> SE3_Jac_j;
        Eigen::Matrix<double, 1, 2> I_Jac_sum;
        I_Jac_sum.setZero();
        Eigen::Matrix<double, 2, 6> SE3_block_Jac_i;
        Eigen::Matrix<double, 2, 6> SE3_block_Jac_j;
        // phi_idx/theta_idx already computed above for SH-table lookup

        if (!is_fault) {
            SE3_Jac_i(Eigen::seq(0, 2), Eigen::seq(0, 2)) = -T_o_i(Eigen::seq(0, 2), Eigen::seq(0, 2));
            SE3_Jac_i(Eigen::seq(0, 2), Eigen::seq(3, 5)) =
                T_o_i(Eigen::seq(0, 2), Eigen::seq(0, 2)) *
                curl::skew(Eigen::Isometry3d(T_w_i).inverse() * Eigen::Isometry3d(T_w_j) * p_j);
            SE3_block_Jac_i.row(0) = patch_procession_ptr->get_G_phi() * SH_table_config_ptr->X_Jac *
                                     SH_table_config_ptr->XY_Jac * SE3_Jac_i;
            SE3_block_Jac_i.row(1) = patch_procession_ptr->get_G_theta() * SH_table_config_ptr->Y_Jac *
                                     SH_table_config_ptr->XY_Jac * SE3_Jac_i;
            SE3_Jac_j(Eigen::seq(0, 2), Eigen::seq(0, 2)) = T_o_i(Eigen::seq(0, 2), Eigen::seq(0, 2)) *
                                                            T_w_i(Eigen::seq(0, 2), Eigen::seq(0, 2)).transpose() *
                                                            T_w_j(Eigen::seq(0, 2), Eigen::seq(0, 2));
            SE3_Jac_j(Eigen::seq(0, 2), Eigen::seq(3, 5)) = -T_o_i(Eigen::seq(0, 2), Eigen::seq(0, 2)) *
                                                            T_w_i(Eigen::seq(0, 2), Eigen::seq(0, 2)).transpose() *
                                                            T_w_j(Eigen::seq(0, 2), Eigen::seq(0, 2)) * curl::skew(p_j);
            SE3_block_Jac_j.row(0) = patch_procession_ptr->get_G_phi() * SH_table_config_ptr->X_Jac *
                                     SH_table_config_ptr->XY_Jac * SE3_Jac_j;
            SE3_block_Jac_j.row(1) = patch_procession_ptr->get_G_theta() * SH_table_config_ptr->Y_Jac *
                                     SH_table_config_ptr->XY_Jac * SE3_Jac_j;
            Eigen::Matrix<double, 1, 2> SPH_Jac;
            SPH_Jac.setZero();
            const std::pair<bool, bool> is_on_edge = patch_procession_ptr->check_is_on_edge_low_RAM(p_o(0), p_o(1));
            // const std::pair<bool, bool> is_on_edge = std::make_pair(false, false);
            for (int l = 0; l <= degree; ++l) {
                for (int m = -l; m <= l; ++m) {
                    if (!is_on_edge.first) {
                        SPH_Jac(0, 0) = static_cast<double>(SH_table_config_ptr->SH_G_phi_table(
                            phi_idx * SH_table_config_ptr->elev_rso + theta_idx, l * l + l + m));
                    }
                    if (!is_on_edge.second) {
                        SPH_Jac(0, 1) = static_cast<double>(SH_table_config_ptr->SH_G_theta_table(
                            phi_idx * SH_table_config_ptr->elev_rso + theta_idx, l * l + l + m));
                    }
                    I_Jac_sum += sph_coeff(l * l + l + m) * SPH_Jac;
                }
            }
        }

        if (jacobians != nullptr) {
            if (jacobians[0] != nullptr) {
                Eigen::Map<Eigen::Matrix<double, 1, 16, Eigen::RowMajor>> Jac_i(jacobians[0], 1, 16);
                Jac_i.setZero();
                if (!is_fault && is_optimize_pose) {
                    Jac_i(0, Eigen::seq(0, 5)) =
                        weight * (SH_table_config_ptr->Z_Jac * SE3_Jac_i - I_Jac_sum * SE3_block_Jac_i);
                }
            }
            if (jacobians[1] != nullptr) {
                Eigen::Map<Eigen::Matrix<double, 1, 16, Eigen::RowMajor>> Jac_j(jacobians[1], 1, 16);
                Jac_j.setZero();
                if (!is_fault && is_optimize_pose) {
                    Jac_j(0, Eigen::seq(0, 5)) =
                        weight * (SH_table_config_ptr->Z_Jac * SE3_Jac_j - I_Jac_sum * SE3_block_Jac_j);
                }
            }
            if (jacobians[2] != nullptr) {
                Eigen::Map<Eigen::Matrix<double, 1, DEGREE_SIZE, Eigen::RowMajor>> Jac_c(jacobians[2], 1, DEGREE_SIZE);
                Jac_c.setZero();
                if (!is_fault && is_optimize_sph_coeff) {
                    Jac_c = weight * (-SH_table_config_ptr
                                           ->SH_table(phi_idx * SH_table_config_ptr->elev_rso + theta_idx,
                                                      Eigen::seq(0, DEGREE_SIZE - 1))
                                           .template cast<double>());
                }
            }
        }
        return true;
    }

    void set_sph_coeff_lock() { patch_procession_ptr->set_sph_coeff_lock(); }

    void unset_sph_coeff_lock() { patch_procession_ptr->unset_sph_coeff_lock(); }

    static void set_prior_weight_square_root(double _prior_weight_square_root) {
        prior_weight_square_root = _prior_weight_square_root;
    }

    static void set_observation_weight_square_root(double _observation_weight_square_root) {
        observation_weight_square_root = _observation_weight_square_root;
    }

    static double get_prior_weight_square_root() { return prior_weight_square_root; }

    static double get_observation_weight_square_root() { return observation_weight_square_root; }

    static ceres::CostFunction *Create(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_o_i,
                                       const Eigen::Vector3d &_p_j,
                                       std::shared_ptr<PatchProcession<T>> _patch_processtion_ptr, double _invalid_res,
                                       std::string _opt_status,
                                       std::shared_ptr<SH_TABLE_CONFIG<T>> _SH_table_config_ptr) {
        return (new CoupleSphModFactor(_T_o_i, _p_j, _patch_processtion_ptr, _invalid_res, _opt_status,
                                       _SH_table_config_ptr));
    }

  private:
    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_o_i;
    Eigen::Vector3d p_j;
    std::shared_ptr<PatchProcession<T>> patch_procession_ptr;
    double invalid_res;
    std::shared_ptr<SH_TABLE_CONFIG<T>> SH_table_config_ptr;
    int degree;
    mutable double weight;
    static double prior_weight_square_root;
    static double observation_weight_square_root;
    std::string opt_status;
};

template <typename T, int DEGREE_SIZE> double CoupleSphModFactor<T, DEGREE_SIZE>::prior_weight_square_root = 1;
template <typename T, int DEGREE_SIZE> double CoupleSphModFactor<T, DEGREE_SIZE>::observation_weight_square_root = 1;

template <typename T, int DEGREE_SIZE> class CoupleSphModRegFactor : public ceres::SizedCostFunction<1, DEGREE_SIZE> {
  public:
    CoupleSphModRegFactor(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_o_i, const Eigen::Vector3d &_p_j,
                          const double _max_valid_height, std::shared_ptr<PatchProcession<T>> _patch_processtion_ptr,
                          std::shared_ptr<SH_TABLE_CONFIG<T>> _SH_table_config_ptr, double _weight_square_root)
        : T_o_i(_T_o_i), p_j(_p_j), max_valid_height(_max_valid_height), patch_procession_ptr(_patch_processtion_ptr),
          SH_table_config_ptr(_SH_table_config_ptr), weight_square_root(_weight_square_root) {
        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = T_o_i(Eigen::seq(0, 2), Eigen::seq(0, 2));
        T_o_i(Eigen::seq(0, 2), Eigen::seq(0, 2)) = Eigen::Quaterniond(R).normalized().toRotationMatrix();
        degree = std::sqrt(DEGREE_SIZE) - 1;
    }
    ~CoupleSphModRegFactor() override {}
    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override {
        Eigen::Map<const Eigen::VectorXd> sph_coeff(parameters[0], DEGREE_SIZE);
        Eigen::Vector3d p_o = Eigen::Isometry3d(T_o_i) * p_j;
        double phi = patch_procession_ptr->get_phi(p_o(0));
        double theta = patch_procession_ptr->get_theta(p_o(1));
        int phi_idx = -1;
        int theta_idx = -1;
        double I = patch_procession_ptr->T_INVALID;
        if (phi != patch_procession_ptr->T_INVALID && theta != patch_procession_ptr->T_INVALID) {
            phi_idx = std::round((phi - SH_table_config_ptr->get_azi_low()) /
                                 (SH_table_config_ptr->get_azi_high() - SH_table_config_ptr->get_azi_low()) *
                                 (SH_table_config_ptr->azi_rso - 1));
            theta_idx = std::round((theta - SH_table_config_ptr->get_elev_low()) /
                                   (SH_table_config_ptr->get_elev_high() - SH_table_config_ptr->get_elev_low()) *
                                   (SH_table_config_ptr->elev_rso - 1));
            abort_if_sph_index_oob(phi_idx, theta_idx, SH_table_config_ptr->azi_rso, SH_table_config_ptr->elev_rso, phi,
                                   theta, "CoupleSphModRegFactor");
            const int row = phi_idx * SH_table_config_ptr->elev_rso + theta_idx;
            if (row >= 0 && row < SH_table_config_ptr->SH_table.rows()) {
                I = SH_table_config_ptr->SH_table(row, Eigen::seq(0, DEGREE_SIZE - 1))
                        .template cast<double>()
                        .dot(sph_coeff);
            }
        }
        bool is_fault = false;
        if (I != patch_procession_ptr->T_INVALID && std::abs(p_o(2)) < max_valid_height) {
            double res = p_o(2) - I;
            residuals[0] = weight_square_root * res;
        } else {
            is_fault = true;
            residuals[0] = 0;
        }
        // jacobian building blocks
        if (jacobians != nullptr) {
            if (jacobians[0] != nullptr) {
                Eigen::Map<Eigen::Matrix<double, 1, DEGREE_SIZE, Eigen::RowMajor>> Jac_c(jacobians[0], 1, DEGREE_SIZE);
                Jac_c.setZero();
                if (!is_fault) {
                    Jac_c = weight_square_root * (-SH_table_config_ptr
                                                       ->SH_table(phi_idx * SH_table_config_ptr->elev_rso + theta_idx,
                                                                  Eigen::seq(0, DEGREE_SIZE - 1))
                                                       .template cast<double>());
                }
            }
        }
        return true;
    }

    void set_sph_coeff_lock() { patch_procession_ptr->set_sph_coeff_lock(); }

    void unset_sph_coeff_lock() { patch_procession_ptr->unset_sph_coeff_lock(); }

    static ceres::CostFunction *Create(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_o_i,
                                       const Eigen::Vector3d &_p_j, const double _max_valid_height,
                                       std::shared_ptr<PatchProcession<T>> _patch_processtion_ptr,
                                       std::shared_ptr<SH_TABLE_CONFIG<T>> _SH_table_config_ptr,
                                       double _weight_square_root) {
        return (new CoupleSphModRegFactor(_T_o_i, _p_j, _max_valid_height, _patch_processtion_ptr, _SH_table_config_ptr,
                                          _weight_square_root));
    }

  private:
    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_o_i;
    Eigen::Vector3d p_j;
    std::shared_ptr<PatchProcession<T>> patch_procession_ptr;
    std::shared_ptr<SH_TABLE_CONFIG<T>> SH_table_config_ptr;
    int degree;
    double max_valid_height;
    double weight_square_root;
};

template <int DEGREE_SIZE> class SphRegFactor : public ceres::SizedCostFunction<1, DEGREE_SIZE> {
  public:
    SphRegFactor(double _weight_square_root) : weight_square_root(_weight_square_root){};
    ~SphRegFactor() override {}
    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override {
        Eigen::Map<const Eigen::VectorXd> sph_coeff(parameters[0], DEGREE_SIZE);
        const double norm = sph_coeff.norm();
        residuals[0] = weight_square_root * norm;
        // Compute the Jacobian if asked for.
        if (jacobians != nullptr && jacobians[0] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 1, DEGREE_SIZE, Eigen::RowMajor>> Jac_c(jacobians[0], 1, DEGREE_SIZE);
            Jac_c = weight_square_root / norm * sph_coeff;
        }
        return true;
    }
    static ceres::CostFunction *Create(double _weight_square_root) { return (new SphRegFactor(_weight_square_root)); }

  private:
    double weight_square_root;
};

template <typename T> class IcpFactor : public ceres::SizedCostFunction<3, 16> {
  public:
    IcpFactor(const Eigen::Vector3d _p_j, const Eigen::Vector3d &_p_w) : p_j(_p_j), p_w(_p_w) {}
    ~IcpFactor() override {}

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override {
        Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> T_w_j(parameters[0]);
        Eigen::Map<Eigen::Vector3d> res(residuals);
        res = T_w_j * p_j - p_w;
        if (jacobians != nullptr) {
            if (jacobians[0] != nullptr) {
                Eigen::Map<Eigen::Matrix<double, 3, 16, Eigen::RowMajor>> Jac(jacobians[0]);
                Jac.setZero();
                Jac(Eigen::seq(0, 2)) = T_w_j(Eigen::seq(0, 2), Eigen::seq(0, 2));
                Jac(Eigen::seq(0, 2), Eigen::seq(3, 5)) =
                    -T_w_j(Eigen::seq(0, 2), Eigen::seq(0, 2)) * curl::skew<double>(p_j);
            }
        }
        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d &_p_w) { return (new IcpFactor(_p_w)); }

  private:
    Eigen::Vector3d p_j;
    Eigen::Vector3d p_w;
};

class IcpPointToPlaneFactor : public ceres::SizedCostFunction<1, 16> {
  public:
    IcpPointToPlaneFactor(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> _T_o_w, const Eigen::Vector3d _p_j,
                          const Eigen::Vector3d &_p_o, const Eigen::Vector3d &_p_o_normal)
        : T_o_w(_T_o_w), p_j(_p_j), p_o(_p_o), p_o_normal(_p_o_normal) {}
    ~IcpPointToPlaneFactor() override {}

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override {
        Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> T_w_j(parameters[0]);
        Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_o_j = T_o_w * T_w_j;
        residuals[0] =
            (T_o_j(Eigen::seq(0, 2), Eigen::seq(0, 2)) * p_j + T_o_j(Eigen::seq(0, 2), 3) - p_o).transpose() *
            p_o_normal;
        if (residuals[0] == NAN) {
            std::cout << "p_j: " << p_j << std::endl;
            std::cout << "p_o: " << p_o << std::endl;
            std::cout << "p_o_normal: " << p_o_normal << std::endl;
        }

        if (jacobians != nullptr) {
            if (jacobians[0] != nullptr) {
                Eigen::Map<Eigen::Matrix<double, 1, 16, Eigen::RowMajor>> Jac(jacobians[0]);
                Jac.setZero();
                Eigen::Matrix<double, 3, 6> Jac_SE3;
                Jac_SE3(Eigen::seq(0, 2), Eigen::seq(0, 2)) = T_o_j(Eigen::seq(0, 2), Eigen::seq(0, 2));
                Jac_SE3(Eigen::seq(0, 2), Eigen::seq(3, 5)) =
                    -T_o_j(Eigen::seq(0, 2), Eigen::seq(0, 2)) * curl::skew<double>(p_j);
                Jac(0, Eigen::seq(0, 5)) = p_o_normal.transpose() * Jac_SE3;
            }
        }
        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> _T_o_w,
                                       const Eigen::Vector3d _p_j, const Eigen::Vector3d &_p_o,
                                       const Eigen::Vector3d &_p_o_normal) {
        return (new IcpPointToPlaneFactor(_T_o_w, _p_j, _p_o, _p_o_normal));
    }

  private:
    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_o_w;
    Eigen::Vector3d p_j;
    Eigen::Vector3d p_o;
    Eigen::Vector3d p_o_normal;
};

class PoseGraph3dErrorTerm {
  public:
    PoseGraph3dErrorTerm(const Pose3d &_t_ab_measured, bool _is_icp_constrain)
        : t_ab_measured(_t_ab_measured), is_icp_constrain(_is_icp_constrain) {
        weight = 1;
    }

    PoseGraph3dErrorTerm(const Pose3d &_t_ab_measured) : t_ab_measured(_t_ab_measured) {
        weight = 1;
        is_icp_constrain = true;
    }

    template <typename T>
    bool operator()(const T *const p_a_ptr, const T *const q_a_ptr, const T *const p_b_ptr, const T *const q_b_ptr,
                    T *residuals_ptr) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_a(p_a_ptr);
        Eigen::Map<const Eigen::Quaternion<T>> q_a(q_a_ptr);

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_b(p_b_ptr);
        Eigen::Map<const Eigen::Quaternion<T>> q_b(q_b_ptr);

        Sophus::SE3<T> T_a(q_a, p_a);
        Sophus::SE3<T> T_b(q_b, p_b);
        Sophus::SE3<T> T_ab = T_a.inverse() * T_b;
        Sophus::SE3<T> T_ab_measure(t_ab_measured.q.template cast<T>(), t_ab_measured.p.template cast<T>());
        Sophus::SE3<T> delta_T = T_ab_measure.inverse() * T_ab;
        Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
        residuals = delta_T.log() * T(weight);
        return true;
    }

    void remove_icp_constrain() {
        if (is_icp_constrain) {
            weight = 0;
            std::cout << "++++++++++++++++++++" << std::endl;
            std::cout << "remove icp constrain" << std::endl;
            std::cout << "++++++++++++++++++++" << std::endl;
        }
    }

    void add_icp_constrain() {
        if (is_icp_constrain) {
            weight = 1;
            std::cout << "++++++++++++++++++++" << std::endl;
            std::cout << "add icp constrain" << std::endl;
            std::cout << "++++++++++++++++++++" << std::endl;
        }
    }

    void remove_this_constrain() { weight = 0; }

    void add_this_constrain() { weight = 1; }

    void set_t_ab_measured(const Pose3d &_t_ab_measured) { t_ab_measured = _t_ab_measured; }

    static ceres::AutoDiffCostFunction<PoseGraph3dErrorTerm, 6, 3, 4, 3, 4> *Create(const Pose3d &_t_ab_measured,
                                                                                    bool _is_icp_constrain) {
        return new ceres::AutoDiffCostFunction<PoseGraph3dErrorTerm, 6, 3, 4, 3, 4>(
            new PoseGraph3dErrorTerm(_t_ab_measured, _is_icp_constrain));
    }

  private:
    // The measurement for the position of B relative to A in the A frame.
    Pose3d t_ab_measured;
    bool is_icp_constrain;
    double weight;
};

// SPH modification factor
// template <int DEGREE_SIZE> class CoupleSphModFactor : public ceres::SizedCostFunction<1, 16, DEGREE_SIZE> {
//   public:
//     CoupleSphModFactor(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_o_w, const Eigen::Vector3d &_p_j,
//                        const PatchProcession<double> *_patch_processtion_ptr,
//                        const SH_TABLE_CONFIG<double> *_SH_table_config_ptr)
//         : T_o_w(_T_o_w), p_j(_p_j), patch_procession_ptr(_patch_processtion_ptr),
//           SH_table_config_ptr(_SH_table_config_ptr) {
//         Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = T_o_w(Eigen::seq(0, 2), Eigen::seq(0, 2));
//         T_o_w(Eigen::seq(0, 2), Eigen::seq(0, 2)) = Eigen::Quaterniond(R).normalized().toRotationMatrix();
//     }

//     ~CoupleSphModFactor() override {}
//     bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override {
//         Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> T_w_j(parameters[0]);
//         Eigen::Map<const Eigen::VectorX<double>> sph_coeff(parameters[1], DEGREE_SIZE);
//         Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_o_j = T_o_w * T_w_j;
//         Eigen::Vector3d p_o = T_o_j(Eigen::seq(0, 2), Eigen::seq(0, 2)) * p_j + T_o_j(Eigen::seq(0, 2), 3);
//         double x = patch_procession_ptr->fetch_Gamma(patch_procession_ptr->Gamma_x, p_o(0), p_o(1));
//         //        double x = patch_procession_ptr->fetch_Gamma_x_interp(p_o(0), p_o(1));
//         bool is_fault = false;
//         if (x == patch_procession_ptr->T_INVALID) {
//             is_fault = true;
//         }
//         double y = patch_procession_ptr->fetch_Gamma(patch_procession_ptr->Gamma_y, p_o(0), p_o(1));
//         //        double y = patch_procession_ptr->fetch_Gamma_y_interp(p_o(0), p_o(1));
//         if (y == patch_procession_ptr->T_INVALID) {
//             is_fault = true;
//         }
//         double azi_low = SH_table_config_ptr->get_azi_low();
//         double azi_high = SH_table_config_ptr->get_azi_high();
//         double elev_low = SH_table_config_ptr->get_elev_low();
//         double elev_high = SH_table_config_ptr->get_elev_high();
//         double theta = y / patch_procession_ptr->max_y * M_PI * SH_table_config_ptr->SH_scale + elev_low;
//         double phi = x / patch_procession_ptr->max_x * 2 * M_PI * SH_table_config_ptr->SH_scale + azi_low;
//         int theta_idx = std::round((theta - elev_low) / (elev_high - elev_low) * (SH_table_config_ptr->elev_rso -
//         1)); int phi_idx = std::round((phi - azi_low) / (azi_high - azi_low) * (SH_table_config_ptr->azi_rso - 1));
//         if (!is_fault) {
//             // use spherical harmonics coefficients to reconstruct the point
//             Eigen::MatrixXd dirs(1, 2);
//             dirs << phi, theta;
//             Eigen::VectorXd I = curl::invLeastSquaresSHT_table<double>(sph_coeff, dirs, *SH_table_config_ptr,
//                                                                        SH_table_config_ptr->warping_SH_degree);
//             //            Eigen::VectorXd I =
//             //                curl::invLeastSquaresSHT<double>(sph_coeff, dirs,
//             SH_table_config_ptr->warping_SH_degree); residuals[0] = p_o(2) - I(0); if (I(0) ==
//             patch_procession_ptr->T_INVALID) {
//                 residuals[0] = 0;
//             }

//         } else {
//             residuals[0] = 0;
//         }

//         // Compute the Jacobian if asked for.
//         if (jacobians != nullptr) {
//             if (jacobians[0] != nullptr) {
//                 // Map jacobians
//                 Eigen::Map<Eigen::Matrix<double, 1, 16, Eigen::RowMajor>> Jac_left(jacobians[0], 1, 16);
//                 Jac_left.setZero();
//                 if (!is_fault) {
//                     // Z_jac
//                     Eigen::Matrix<double, 1, 3, Eigen::RowMajor> Z_Jac;
//                     Z_Jac << 0, 0, 1;
//                     // SE3_Jac
//                     Eigen::Matrix<double, 3, 6> SE3_Jac;
//                     SE3_Jac(Eigen::seq(0, 2), Eigen::seq(0, 2)) = T_o_j(Eigen::seq(0, 2), Eigen::seq(0, 2));
//                     SE3_Jac(Eigen::seq(0, 2), Eigen::seq(3, 5)) =
//                         -T_o_j(Eigen::seq(0, 2), Eigen::seq(0, 2)) * curl::skew<double>(p_j);
//                     // Gamma_Jac
//                     Eigen::Matrix<double, 2, 2, Eigen::RowMajor> Gamma_Jac;
//                     Gamma_Jac(0, 0) =
//                         patch_procession_ptr->fetch_Gamma(patch_procession_ptr->Gamma_x_Gx, p_o(0), p_o(1));
//                     //                    Gamma_Jac(0, 0) = patch_procession_ptr->fetch_Gamma_x_Gx_interp(p_o(0),
//                     //                    p_o(1));
//                     if (Gamma_Jac(0, 0) == patch_procession_ptr->T_INVALID) {
//                         is_fault = true;
//                     }
//                     Gamma_Jac(0, 1) =
//                         patch_procession_ptr->fetch_Gamma(patch_procession_ptr->Gamma_x_Gy, p_o(0), p_o(1));
//                     //                    Gamma_Jac(0, 1) = patch_procession_ptr->fetch_Gamma_x_Gy_interp(p_o(0),
//                     //                    p_o(1));
//                     if (Gamma_Jac(0, 1) == patch_procession_ptr->T_INVALID) {
//                         is_fault = true;
//                     }
//                     Gamma_Jac(1, 0) =
//                         patch_procession_ptr->fetch_Gamma(patch_procession_ptr->Gamma_y_Gx, p_o(0), p_o(1));
//                     //                    Gamma_Jac(1, 0) = patch_procession_ptr->fetch_Gamma_y_Gx_interp(p_o(0),
//                     //                    p_o(1));
//                     if (Gamma_Jac(1, 0) == patch_procession_ptr->T_INVALID) {
//                         is_fault = true;
//                     }
//                     Gamma_Jac(1, 1) =
//                         patch_procession_ptr->fetch_Gamma(patch_procession_ptr->Gamma_y_Gy, p_o(0), p_o(1));
//                     //                    Gamma_Jac(1, 1) = patch_procession_ptr->fetch_Gamma_y_Gy_interp(p_o(0),
//                     //                    p_o(1));
//                     if (Gamma_Jac(1, 1) == patch_procession_ptr->T_INVALID) {
//                         is_fault = true;
//                     }
//                     if (!is_fault) {
//                         // XY_Jac
//                         Eigen::Matrix<double, 2, 3, Eigen::RowMajor> XY_Jac;
//                         XY_Jac << 1, 0, 0, 0, 1, 0;
//                         // theta_Jac
//                         double theta_Jac = SH_table_config_ptr->SH_scale * M_PI / patch_procession_ptr->max_y;
//                         // phi_Jac
//                         double phi_Jac = SH_table_config_ptr->SH_scale * 2 * M_PI / patch_procession_ptr->max_x;
//                         // Y_Jac
//                         Eigen::Matrix<double, 1, 2, Eigen::RowMajor> Y_Jac;
//                         Y_Jac << 0, 1;
//                         // X_Jac
//                         Eigen::Matrix<double, 1, 2, Eigen::RowMajor> X_Jac;
//                         X_Jac << 1, 0;
//                         // Block_Jac
//                         Eigen::Matrix<double, 2, 6, Eigen::RowMajor> Block_Jac;
//                         Block_Jac.row(0) = theta_Jac * Y_Jac * Gamma_Jac * XY_Jac * SE3_Jac;
//                         Block_Jac.row(1) = phi_Jac * X_Jac * Gamma_Jac * XY_Jac * SE3_Jac;
//                         // Right_Jac & SPH_Jac
//                         Eigen::Matrix<double, 1, 6, Eigen::RowMajor> Right_Jac;
//                         Right_Jac.setZero();
//                         Eigen::Matrix<double, 1, 2, Eigen::RowMajor> SPH_Jac;
//                         for (int l = 0; l <= SH_table_config_ptr->warping_SH_degree; ++l) {
//                             for (int m = -l; m <= l; ++m) {
//                                 SPH_Jac(0, 0) = SH_table_config_ptr->SH_G_theta_table(
//                                     phi_idx * SH_table_config_ptr->elev_rso + theta_idx, l * l + l + m);
//                                 SPH_Jac(0, 1) = SH_table_config_ptr->SH_G_phi_table(
//                                     phi_idx * SH_table_config_ptr->elev_rso + theta_idx, l * l + l + m);
//                                 Right_Jac = (Right_Jac + sph_coeff(l * l + l + m) * SPH_Jac * Block_Jac).eval();
//                             }
//                         }
//                         // Jac_left equation
//                         Jac_left(0, Eigen::seq(0, 5)) = Z_Jac * SE3_Jac - Right_Jac;
//                     }
//                 }
//             }
//             if (jacobians[1] != nullptr) {
//                 Eigen::Map<Eigen::Matrix<double, 1, DEGREE_SIZE>> Jac_right(jacobians[1]);
//                 Jac_right.setZero();
//                 if (!is_fault) {
//                     Jac_right = -SH_table_config_ptr->SH_table(phi_idx * SH_table_config_ptr->elev_rso + theta_idx,
//                                                                Eigen::seq(0, DEGREE_SIZE - 1));
//                 }
//             }
//         }
//         return true;
//     }

//     static ceres::CostFunction *Create(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &_T_o_w,
//                                        const Eigen::Vector3d &_p_j,
//                                        const PatchProcession<double> *_patch_processtion_ptr,
//                                        const SH_TABLE_CONFIG<double> *_SH_table_config_ptr) {
//         return (new CoupleSphModFactor(_T_o_w, _p_j, _patch_processtion_ptr, _SH_table_config_ptr));
//     }

//   private:
//     Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_o_w;
//     Eigen::Vector3d p_j;
//     const PatchProcession<double> *patch_procession_ptr;
//     const SH_TABLE_CONFIG<T><double> *SH_table_config_ptr;
// };

#endif // CURL_SLAM_OPTIMIZATION_TYPES_H
