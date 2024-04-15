// Copyright 2022 Chen Jun

#ifndef ARMOR_PROCESSOR__KALMAN_FILTER_HPP_
#define ARMOR_PROCESSOR__KALMAN_FILTER_HPP_

#include <Eigen/Dense>
#include <functional>

namespace rm_auto_aim
{

class UnscentedKalmanFilter
{
  public:
    UnscentedKalmanFilter() = default;

    using VecVecFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd &)>;
    using VecMatFunc = std::function<Eigen::MatrixXd(const Eigen::VectorXd &)>;
    using VoidMatFunc = std::function<Eigen::MatrixXd()>;

    explicit UnscentedKalmanFilter(const VecVecFunc & f, const VecVecFunc & h,
    const VoidMatFunc & u_q, const VecMatFunc & u_r, const Eigen::MatrixXd & P0);

    void setState(const Eigen::VectorXd & x0);
    
    Eigen::VectorXd predict();
    Eigen::VectorXd update(const Eigen::VectorXd & z);

  private:
  // Process nonlinear vector function
  VecVecFunc f;
  // Observation nonlinear vector function
  VecVecFunc h;
  // Process noise covariance matrix
  VoidMatFunc update_Q;
  Eigen::MatrixXd Q;
  // Measurement noise covariance matrix
  VecMatFunc update_R;
  Eigen::MatrixXd R;

  // Priori error estimate covariance matrix
  Eigen::MatrixXd P_pri;
  // Posteriori error estimate covariance matrix
  Eigen::MatrixXd P_post;

  // Kalman gain
  Eigen::MatrixXd K;

  // System dimensions
  int n;
  int p;

  // N-size identity
  Eigen::MatrixXd I;

  // Priori state
  Eigen::VectorXd x_pri;
  // Posteriori state
  Eigen::VectorXd x_post;
  //预测观测均值
  Eigen::VectorXd z_pri;

  //权重向量
  Eigen::VectorXd weights;

  //sigma点矩阵
  Eigen::MatrixXd sigma_points;
  //预测生成的sigma点
  Eigen::MatrixXd predicted_sigma_points;
  //预测的观测sigma点
  Eigen::MatrixXd predicted_observation_sigma_points;
};
}  // namespace rm_auto_aim

#endif  // ARMOR_PROCESSOR__KALMAN_FILTER_HPP_
