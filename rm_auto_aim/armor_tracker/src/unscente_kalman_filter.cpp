#include <cmath>
#include <Eigen/Dense>

#include "armor_tracker/unscente_kalman_filter.hpp"

namespace rm_auto_aim
{
    UnscentedKalmanFilter::UnscentedKalmanFilter(const VecVecFunc & f, const VecVecFunc & h,
    const VoidMatFunc & u_q, const VecMatFunc & u_r, const Eigen::MatrixXd & P0)
    :f(f),
    h(h),
    update_Q(u_q),
    update_R(u_r),
    P_post(P0),
    n(P0.rows()),
    I(Eigen::MatrixXd::Identity(n, n)),
    x_pri(n),
    x_post(n),
    weights(2*n + 1),
    sigma_points(n, n*2 + 1),
    predicted_sigma_points(n, n*2 + 1),
    predicted_observation_sigma_points(4, 2 * n + 1)
    {
        //生成权重向量的值
        weights(0) = 1.0 - n / 3.0;
        for (int i = 1; i < 2 * n + 1; ++i)
        {
            weights(i) = 1.0 / (2.0 * (n + 1.0));
        }

        //初始化sigma矩阵
        Eigen::MatrixXd sqrt_covariance = P_post.llt().matrixL();
        sigma_points.col(0) = x_post;
        for (int i = 0; i < n; ++i)
        {
            sigma_points.col(i + 1) = x_post + std::sqrt(n + 1.0) * sqrt_covariance.col(i);
            sigma_points.col(i + 1 + n) = x_post - std::sqrt(n + 1.0) * sqrt_covariance.col(i);
        }
    }

    viod UnscentedKalmanFilter::setState(const Eigen::VectorXd & x0)
    {
        x_post = x0;
        //设置sigma矩阵
        Eigen::MatrixXd sqrt_covariance = P_post.llt().matrixL();
        sigma_points.col(0) = x_post;
        for (int i = 0; i < n; ++i)
        {
            sigma_points.col(i + 1) = x_post + std::sqrt(n + 1.0) * sqrt_covariance.col(i);
            sigma_points.col(i + 1 + n) = x_post - std::sqrt(n + 1.0) * sqrt_covariance.col(i);
        }
    }

    Eigen::VectorXd UnscentedKalmanFilter::predict()
    {
        Q = update_Q()
        for (int i = 0; i < 2 * n + 1; ++i)
        {
            predicted_sigma_points.col(i) = f(sigma_points.col(i));
        }
        x_pri = predicted_sigma_points * weights;
        for (int i = 0; i < 2 * n + 1; ++i)
        {
            VectorXd residual = predicted_sigma_points.col(i) - x_pri;
            P_pri += weights(i) * residual * residual.transpose();
        }
        P_pri += Q;

        // 通过观测函数预测观测sigma点
        for (int i = 0; i < 2 * n + 1; ++i)
        {
            predicted_observation_sigma_points.col(i) = h(predicted_sigma_points.col(i));
        }

        // 预测观测均值和协方差
        x_pri = predicted_observation_sigma_points * weights;

        x_post = x_pri;
        P_post = P_pri;

        return x_pri;


    }

    Eigen::VectorXd UnscentedKalmanFilter::update(const Eigen::VectorXd & z)
    {
        R = update_R(z)
        //更新观测协方差
        Eigen::MatrixXd predicted_observation_covariance(p, p);
        predicted_observation_covariance.fill(0.0);
        for (int i = 0; i < 2 * n + 1; ++i)
        {
            Eigen::VectorXd residual = predicted_observation_sigma_points.col(i) - x_pri;
            predicted_observation_covariance += weights(i) * residual * residual.transpose();
        }
        predicted_observation_covariance += R;

        // 计算预测和观测的交叉协方差
        Eigen::MatrixXd cross_covariance(n, p);
        cross_covariance.fill(0.0);
        for (int i = 0; i < 2 * n + 1; ++i)
        {
            Eigen::VectorXd state_residual = predicted_sigma_points.col(i) - predicted_state_mean;
            Eigen::VectorXd observation_residual = predicted_observation_sigma_points.col(i) - x_pri;
            cross_covariance += weights(i) * state_residual * observation_residual.transpose();
        }

        // 计算卡尔曼增益
        K = cross_covariance * predicted_observation_covariance.inverse();

        // 更新状态均值和协方差
        Eigen::VectorXd observation = observationFunction(state_mean);
        Eigen::VectorXd innovation = observation - x_pri;
        x_post = predicted_state_mean + kalman_gain * innovation;
        P_post = predicted_state_covariance - kalman_gain * predicted_observation_covariance * kalman_gain.transpose();

        return x_post;
    }
}
