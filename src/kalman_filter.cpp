#include <iostream>
#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::pow;
using std::sqrt;
using std::atan2;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Predict() {
    x_ = F_ * x_;
    MatrixXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z, const VectorXd &z_pred) {
    // Standard kalman filter update equations.
    VectorXd y = z - z_pred;
    MatrixXd Ht = H_.transpose();
    MatrixXd PHt = P_ * Ht;
    MatrixXd S = H_ * PHt + R_;
    MatrixXd Si = S.inverse();
    MatrixXd K = PHt * Si;

    //new estimate
    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateFMatrix(float dt) {

    F_ << 1, 0, dt, 0,
          0, 1, 0, dt,
          0, 0, 1, 0,
          0, 0, 0, 1;

}

void KalmanFilter::UpdateQMatrix(float dt, float noise_ax, float noise_ay) {

    float dt_2 = pow(dt, 2);
    float dt_3 = pow(dt, 3);
    float dt_4 = pow(dt, 4);

    Q_ << 0.25 * dt_4 * noise_ax, 0, 0.5 * dt_3 * noise_ax, 0,
          0, 0.25 * dt_4 * noise_ay, 0, 0.5 * dt_3 * noise_ay,
          0.5 * dt_3 * noise_ax, 0, dt_2 * noise_ax, 0,
          0, 0.5 * dt_3 * noise_ay, 0, dt_2 * noise_ay;

}

void KalmanFilter::ProjectToRadarMeasurementSpace(VectorXd *result) {

    float px = x_(0);
    float py = x_(1);
    float vx = x_(2);
    float vy = x_(3);
    float norm = sqrt(pow(px, 2) + pow(py, 2));

    // Make sure norm is not small to avoid numerical error in division.
    if (norm < 0.0001) {
        (*result).setZero();
    } else {
        (*result)(0) = norm;
        (*result)(1) = atan2(py, px);
        (*result)(2) = (px * vx + py * vy) / norm;
    }

}


void KalmanFilter::ProjectToLaserMeasurementSpace(VectorXd *result) {
    *result = H_ * x_;
}