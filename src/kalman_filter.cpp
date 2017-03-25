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

void KalmanFilter::Update(const VectorXd &z) {
    // Standard kalman filter update equations.
    VectorXd z_pred = H_ * x_;
    VectorXd y = z - z_pred;
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;

    //new estimate
    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    // Project filter state into radar measurement space.
    VectorXd zpred = ProjectToMeasurementSpace();
    VectorXd y = z - zpred;
    MatrixXd Ht = H_.transpose();  // Note: H_ has already been linearized.
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
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

void KalmanFilter::PrintStateVector() {
    std::cout << x_ << std::endl;
}

VectorXd KalmanFilter::ProjectToMeasurmentSpace() {

    VectorXd measurement_vector(3);

    float px = x_(0);
    float py = x_(1);
    float vx = x_(2);
    float vy = x_(3);
    float norm = sqrt(pow(px, 2) + pow(py, 2));

    measurement_vector(0) = norm;
    measurement_vector(1) = atan2(py, px);
    measurement_vector(2) = (px * vx + py * vy) / norm;

    return measurement_vector;

}