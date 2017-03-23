#include <iostream>
#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
    x_ = x_in;
    P_ = P_in;
    F_ = F_in;
    H_ = H_in;
    R_ = R_in;
    Q_ = Q_in;
}

void KalmanFilter::Predict() {
    x_ = F_ * x_;
    MatrixXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
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
    /**
    TODO:
      * update the state by using Extended Kalman Filter equations
    */
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

    Q_ << 0.25*dt_4*noise_ax, 0, 0.5*dt_3*noise_ax, 0,
        0, 0.25*dt_4*noise_ay, 0, 0.5*dt_3*noise_ay,
        0.5*dt_3*noise_ax, 0, dt_2*noise_ax, 0,
        0, 0.5*dt_3*noise_ay, 0, dt_2*noise_ay;

}

void KalmanFilter::PrintStateVector() {
    std::cout << x_;
}
