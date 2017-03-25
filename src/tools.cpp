#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    // Initialize vector to accumulate rmse.
    VectorXd rmse = VectorXd::Zero(4);
    for(int i = 0; i < estimations.size(); ++i) {
        // Compute residual for sample i.
        VectorXd residual = ground_truth[i] - estimations[i];
        // Square residual.
        residual = residual.array() * residual.array();
        // Accumulate.
        rmse += residual;
    }
    // Compute Mean Square Error (MSE).
    rmse /= estimations.size();
    // Compute square root of mean squared error (RMSE).
    return rmse.array().sqrt();
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state) {

    // Instantiate matrix to hold jacobian result.
    MatrixXd Hj(3, 4);

    //  Recover state parameters
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);

    // Compute repeated terms in jacobian.
    float c1 = px * px + py * py;
    float c2 = sqrt(c1);
    float c3 = (c1 * c2);

    // Check division by zero
    if (fabs(c1) < 0.0001) {
        // Return an all-zeros matrix in the case of a division by zero.
        return Hj.setZero();
    }

    // Compute the Jacobian matrix
    Hj << (px / c2), (py / c2), 0, 0,
          -(py / c1), (px / c1), 0, 0,
          py * (vx * py - vy * px) / c3, px * (px * vy - py * vx) / c3, px / c2, py / c2;

    return Hj;

}
