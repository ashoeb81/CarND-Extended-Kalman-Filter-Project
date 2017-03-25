#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_

#include "Eigen/Dense"
#include "tools.h"

class KalmanFilter {
public:

    // State vector.
    Eigen::VectorXd x_;

    // State covariance matrix.
    Eigen::MatrixXd P_;

    // State transition matrix.
    Eigen::MatrixXd F_;

    // Process covariance matrix.
    Eigen::MatrixXd Q_;

    // Measurement matrix.
    Eigen::MatrixXd H_;

    // Measurement covariance matrix.
    Eigen::MatrixXd R_;

    // Utilities for computing Jacobian and RMSE.
    Tools tools;

    /**
     * Constructor
     */
    KalmanFilter();

    /**
     * Destructor
     */
    virtual ~KalmanFilter();

    /**
     * Prediction Predicts the state and the state covariance
     * using the process model
     */
    void Predict();

    /**
     * Updates the state by using standard Kalman Filter equations
     * @param z The measurement at k+1
     * @param z_pred The measurement prediction at k+1
     */
    void Update(const Eigen::VectorXd &z, const Eigen::VectorXd &z_pred);


    /**
     * Updates Transition matrix based on elapsed time since last measurement.
     * @param dt The elapsed time (in seconds) since last measurement.
     */
    void UpdateFMatrix(float dt);

    /**
     * Updates Process covariance matrix based on elapsed time since last measurement.
     * @param dt The elapsed time (in seconds) since last measurement.
     * @param noise_ax Acceleration noise variance in x-direction.
     * @param noise_ay Acceleration noise variance in y-direction.
     */
    void UpdateQMatrix(float dt, float noise_ax, float noise_ay);


    /**
    * Projects state vector to radar measurement space
    * @param result Pointer to an Eigen::VectorXd that will populated with result.
    */
    void ProjectToRadarMeasurementSpace(Eigen::VectorXd *result);

    /**
    * Projects state vector to laser measurement space
    * @param result Pointer to an Eigen::VectorXd that will populated with result.
    */
    void ProjectToLaserMeasurementSpace(Eigen::VectorXd *result);

};

#endif /* KALMAN_FILTER_H_ */
