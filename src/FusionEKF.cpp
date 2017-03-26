#include "FusionEKF.h"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
    // Initial FusionEKF state.
    is_initialized_ = false;

    // Initial timestamp value.
    previous_timestamp_ = 0;

    // Initializing measurement covariance matrices.
    R_laser_ = MatrixXd(2, 2);
    R_radar_ = MatrixXd(3, 3);

    // Measurement covariance matrix - laser
    R_laser_ << 0.0225, 0,
            0, 0.0225;

    // Measurement covariance matrix - radar
    R_radar_ << 0.09, 0, 0,
            0, 0.0009, 0,
            0, 0, 0.09;

    // Initialize Laser state to measurement matrix.
    H_laser_ = MatrixXd(2, 4);
    H_laser_ << 1, 0, 0, 0,
            0, 1, 0, 0;

    // Instantiate kalman filter state vector.
    ekf_.x_ = VectorXd(4);

    // Instantiate kalman filter state transition matrix.
    ekf_.F_ = MatrixXd(4, 4);

    // Initialize kalman filter state covariance matrix.
    // We place more confidence in our initial position estimate
    // and little confidence in our initial velocity estimate.
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 10, 0,
            0, 0, 0, 10;

    // Instantiate kalman filter state noise covariance matrix.
    ekf_.Q_ = MatrixXd(4, 4);

    // Initialize process noise variables.
    noise_ax = 9;
    noise_ay = 9;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


    /*****************************************************************************
     *  Initialization
     ****************************************************************************/
    if (!is_initialized_) {
        // Check if the first measurement is a radar or laser measurement.
        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
            // Convert polar coordinates of object (rho, tehta) to euclidean coordinates (x,y).
            ekf_.x_ << measurement_pack.raw_measurements_(0) * cos(measurement_pack.raw_measurements_(1)),  // px
                    measurement_pack.raw_measurements_(0) * sin(measurement_pack.raw_measurements_(1)),  // py
                    measurement_pack.raw_measurements_(2) * cos(measurement_pack.raw_measurements_(1)),  // vx
                    measurement_pack.raw_measurements_(2) * cos(measurement_pack.raw_measurements_(1));  // vy
        } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
            // Laser measurements are in euclidean coordinates.
            ekf_.x_ << measurement_pack.raw_measurements_(0),  // px
                    measurement_pack.raw_measurements_(1),  // py
                    0.0,  // vx
                    0.0;  // vy
        }
        // done initializing, no need to predict or update
        previous_timestamp_ = measurement_pack.timestamp_;
        is_initialized_ = true;
        return;
    }

    /*****************************************************************************
    *  Prediction
    ****************************************************************************/
    // Compute the elapsed time between measurements in seconds.
    float dt = (measurement_pack.timestamp_ - previous_timestamp_) / (1000000.0);
    previous_timestamp_ = measurement_pack.timestamp_;

    // If dt is too small we skip making a prediction to avoid the numerical errors
    // associated with updating the F and Q matrices which depend directly on dt.
    if (dt > 0.001) {
        // Update state transition matrix given computed elapsed time.
        ekf_.UpdateFMatrix(dt);

        // Update process noise covariance given computed elapsed time.
        ekf_.UpdateQMatrix(dt, noise_ax, noise_ay);

        // Predict new state
        ekf_.Predict();
    }

    /*****************************************************************************
    *  Update
    ****************************************************************************/
    // Vector to hold measurement prediction
    VectorXd z_pred(3);
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        // Assign appropriate measurement covariance matrix for radar.
        ekf_.R_ = R_radar_;
        // Compute linearized measurement matrix.
        ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
        // Project filter state to radar measurement state.
        ekf_.ProjectToRadarMeasurementSpace(&z_pred);
        // If we succeeded in computing H_, then update kalman filter state otherwise skip.
        if (!ekf_.H_.isZero() && !z_pred.isZero()) {
            // Update filter state based on measurement.
            ekf_.Update(measurement_pack.raw_measurements_, z_pred);
        }
    } else {
        // Assign appropriate measurement matrix for laser.
        ekf_.H_ = H_laser_;
        // Assign appropriate measurement covariance matrix for laser.
        ekf_.R_ = R_laser_;
        // Project filter state to laser measurement state
        ekf_.ProjectToLaserMeasurementSpace(&z_pred);
        // Update state based on measurement.
        ekf_.Update(measurement_pack.raw_measurements_, z_pred);
    }

    // Print the output
    cout << "x_ = " << ekf_.x_ << endl;
    cout << "P_ = " << ekf_.P_ << endl;
}
