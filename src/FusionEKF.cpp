#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
    // set FusionEKF state.
    is_initialized_ = false;

    // set initial timestamp value.
    previous_timestamp_ = 0;

    // initializing matrices
    R_laser_ = MatrixXd(2, 2);
    R_radar_ = MatrixXd(3, 3);
    H_laser_ = MatrixXd(2, 4);
    Hj_ = MatrixXd(3, 4);

    //measurement covariance matrix - laser
    R_laser_ << 0.0225, 0,
            0, 0.0225;

    //measurement covariance matrix - radar
    R_radar_ << 0.09, 0, 0,
            0, 0.0009, 0,
            0, 0, 0.09;

    // initialize kalman filter matrices.
    ekf_.x_ = VectorXd(4);
    ekf_.F_ = MatrixXd(4,4);
    ekf_.P_ = MatrixXd(4,4);
    ekf_.Q_ = MatrixXd(4,4);

    // initialize process noise variables.
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
        /**
         * TODO:
         * Initialize the state ekf_.x_ with the first measurement.
         * Create the covariance matrix.
         * Remember: you'll need to convert radar from polar to cartesian coordinates.
        */
        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
            ekf_.x_ << cos(measurement_pack.raw_measurements_(0)),
                    sin(measurement_pack.raw_measurements_(1)),
                    0.0,
                    0.0;
        } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
            ekf_.x_ << measurement_pack.raw_measurements_(0),
                    measurement_pack.raw_measurements_(1),
                    0.0,
                    0.0;
        }

        ekf_.PrintStateVector();

        // done initializing, no need to predict or update
        previous_timestamp_ = measurement_pack.timestamp_;
        is_initialized_ = true;
        return;
    }

    /*****************************************************************************
    *  Prediction
    ****************************************************************************/

    /**
    TODO:
    * Update the state transition matrix F according to the new elapsed time.
     - Time is measured in seconds.
    * Update the process noise covariance matrix.
    * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
    */
    float dt = (measurement_pack.timestamp_ - previous_timestamp_)/(1000000.0);
    previous_timestamp_ =  measurement_pack.timestamp_;

    // Update state transition and process covariance matrices based on
    // elapsed time.
    ekf_.UpdateFMatrix(dt);
    ekf_.UpdateQMatrix(dt, noise_ax, noise_ay);

    // Predict new state
    ekf_.Predict();

    /*****************************************************************************
    *  Update
    ****************************************************************************/

    /**
    TODO:
    * Use the sensor type to perform the update step.
    * Update the state and covariance matrices.
    */

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        // Radar updates
    } else {
        ekf_.H_ = H_laser_;
        ekf_.R_ = R_laser_;
        ekf_.Update(measurement_pack.raw_measurements_);
    }

    // print the output
    cout << "x_ = " << ekf_.x_ << endl;
    cout << "P_ = " << ekf_.P_ << endl;
}
