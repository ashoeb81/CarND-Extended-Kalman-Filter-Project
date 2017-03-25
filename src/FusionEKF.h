#ifndef FusionEKF_H_
#define FusionEKF_H_

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>
#include "kalman_filter.h"
#include "tools.h"

class FusionEKF {
public:
    /**
    * Constructor.
    */
    FusionEKF();

    /**
    * Destructor.
    */
    virtual ~FusionEKF();

    /**
    * Run the whole flow of the Kalman Filter from here.
    */
    void ProcessMeasurement(const MeasurementPackage &measurement_pack);

    /**
    * Kalman Filter update and prediction math lives in here.
    */
    KalmanFilter ekf_;

private:
    // Check whether the tracking toolbox was initiallized or not (first measurement).
    bool is_initialized_;

    // Previous timestamp
    long previous_timestamp_;

    // tool object used to compute Jacobian and RMSE.
    Tools tools;

    // Measurement covariance matrices.
    Eigen::MatrixXd R_laser_;
    Eigen::MatrixXd R_radar_;

    // Measurement matrices.
    Eigen::MatrixXd H_laser_;

    // State noise variables.
    float noise_ax;
    float noise_ay;
};

#endif /* FusionEKF_H_ */
