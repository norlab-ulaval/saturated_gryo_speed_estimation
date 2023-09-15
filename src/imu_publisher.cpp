#include <rclcpp/rclcpp.hpp>
#include <Eigen/Core>
#include <fstream>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <steam.hpp>

const int SEC_INDEX = 0;
const int NANOSEC_INDEX = 1;
const int ANGULAR_VELOCITY_X_INDEX = 8;
const int ANGULAR_VELOCITY_Y_INDEX = 9;
const int ANGULAR_VELOCITY_Z_INDEX = 10;
const int ANGULAR_VELOCITY_COVARIANCE_INDEX = 11;
const int LINEAR_ACCELERATION_X_INDEX = 12;
const int LINEAR_ACCELERATION_Y_INDEX = 13;
const int LINEAR_ACCELERATION_Z_INDEX = 14;

class IMUPublisher : public rclcpp::Node
{
public:
    IMUPublisher():
            Node("imu_publisher_node")
    {
        this->declare_parameter<std::string>("imu_measurements_file_name", "");
        std::string imuMeasurementsFileName = this->get_parameter("imu_measurements_file_name").as_string();
        this->declare_parameter<double>("saturation_point", 12);
        saturationPoint = this->get_parameter("saturation_point").as_double();
        imuToBody = Eigen::Vector3d(-0.022, -0.085, 0.055);
        if(imuMeasurementsFileName.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "The imu_measurements_file_name parameter was not set, exiting...");
            exit(1);
        }

        measurementIndex = 0;

        std::vector<double> angularVelocitiesX, angularVelocitiesY, angularVelocitiesZ, linearAccelerationsX, linearAccelerationsY, linearAccelerationsZ;
        std::vector<Eigen::Matrix3d> angularVelocityCovariancesXYZ;
        loadImuMeasurements(imuMeasurementsFileName, timeStamps, angularVelocitiesX, angularVelocitiesY, angularVelocitiesZ, angularVelocityCovariancesXYZ, linearAccelerationsX, linearAccelerationsY, linearAccelerationsZ);

        Eigen::Matrix<double, 6, 1> powerSpectralDensityMatrixDiagonal = Eigen::Matrix<double, 6, 1>::Ones();
        powerSpectralDensityMatrixDiagonal.bottomRows<3>() *= 1e6;
        steam::traj::const_acc::Interface traj(powerSpectralDensityMatrixDiagonal);
        steam::OptimizationProblem problem;
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> measurements = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(6, timeStamps.size());
        Eigen::Vector3d rotationAxis = Eigen::Vector3d(0, 0, 1);
        for(size_t i = 0; i < timeStamps.size(); ++i)
        {
            Eigen::Matrix<double, 6, 1> measurement = Eigen::Matrix<double, 6, 1>::Zero();
            Eigen::Matrix<double, 6, 6> informationMatrix = Eigen::Matrix<double, 6, 6>::Identity() * 1e6;
            measurement(3) = angularVelocitiesX[i];
            measurement(4) = angularVelocitiesY[i];
            measurement(5) = angularVelocitiesZ[i];
            informationMatrix(3, 3) = 3.648e4;
            informationMatrix(4, 4) = 3.648e4;
            informationMatrix(5, 5) = 3.648e4;

            bool validAngularSpeed = true;
            if(std::fabs(angularVelocitiesX[i]) > saturationPoint)
            {
                Eigen::Vector3d leverArm = imuToBody.dot(rotationAxis) * rotationAxis - imuToBody;
                Eigen::Matrix3d imuToRotationalFrame = Eigen::Matrix3d::Zero();
                imuToRotationalFrame.row(0) = leverArm.normalized();
                imuToRotationalFrame.row(1) = rotationAxis.cross(leverArm.normalized());
                imuToRotationalFrame.row(2) = rotationAxis;
                Eigen::Vector3d linearAcceleration(linearAccelerationsX[i], linearAccelerationsY[i], linearAccelerationsZ[i]);
                Eigen::Vector3d linearAccelerationInRotationalFrame = imuToRotationalFrame * linearAcceleration;
                double angularSpeedSquared = linearAccelerationInRotationalFrame[0] / leverArm.norm();
                if(angularSpeedSquared > std::pow(angularVelocitiesY[i], 2) + std::pow(angularVelocitiesZ[i], 2))
                {
                    measurement(3) = std::sqrt(angularSpeedSquared - std::pow(angularVelocitiesY[i], 2) - std::pow(angularVelocitiesZ[i], 2));
                    measurement(3) = std::max(measurement(3), std::fabs(angularVelocitiesX[i]));
                    measurement(3) *= angularVelocitiesX[i] / std::fabs(angularVelocitiesX[i]);
                    informationMatrix(3, 3) = 3.648;
                }
                else
                {
                    informationMatrix(3, 3) = 1e-6;
                    validAngularSpeed = false;
                }
            }
            else if(std::fabs(angularVelocitiesY[i]) > saturationPoint)
            {
                Eigen::Vector3d leverArm = imuToBody.dot(rotationAxis) * rotationAxis - imuToBody;
                Eigen::Matrix3d imuToRotationalFrame = Eigen::Matrix3d::Zero();
                imuToRotationalFrame.row(0) = leverArm.normalized();
                imuToRotationalFrame.row(1) = rotationAxis.cross(leverArm.normalized());
                imuToRotationalFrame.row(2) = rotationAxis;
                Eigen::Vector3d linearAcceleration(linearAccelerationsX[i], linearAccelerationsY[i], linearAccelerationsZ[i]);
                Eigen::Vector3d linearAccelerationInRotationalFrame = imuToRotationalFrame * linearAcceleration;
                double angularSpeedSquared = linearAccelerationInRotationalFrame[0] / leverArm.norm();
                if(angularSpeedSquared > std::pow(angularVelocitiesX[i], 2) + std::pow(angularVelocitiesZ[i], 2))
                {
                    measurement(4) = std::sqrt(angularSpeedSquared - std::pow(angularVelocitiesX[i], 2) - std::pow(angularVelocitiesZ[i], 2));
                    measurement(4) = std::max(measurement(4), std::fabs(angularVelocitiesY[i]));
                    measurement(4) *= angularVelocitiesY[i] / std::fabs(angularVelocitiesY[i]);
                    informationMatrix(4, 4) = 3.648;
                }
                else
                {
                    informationMatrix(4, 4) = 1e-6;
                    validAngularSpeed = false;
                }
            }
            else if(std::fabs(angularVelocitiesZ[i]) > saturationPoint)
            {
                Eigen::Vector3d leverArm = imuToBody.dot(rotationAxis) * rotationAxis - imuToBody;
                Eigen::Matrix3d imuToRotationalFrame = Eigen::Matrix3d::Zero();
                imuToRotationalFrame.row(0) = leverArm.normalized();
                imuToRotationalFrame.row(1) = rotationAxis.cross(leverArm.normalized());
                imuToRotationalFrame.row(2) = rotationAxis;
                Eigen::Vector3d linearAcceleration(linearAccelerationsX[i], linearAccelerationsY[i], linearAccelerationsZ[i]);
                Eigen::Vector3d linearAccelerationInRotationalFrame = imuToRotationalFrame * linearAcceleration;
                double angularSpeedSquared = linearAccelerationInRotationalFrame[0] / leverArm.norm();
                if(angularSpeedSquared > std::pow(angularVelocitiesX[i], 2) + std::pow(angularVelocitiesY[i], 2))
                {
                    measurement(5) = std::sqrt(angularSpeedSquared - std::pow(angularVelocitiesX[i], 2) - std::pow(angularVelocitiesY[i], 2));
                    measurement(5) = std::max(measurement(5), std::fabs(angularVelocitiesZ[i]));
                    measurement(5) *= angularVelocitiesZ[i] / std::fabs(angularVelocitiesZ[i]);
                    informationMatrix(5, 5) = 3.648;
                }
                else
                {
                    informationMatrix(5, 5) = 1e-6;
                    validAngularSpeed = false;
                }
            }
            measurements.col(i) = measurement;
            if(validAngularSpeed)
            {
                rotationAxis = measurement.bottomRows(3).normalized();
            }

            std::shared_ptr<steam::se3::SE3StateVar> T_vi = steam::se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(pose));
            std::shared_ptr<steam::vspace::VSpaceStateVar<6>> w_iv_inv = steam::vspace::VSpaceStateVar<6>::MakeShared(measurement);
            std::shared_ptr<steam::vspace::VSpaceStateVar<6>> dw_iv_inv = steam::vspace::VSpaceStateVar<6>::MakeShared(Eigen::Matrix<double, 6, 1>::Zero());
            if(i == 0)
            {
                T_vi->locked() = true;
            }
            traj.add(steam::traj::Time(timeStamps[i] - timeStamps[0]), T_vi, w_iv_inv, dw_iv_inv);
            problem.addStateVariable(T_vi);
            problem.addStateVariable(w_iv_inv);
            problem.addStateVariable(dw_iv_inv);

            std::shared_ptr<steam::vspace::VSpaceErrorEvaluator<6>> errorFunc = steam::vspace::VSpaceErrorEvaluator<6>::MakeShared(w_iv_inv, measurement);
            std::shared_ptr<steam::StaticNoiseModel<6>> noiseModel = steam::StaticNoiseModel<6>::MakeShared(informationMatrix, steam::NoiseType::INFORMATION);
            std::shared_ptr<steam::L2LossFunc> lossFunc = steam::L2LossFunc::MakeShared();
            std::shared_ptr<steam::WeightedLeastSqCostTerm<6>> costTerm = steam::WeightedLeastSqCostTerm<6>::MakeShared(errorFunc, noiseModel, lossFunc);
            problem.addCostTerm(costTerm);
        }
        traj.addPriorCostTerms(problem);
        steam::GaussNewtonSolver::Params params;
        params.verbose = true;
        steam::GaussNewtonSolver solver(problem, params);
        solver.optimize();

        steam::Covariance covariance(problem);
        for(size_t i = 0; i < timeStamps.size(); ++i)
        {
            if(!rclcpp::ok())
            {
                return;
            }
            angularVelocities.emplace_back(traj.getVelocityInterpolator(timeStamps[i] - timeStamps[0])->value().matrix().bottomRows<3>());
            if(i == 0)
            {
                angularVelocityCovariances.emplace_back(angularVelocityCovariancesXYZ[i]);
            }
            else
            {
                angularVelocityCovariances.emplace_back(traj.getCovariance(covariance, timeStamps[i] - timeStamps[0]).block<3, 3>(9, 9));
            }
            RCLCPP_INFO_STREAM(this->get_logger(), "Progression: " << i * 100.0 / (timeStamps.size() - 1) << "%");
        }

        std_msgs::msg::String stringMessage;
        stringMessage.data = "Initialization done";
        this->create_publisher<std_msgs::msg::String>("imu_publisher_status", 1)->publish(stringMessage);

        publisher = this->create_publisher<sensor_msgs::msg::Imu>("imu_out", 10);
        subscription = this->create_subscription<sensor_msgs::msg::Imu>("imu_in", 10, std::bind(&IMUPublisher::imuCallback, this, std::placeholders::_1));
    }

private:
    void
    loadImuMeasurements(const std::string& imuMeasurementsFileName, std::vector<double>& timeStamps, std::vector<double>& angularVelocitiesX,
                        std::vector<double>& angularVelocitiesY, std::vector<double>& angularVelocitiesZ, std::vector<Eigen::Matrix3d>& angularVelocityCovariancesXYZ,
                        std::vector<double>& linearAccelerationsX, std::vector<double>& linearAccelerationsY, std::vector<double>& linearAccelerationsZ)
    {
        std::ifstream imuMeasurementsFile(imuMeasurementsFileName);
        std::string line;
        std::getline(imuMeasurementsFile, line); // skip header
        while(std::getline(imuMeasurementsFile, line))
        {
            double timeStamp = 0;
            double angularVelocityX, angularVelocityY, angularVelocityZ, linearAccelerationX, linearAccelerationY, linearAccelerationZ = 0;
            Eigen::Matrix3d angularVelocityCovariance;
            size_t tokenIndex = 0;
            int currentCommaPosition = -1;
            int nextCommaPosition = line.find(',', currentCommaPosition + 1);
            bool lastTokenWasRead = false;
            while(!lastTokenWasRead)
            {
                std::string token = line.substr(currentCommaPosition + 1, nextCommaPosition - currentCommaPosition - 1);

                if(tokenIndex == SEC_INDEX)
                {
                    timeStamp += std::stod(token);
                }
                else if(tokenIndex == NANOSEC_INDEX)
                {
                    timeStamp += std::stod(token) / 1e9;
                }
                else if(tokenIndex == ANGULAR_VELOCITY_X_INDEX)
                {
                    angularVelocityX = std::stof(token);
                }
                else if(tokenIndex == ANGULAR_VELOCITY_Y_INDEX)
                {
                    angularVelocityY = std::stof(token);
                }
                else if(tokenIndex == ANGULAR_VELOCITY_Z_INDEX)
                {
                    angularVelocityZ = std::stof(token);
                }
                else if(tokenIndex == ANGULAR_VELOCITY_COVARIANCE_INDEX)
                {
                    angularVelocityCovariance = stringToMatrix(token);
                }
                else if(tokenIndex == LINEAR_ACCELERATION_X_INDEX)
                {
                    linearAccelerationX = std::stof(token);
                }
                else if(tokenIndex == LINEAR_ACCELERATION_Y_INDEX)
                {
                    linearAccelerationY = std::stof(token);
                }
                else if(tokenIndex == LINEAR_ACCELERATION_Z_INDEX)
                {
                    linearAccelerationZ = std::stof(token);
                }

                tokenIndex += 1;
                currentCommaPosition = nextCommaPosition;
                nextCommaPosition = line.find(',', currentCommaPosition + 1);

                if(currentCommaPosition == std::string::npos)
                {
                    lastTokenWasRead = true;
                }
            }

            timeStamps.push_back(timeStamp);
            angularVelocitiesX.push_back(angularVelocityX);
            angularVelocitiesY.push_back(angularVelocityY);
            angularVelocitiesZ.push_back(angularVelocityZ);
            angularVelocityCovariancesXYZ.push_back(angularVelocityCovariance);
            linearAccelerationsX.push_back(linearAccelerationX);
            linearAccelerationsY.push_back(linearAccelerationY);
            linearAccelerationsZ.push_back(linearAccelerationZ);
        }
    }

    Eigen::Matrix3d stringToMatrix(std::string matrixString)
    {
        Eigen::Matrix3d matrix = Eigen::Matrix3d::Zero();
        matrixString = matrixString.substr(1, matrixString.length() - 2);
        size_t tokenIndex = 0;
        int currentSpacePosition = -1;
        int nextSpacePosition = matrixString.find(' ', currentSpacePosition + 1);
        bool lastTokenWasRead = false;
        while(!lastTokenWasRead)
        {
            std::string token = matrixString.substr(currentSpacePosition + 1, nextSpacePosition - currentSpacePosition - 1);
            if(!token.empty())
            {
                matrix(int(tokenIndex / 3), tokenIndex % 3) = std::stof(token);
                tokenIndex += 1;
            }

            currentSpacePosition = nextSpacePosition;
            nextSpacePosition = matrixString.find(' ', currentSpacePosition + 1);

            if(currentSpacePosition == std::string::npos)
            {
                lastTokenWasRead = true;
            }
        }
        return matrix;
    }

    void imuCallback(sensor_msgs::msg::Imu msg)
    {
        double timeStamp = rclcpp::Time(msg.header.stamp).seconds();
        while(measurementIndex + 1 < timeStamps.size() && std::fabs(timeStamps[measurementIndex + 1] - timeStamp) < std::fabs(timeStamps[measurementIndex] - timeStamp))
        {
            measurementIndex += 1;
        }
        msg.angular_velocity = vectorToVector3(angularVelocities[measurementIndex]);
        msg.angular_velocity_covariance = matrixToDoubleArray(angularVelocityCovariances[measurementIndex]);
        publisher->publish(msg);
    }

    geometry_msgs::msg::Vector3 vectorToVector3(const Eigen::Vector3d& vector)
    {
        geometry_msgs::msg::Vector3 vector3;
        vector3.x = vector(0);
        vector3.y = vector(1);
        vector3.z = vector(2);
        return vector3;
    }

    std::array<double, 9> matrixToDoubleArray(const Eigen::Matrix3d& matrix)
    {
        std::array<double, 9> array;
        for(size_t i = 0; i < 3; ++i)
        {
            for(size_t j = 0; j < 3; ++j)
            {
                array[i * 3 + j] = matrix(i, j);
            }
        }
        return array;
    }

    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr publisher;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subscription;
    std::vector<double> timeStamps;
    std::vector<Eigen::Vector3d> angularVelocities;
    std::vector<Eigen::Matrix3d> angularVelocityCovariances;
    size_t measurementIndex;
    double saturationPoint;
    Eigen::Vector3d imuToBody;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<IMUPublisher>());
    rclcpp::shutdown();
    return 0;
}
