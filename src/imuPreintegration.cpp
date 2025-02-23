#include "utility.hpp"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

class TransformFusion : public ParamServer
{
public:
    std::mutex mtx;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subImuOdometry;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subLaserOdometry;

    rclcpp::CallbackGroup::SharedPtr callbackGroupImuOdometry;
    rclcpp::CallbackGroup::SharedPtr callbackGroupLaserOdometry;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubImuOdometry;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubImuPath;

    Eigen::Isometry3d lidarOdomAffine;
    Eigen::Isometry3d imuOdomAffineFront;
    Eigen::Isometry3d imuOdomAffineBack;

    std::shared_ptr<tf2_ros::Buffer> tfBuffer;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster;
    std::shared_ptr<tf2_ros::TransformListener> tfListener;
    tf2::Stamped<tf2::Transform> lidar2Baselink;

    double lidarOdomTime = -1;
    deque<nav_msgs::msg::Odometry> imuOdomQueue;

    TransformFusion(const rclcpp::NodeOptions & options) : ParamServer("lio_sam_transformFusion", options)
    {
        tfBuffer = std::make_shared<tf2_ros::Buffer>(get_clock());
        tfListener = std::make_shared<tf2_ros::TransformListener>(*tfBuffer);

        callbackGroupImuOdometry = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        callbackGroupLaserOdometry = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        auto imuOdomOpt = rclcpp::SubscriptionOptions();
        imuOdomOpt.callback_group = callbackGroupImuOdometry;
        auto laserOdomOpt = rclcpp::SubscriptionOptions();
        laserOdomOpt.callback_group = callbackGroupLaserOdometry;

        subLaserOdometry = create_subscription<nav_msgs::msg::Odometry>(
            "lio_sam/mapping/odometry", qos,
            std::bind(&TransformFusion::lidarOdometryHandler, this, std::placeholders::_1),
            laserOdomOpt);
        subImuOdometry = create_subscription<nav_msgs::msg::Odometry>(
            odomTopic+"_incremental", qos_imu,
            std::bind(&TransformFusion::imuOdometryHandler, this, std::placeholders::_1),
            imuOdomOpt);

        pubImuOdometry = create_publisher<nav_msgs::msg::Odometry>(odomTopic, qos_imu);
        pubImuPath = create_publisher<nav_msgs::msg::Path>("lio_sam/imu/path", qos);

        tfBroadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(this);
    }

    Eigen::Isometry3d odom2affine(nav_msgs::msg::Odometry odom)
    {
        tf2::Transform t;
        tf2::fromMsg(odom.pose.pose, t);
        return tf2::transformToEigen(tf2::toMsg(t));
    }

    void lidarOdometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

        lidarOdomAffine = odom2affine(*odomMsg);

        lidarOdomTime = stamp2Sec(odomMsg->header.stamp);
    }

    void imuOdometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

        imuOdomQueue.push_back(*odomMsg);

        // get latest odometry (at current IMU stamp)
        if (lidarOdomTime == -1)
            return;
        while (!imuOdomQueue.empty())
        {
            if (stamp2Sec(imuOdomQueue.front().header.stamp) <= lidarOdomTime)
                imuOdomQueue.pop_front();
            else
                break;
        }
        Eigen::Isometry3d imuOdomAffineFront = odom2affine(imuOdomQueue.front());
        Eigen::Isometry3d imuOdomAffineBack = odom2affine(imuOdomQueue.back());
        Eigen::Isometry3d imuOdomAffineIncre = imuOdomAffineFront.inverse() * imuOdomAffineBack;
        Eigen::Isometry3d imuOdomAffineLast = lidarOdomAffine * imuOdomAffineIncre;
        auto t = tf2::eigenToTransform(imuOdomAffineLast);
        tf2::Stamped<tf2::Transform> tCur;
        tf2::convert(t, tCur);

        // publish latest odometry
        nav_msgs::msg::Odometry laserOdometry = imuOdomQueue.back();
        laserOdometry.pose.pose.position.x = t.transform.translation.x;
        laserOdometry.pose.pose.position.y = t.transform.translation.y;
        laserOdometry.pose.pose.position.z = t.transform.translation.z;
        laserOdometry.pose.pose.orientation = t.transform.rotation;
        pubImuOdometry->publish(laserOdometry);

        // publish tf
        // if(lidarFrame != "base_link_slam_liosam")
        if(true)
        {
            // try
            // {
            //     tf2::fromMsg(tfBuffer->lookupTransform(
            //         lidarFrame, "base_link_slam_liosam", rclcpp::Time(0)), lidar2Baselink);
            // }
            // catch (tf2::TransformException ex)
            // {
            //     RCLCPP_ERROR(get_logger(), "%s", ex.what());
            // }
            geometry_msgs::msg::TransformStamped tf_unit;
            tf_unit.transform.rotation.w = 1;
            tf2::fromMsg(tf_unit, lidar2Baselink);
            tf2::Stamped<tf2::Transform> tb(
                tCur * lidar2Baselink, tf2_ros::fromMsg(odomMsg->header.stamp), "odom");
            tCur = tb;
        }
        geometry_msgs::msg::TransformStamped ts;
        tf2::convert(tCur, ts);
        ts.child_frame_id = "base_link_slam_liosam";
        tfBroadcaster->sendTransform(ts);

        // publish IMU path
        static nav_msgs::msg::Path imuPath;
        static double last_path_time = -1;
        double imuTime = stamp2Sec(imuOdomQueue.back().header.stamp);
        if (imuTime - last_path_time > 0.1)
        {
            last_path_time = imuTime;
            geometry_msgs::msg::PoseStamped pose_stamped;
            pose_stamped.header.stamp = imuOdomQueue.back().header.stamp;
            pose_stamped.header.frame_id = "odom";
            pose_stamped.pose = laserOdometry.pose.pose;
            imuPath.poses.push_back(pose_stamped);
            while(!imuPath.poses.empty() && stamp2Sec(imuPath.poses.front().header.stamp) < lidarOdomTime - 1.0)
                imuPath.poses.erase(imuPath.poses.begin());
            if (pubImuPath->get_subscription_count() != 0)
            {
                imuPath.header.stamp = imuOdomQueue.back().header.stamp;
                imuPath.header.frame_id = "odom";
                pubImuPath->publish(imuPath);
            }
        }
    }
};

class IMUPreintegration : public ParamServer
{
public:

    std::mutex mtx;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subImu;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdometry;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subVelocity;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubImuOdometry;

    rclcpp::CallbackGroup::SharedPtr callbackGroupImu;
    rclcpp::CallbackGroup::SharedPtr callbackGroupOdom;

    bool systemInitialized = false;

    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
    gtsam::Vector noiseModelBetweenBias;


    gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;
    gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;

    std::deque<sensor_msgs::msg::Imu> imuQueOpt;
    std::deque<sensor_msgs::msg::Imu> imuQueImu;
    sensor_msgs::msg::Imu latestImu;
    bool ImuInited = false;

    gtsam::Pose3 prevPose_;
    gtsam::Vector3 prevVel_;
    gtsam::NavState prevState_;
    gtsam::imuBias::ConstantBias prevBias_;
    Eigen::Vector3d odomSinceLastOpt_ = Eigen::Vector3d(0, 0, 0);
    Eigen::Matrix3d RotSinceLastOpt_ = Eigen::Matrix3d::Identity();

    gtsam::Vector3 obsVel_;
    bool obsVelInited = false;

    gtsam::NavState prevStateOdom;
    gtsam::imuBias::ConstantBias prevBiasOdom;

    bool doneFirstOpt = false;
    double lastImuT_imu = -1;
    double lastImuT_opt = -1;
    double lastVelT = -1;
    rclcpp::Time lastImuTimestamp_imu = this->now();
    rclcpp::Time lastImuTimestamp_opt = this->now();
    rclcpp::Time lastVelTimestamp = this->now();

    gtsam::ISAM2 optimizer;
    gtsam::NonlinearFactorGraph graphFactors;
    gtsam::Values graphValues;

    const double delta_t = 0;

    int key = 1;

    gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));
    gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));

    IMUPreintegration(const rclcpp::NodeOptions & options) :
            ParamServer("lio_sam_imu_preintegration", options)
    {
        callbackGroupImu = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        callbackGroupOdom = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        auto imuOpt = rclcpp::SubscriptionOptions();
        imuOpt.callback_group = callbackGroupImu;
        auto odomOpt = rclcpp::SubscriptionOptions();
        odomOpt.callback_group = callbackGroupOdom;

        subImu = create_subscription<sensor_msgs::msg::Imu>(
            imuTopic, rclcpp::SensorDataQoS(),
            std::bind(&IMUPreintegration::imuHandler, this, std::placeholders::_1),
            imuOpt);
        subOdometry = create_subscription<nav_msgs::msg::Odometry>(
            "lio_sam/mapping/odometry_incremental", qos,
            std::bind(&IMUPreintegration::odometryHandler, this, std::placeholders::_1),
            odomOpt);

        subVelocity = create_subscription<nav_msgs::msg::Odometry>(
            "gimbal_vel", rclcpp::SystemDefaultsQoS(),
            std::bind(&IMUPreintegration::velHandler, this, std::placeholders::_1),
            odomOpt);

        pubImuOdometry = create_publisher<nav_msgs::msg::Odometry>(odomTopic+"_incremental", qos_imu);

        boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
        p->accelerometerCovariance  = gtsam::Matrix33::Identity(3,3) * pow(imuAccNoise, 2); // acc white noise in continuous
        p->gyroscopeCovariance      = gtsam::Matrix33::Identity(3,3) * pow(imuGyrNoise, 2); // gyro white noise in continuous
        p->integrationCovariance    = gtsam::Matrix33::Identity(3,3) * pow(1e-4, 2); // error committed in integrating position from velocities
        gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());; // assume zero initial bias

        priorPoseNoise  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
        priorVelNoise   = gtsam::noiseModel::Isotropic::Sigma(3, 1e4); // m/s
        priorBiasNoise  = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3); // 1e-2 ~ 1e-3 seems to be good
        correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished()); // rad,rad,rad,m, m, m
        correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished()); // rad,rad,rad,m, m, m
        noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();
        
        imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread
        imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization        
    }

    void resetOptimization()
    {
        gtsam::ISAM2Params optParameters;
        optParameters.relinearizeThreshold = 0.1;
        optParameters.relinearizeSkip = 1;
        optimizer = gtsam::ISAM2(optParameters);

        gtsam::NonlinearFactorGraph newGraphFactors;
        graphFactors = newGraphFactors;

        gtsam::Values NewGraphValues;
        graphValues = NewGraphValues;
    }

    void resetParams()
    {
        lastImuT_imu = -1;
        lastVelT = -1;
        doneFirstOpt = false;
        systemInitialized = false;
    }

    void odometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

        double currentCorrectionTime = stamp2Sec(odomMsg->header.stamp);

        // make sure we have imu data to integrate
        if (imuQueOpt.empty())
            return;

        float p_x = odomMsg->pose.pose.position.x;
        float p_y = odomMsg->pose.pose.position.y;
        float p_z = odomMsg->pose.pose.position.z;
        float r_x = odomMsg->pose.pose.orientation.x;
        float r_y = odomMsg->pose.pose.orientation.y;
        float r_z = odomMsg->pose.pose.orientation.z;
        float r_w = odomMsg->pose.pose.orientation.w;
        bool degenerate = (int)odomMsg->pose.covariance[0] == 1 ? true : false;
        gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));


        // 0. initialize system
        if (systemInitialized == false)
        {
            resetOptimization();

            // pop old IMU message
            while (!imuQueOpt.empty())
            {
                if (stamp2Sec(imuQueOpt.front().header.stamp) < currentCorrectionTime - delta_t)
                {
                    lastImuT_opt = stamp2Sec(imuQueOpt.front().header.stamp);
                    lastImuTimestamp_opt = imuQueOpt.front().header.stamp;
                    imuQueOpt.pop_front();
                }
                else
                    break;
            }
            // initial pose
            prevPose_ = lidarPose.compose(lidar2Imu);
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
            graphFactors.add(priorPose);
            // initial velocity
            prevVel_ = gtsam::Vector3(0, 0, 0);
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
            graphFactors.add(priorVel);
            // initial bias
            prevBias_ = gtsam::imuBias::ConstantBias();
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
            graphFactors.add(priorBias);
            // add values
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();

            imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
            imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
            
            key = 1;
            systemInitialized = true;
            return;
        }


        // reset graph for speed
        if (key == 100)
        {
            // get updated noise before reset
            gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise  = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key-1)));
            // reset graph
            resetOptimization();
            // add pose
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
            graphFactors.add(priorPose);
            // add velocity
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
            graphFactors.add(priorVel);
            // add bias
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
            graphFactors.add(priorBias);
            // add values
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();

            key = 1;
        }


        // 1. integrate imu data and optimize
        while (!imuQueOpt.empty())
        {
            // pop and integrate imu data that is between two optimizations
            sensor_msgs::msg::Imu *thisImu = &imuQueOpt.front();
            double imuTime = stamp2Sec(thisImu->header.stamp);
            if (imuTime < currentCorrectionTime - delta_t)
            {
                if (lastImuT_opt < 0)
                {
                    lastImuT_opt = imuTime;
                    lastImuTimestamp_opt = thisImu->header.stamp;
                }
                rclcpp::Time thisImuTimestamp = thisImu->header.stamp;
                rclcpp::Duration dt_ = thisImuTimestamp - lastImuTimestamp_opt;
                double dt = dt_.seconds();
                dt = (dt <= 0) ? (1.0 / 500) : dt;
                imuIntegratorOpt_->integrateMeasurement(
                        gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                
                lastImuT_opt = imuTime;
                lastImuTimestamp_opt = thisImuTimestamp;
                imuQueOpt.pop_front();
            }
            else
                break;
        }
        // add imu factor to graph
        const gtsam::PreintegratedImuMeasurements& preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements&>(*imuIntegratorOpt_);
        gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
        graphFactors.add(imu_factor);
        // add imu bias between factor
        graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                         gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
        // add pose factor
        gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
        gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, degenerate ? correctionNoise2 : correctionNoise);
        graphFactors.add(pose_factor);
        // add velocity factor
        if (obsVelInited)
        {
            gtsam::noiseModel::Diagonal::shared_ptr velocityNoise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(0.1, 0.1, 0.1));
            gtsam::PriorFactor<gtsam::Vector3> vel_factor(V(key), obsVel_, velocityNoise);
            graphFactors.add(vel_factor);
            obsVelInited = false;
        }
        // insert predicted values
        gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
        graphValues.insert(X(key), propState_.pose());
        graphValues.insert(V(key), propState_.v());
        graphValues.insert(B(key), prevBias_);
        // optimize
        optimizer.update(graphFactors, graphValues);
        optimizer.update();
        graphFactors.resize(0);
        graphValues.clear();
        // Overwrite the beginning of the preintegration for the next step.
        gtsam::Values result = optimizer.calculateEstimate();
        prevPose_  = result.at<gtsam::Pose3>(X(key));
        prevVel_   = result.at<gtsam::Vector3>(V(key));
        prevState_ = gtsam::NavState(prevPose_, prevVel_);
        prevBias_  = result.at<gtsam::imuBias::ConstantBias>(B(key));
        // Reset the optimization preintegration object.
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
        // check optimization
        if (failureDetection(prevVel_, prevBias_))
        {
            resetParams();
            return;
        }


        // 2. after optiization, re-propagate imu odometry preintegration
        prevStateOdom = prevState_;
        prevBiasOdom  = prevBias_;
        // first pop imu message older than current correction data
        double lastImuQT = -1;
        rclcpp::Time lastImuQTimestamp;
        while (!imuQueImu.empty() && stamp2Sec(imuQueImu.front().header.stamp) < currentCorrectionTime - delta_t)
        {
            lastImuQT = stamp2Sec(imuQueImu.front().header.stamp);
            lastImuQTimestamp = imuQueImu.front().header.stamp;
            imuQueImu.pop_front();
        }
        // repropogate
        if (!imuQueImu.empty())
        {
            // reset bias use the newly optimized bias
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
            // integrate imu message from the beginning of this optimization
            for (int i = 0; i < (int)imuQueImu.size(); ++i)
            {
                sensor_msgs::msg::Imu *thisImu = &imuQueImu[i];
                double imuTime = stamp2Sec(thisImu->header.stamp);
                if (lastImuQT < 0)
                {
                    lastImuQT = imuTime;
                    lastImuQTimestamp = thisImu->header.stamp;
                }
                // double dt = (lastImuQT < 0) ? (1.0 / 500.0) :(imuTime - lastImuQT);
                rclcpp::Time thisImuTimestamp = thisImu->header.stamp;
                rclcpp::Duration dt_ = thisImuTimestamp - lastImuQTimestamp;
                double dt = dt_.seconds();
                dt = (dt <= 0) ? (1.0 / 500) : dt;
                imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                                                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                lastImuQT = imuTime;
            }
        }

        ++key;
        odomSinceLastOpt_ = Eigen::Vector3d(0, 0, 0);
        RotSinceLastOpt_ = Eigen::Matrix3d::Identity();
        doneFirstOpt = true;
    }

    void velHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

        double currentCorrectionTime = stamp2Sec(odomMsg->header.stamp);

        // make sure we have imu data to convert velocity in local frame into odom frame
        if (!ImuInited)
            return;
        // make sure we have odometry data
        if (doneFirstOpt == false)
            return;

        // local velocity
        Eigen::Vector3d vel(odomMsg->twist.twist.linear.x, odomMsg->twist.twist.linear.y, odomMsg->twist.twist.linear.z);
        // rotation matrix from local frame to odom frame extracted from imu orientation
        geometry_msgs::msg::Quaternion orientation = latestImu.orientation;
        Eigen::Quaterniond q(orientation.w, orientation.x, orientation.y, orientation.z);
        Eigen::Matrix3d R = q.toRotationMatrix();
        // global velocity
        Eigen::Vector3d velGlobal = R * vel;
        obsVel_ = gtsam::Vector3(velGlobal.x(), velGlobal.y(), velGlobal.z());
        // calculate time interval
        double thisVelT = stamp2Sec(odomMsg->header.stamp);
        // double dt = (lastVelT < 0) ? (1.0 / 500.0) : (thisVelT - lastVelT);
        if (lastVelT < 0)
        {
            lastVelT = thisVelT;
            lastVelTimestamp = odomMsg->header.stamp;
        }
        rclcpp::Time thisVelTimestamp = odomMsg->header.stamp;
        rclcpp::Duration dt_ = thisVelTimestamp - lastVelTimestamp;
        double dt = dt_.seconds();
        dt = (dt <= 0) ? (1.0 / 500) : dt;
        lastVelT = thisVelT;
        lastVelTimestamp = thisVelTimestamp;
        // predict odometry
        // position
        Eigen::Vector3d velCur(odomMsg->twist.twist.linear.x, odomMsg->twist.twist.linear.y, odomMsg->twist.twist.linear.z);
        odomSinceLastOpt_ += velCur * dt;
        // // orientation

        // imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(latestImu.linear_acceleration.x, latestImu.linear_acceleration.y, latestImu.linear_acceleration.z),
        //                                         gtsam::Vector3(latestImu.angular_velocity.x,    latestImu.angular_velocity.y,    latestImu.angular_velocity.z), dt);
        // gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);
        // gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
        // gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);
        // // extract pose in last optimization
        // gtsam::Pose3 oldPose(prevStateOdom.quaternion(), prevStateOdom.position());
        // Eigen::Vector3d oldPoseEigen(oldPose.translation().x(), oldPose.translation().y(), oldPose.translation().z());
        // // publish odometry
        // auto odometry = nav_msgs::msg::Odometry();
        // odometry.header.stamp = odomMsg->header.stamp;
        // odometry.header.frame_id = "odom";
        // odometry.child_frame_id = "odom_imu";

        // // odometry.pose.pose.position.x = lidarPose.translation().x();
        // // odometry.pose.pose.position.y = lidarPose.translation().y();
        // // odometry.pose.pose.position.z = lidarPose.translation().z();
        // Eigen::Vector3d pose = oldPoseEigen + odomSinceLastOpt_;
        // odometry.pose.pose.position.x = pose.x();
        // odometry.pose.pose.position.y = pose.y();
        // odometry.pose.pose.position.z = pose.z();
        // odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
        // odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
        // odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
        // odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();
        // // odometry.pose.pose.orientation.x = orientation.x;
        // // odometry.pose.pose.orientation.y = orientation.y;
        // // odometry.pose.pose.orientation.z = orientation.z;
        // // odometry.pose.pose.orientation.w = orientation.w;
        
        // // odometry.twist.twist.linear.x = currentState.velocity().x();
        // // odometry.twist.twist.linear.y = currentState.velocity().y();
        // // odometry.twist.twist.linear.z = currentState.velocity().z();
        // // odometry.twist.twist.angular.x = odomMsg->angular_velocity.x + prevBiasOdom.gyroscope().x();
        // // odometry.twist.twist.angular.y = odomMsg->angular_velocity.y + prevBiasOdom.gyroscope().y();
        // // odometry.twist.twist.angular.z = odomMsg->angular_velocity.z + prevBiasOdom.gyroscope().z();
        // odometry.twist.twist.linear.x = velGlobal.x();
        // odometry.twist.twist.linear.y = velGlobal.y();
        // odometry.twist.twist.linear.z = velGlobal.z();
        // odometry.twist.twist.angular.x = odomMsg->twist.twist.angular.x + prevBiasOdom.gyroscope().x();
        // odometry.twist.twist.angular.y = odomMsg->twist.twist.angular.y + prevBiasOdom.gyroscope().y();
        // odometry.twist.twist.angular.z = odomMsg->twist.twist.angular.z + prevBiasOdom.gyroscope().z();
        // pubImuOdometry->publish(odometry);

        // obsVelInited = true;
    }

    bool failureDetection(const gtsam::Vector3& velCur, const gtsam::imuBias::ConstantBias& biasCur)
    {
        Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
        if (vel.norm() > 30)
        {
            RCLCPP_WARN(get_logger(), "Large velocity, reset IMU-preintegration!");
            return true;
        }

        Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
        Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
        if (ba.norm() > 1.0 || bg.norm() > 1.0)
        {
            RCLCPP_WARN(get_logger(), "Large bias, reset IMU-preintegration!");
            return true;
        }

        return false;
    }

    void imuHandler(const sensor_msgs::msg::Imu::SharedPtr imu_raw)
    {
        std::lock_guard<std::mutex> lock(mtx);

        // sensor_msgs::msg::Imu thisImu = imuConverter(*imu_raw);
        sensor_msgs::msg::Imu thisImu;
        if (!imuConverter(*imu_raw, thisImu))
            return;

        latestImu = thisImu;
        ImuInited = true;

        imuQueOpt.push_back(thisImu);
        imuQueImu.push_back(thisImu);

        if (doneFirstOpt == false)
            return;

        double imuTime = stamp2Sec(thisImu.header.stamp);
        double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu);
        dt = (dt <= 0) ? (1.0 / 500) : dt;
        lastImuT_imu = imuTime;

        // predict pose based on encoder data
        gtsam::Pose3 oldPose(prevStateOdom.quaternion(), prevStateOdom.position());
        Eigen::Vector3d oldPoseEigen(oldPose.translation().x(), oldPose.translation().y(), oldPose.translation().z());
        Eigen::Vector3d pose = oldPoseEigen + odomSinceLastOpt_;

        // predict orientation
        geometry_msgs::msg::Quaternion orientation = thisImu.orientation;
        Eigen::Quaterniond q(orientation.w, orientation.x, orientation.y, orientation.z);
        Eigen::Matrix3d R = q.toRotationMatrix();
        Eigen::Vector3d omega = Eigen::Vector3d(thisImu.angular_velocity.x, thisImu.angular_velocity.y, thisImu.angular_velocity.z);
        Eigen::Vector3d rotation_vector = omega * dt;
        Eigen::AngleAxisd rotation_vector_(rotation_vector.norm(), rotation_vector.normalized());
        Eigen::Matrix3d R_ = rotation_vector_.toRotationMatrix();
        RotSinceLastOpt_ = RotSinceLastOpt_ * R_;
        Eigen::Matrix3d R__ = R * RotSinceLastOpt_;
        Eigen::Quaterniond q_(R__);


        // publish odometry
        auto odometry = nav_msgs::msg::Odometry();
        odometry.header.stamp = thisImu.header.stamp;
        odometry.header.frame_id = "odom";
        odometry.child_frame_id = "odom_imu";

        odometry.pose.pose.position.x = pose.x();
        odometry.pose.pose.position.y = pose.y();
        odometry.pose.pose.position.z = pose.z();
        // odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
        // odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
        // odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
        // odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();
        odometry.pose.pose.orientation.x = q_.x();
        odometry.pose.pose.orientation.y = q_.y();
        odometry.pose.pose.orientation.z = q_.z();
        odometry.pose.pose.orientation.w = q_.w();

        odometry.twist.twist.linear.x = prevState_.velocity().x();
        odometry.twist.twist.linear.y = prevState_.velocity().y();
        odometry.twist.twist.linear.z = prevState_.velocity().z();
        odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
        odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
        odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
        pubImuOdometry->publish(odometry);
    }
};


int main(int argc, char** argv)
{   
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::MultiThreadedExecutor e;

    auto ImuP = std::make_shared<IMUPreintegration>(options);
    auto TF = std::make_shared<TransformFusion>(options);
    e.add_node(ImuP);
    e.add_node(TF);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> IMU Preintegration Started.\033[0m");

    e.spin();

    rclcpp::shutdown();
    return 0;
}
