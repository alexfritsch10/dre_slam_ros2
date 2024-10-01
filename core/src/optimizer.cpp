// This file is part of dre_slam - Dynamic RGB-D Encoder SLAM for Differential-Drive Robot.
//
// Copyright (C) 2019 Dongsheng Yang <ydsf16@buaa.edu.cn>
// (Biologically Inspired Mobile Robot Laboratory, Robotics Institute, Beihang University)
//
// dre_slam is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// dre_slam is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <dre_slam/optimizer.h>
#include <dre_slam/keyframe.h>
#include <cmath> // Include this for std::floor and M_PI
#include <Eigen/Core>
#include <Eigen/SVD>

namespace dre_slam
{

    template <typename T>
    inline T NormalizeAngle(const T &angle_radians)
    {
        // Use std::floor instead of ceres::floor.
        T two_pi(2.0 * M_PI);
        return angle_radians - two_pi * std::floor((angle_radians + T(M_PI)) / two_pi);
    }

    // Defines a local parameterization for updating the angle to be constrained in
    // [-pi to pi).
    class AngleLocalParameterization : public ceres::LocalParameterization
    {
    public:
        // Compute the update for the parameter theta.
        template <typename T>
        bool operator()(const T *theta_radians, const T *delta_theta_radians,
                        T *theta_radians_plus_delta) const
        {
            *theta_radians_plus_delta = NormalizeAngle(*theta_radians + *delta_theta_radians);
            return true;
        }

        // The function to implement the LocalParameterization interface in Ceres 2.2.0.
        virtual bool Plus(const double *theta_radians,
                          const double *delta_theta_radians,
                          double *theta_radians_plus_delta) const override
        {
            *theta_radians_plus_delta = NormalizeAngle(theta_radians[0] + delta_theta_radians[0]);
            return true;
        }

        // The function to calculate the Jacobian matrix of the Plus operation.
        virtual bool ComputeJacobian(const double *theta_radians,
                                     double *jacobian) const override
        {
            jacobian[0] = 1.0;
            return true;
        }

        // Specify the size of parameter blocks.
        virtual int GlobalSize() const override { return 1; }
        virtual int LocalSize() const override { return 1; }

        // Factory function to create a shared pointer to this parameterization.
        static std::shared_ptr<ceres::LocalParameterization> Create()
        {
            return std::make_shared<AngleLocalParameterization>();
        }

    private:
        // Normalize the angle to be within [-pi, pi).
        template <typename T>
        static T NormalizeAngle(const T &angle_radians)
        {
            const T two_pi = T(2.0 * M_PI);
            return angle_radians - two_pi * std::floor((angle_radians + T(M_PI)) / two_pi);
        }
    };

    // sqrtMatrix
    template <typename Derived>
    Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
    Optimizer::sqrtMatrix(const Eigen::MatrixBase<Derived> &TT)
    {
        using Scalar = typename Derived::Scalar;
        constexpr Scalar kMaxValue = 1e8;

        // Perform SVD decomposition
        Eigen::JacobiSVD<Derived> svd(TT, Eigen::ComputeFullU | Eigen::ComputeFullV);

        Eigen::Matrix<Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> V = svd.matrixV();
        Eigen::Matrix<Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> U = svd.matrixU();

        // Compute the square root of the singular values
        Eigen::Matrix<Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> S = U.inverse() * TT * V.transpose().inverse();
        Eigen::Matrix<Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> sqrt_S = S;

        for (int i = 0; i < S.rows(); ++i)
        {
            sqrt_S(i, i) = std::sqrt(S(i, i));
        }

        // Compute the square root matrix
        Eigen::Matrix<Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> sT = sqrt_S * V.transpose();

        // Check for non-finite values (NaN or Inf)
        if (!sT.allFinite())
        {
            sT = Derived::Identity() * kMaxValue;
        }

        return sT;
    }

    double getVisualSqrtInfo(int octave, double scale_factor)
    {
        double scale = 1.0;
        for (int i = 0; i < octave; i++)
        {
            scale *= scale_factor;
        }
        return 1.0 / scale;
    }

    class ProjectionError
    {
    public:
        ProjectionError(const Eigen::Matrix4d &T_c_r,
                        const double &fx, const double &fy, const double &cx, const double &cy,
                        const cv::KeyPoint &kp, const Eigen::Matrix2d &sqrt_info)
            : T_c_r_(T_c_r), fx_(fx), fy_(fy), cx_(cx), cy_(cy), kp_(kp), sqrt_info_(sqrt_info) {}

        template <typename T>
        bool operator()(const T *const rx, const T *const ry, const T *const rth,
                        const T *const pwx, const T *const pwy, const T *const pwz,
                        T *residuals) const
        {
            // Transformation matrix elements (pre-converted to T type)
            const T c00 = T(T_c_r_(0, 0)), c01 = T(T_c_r_(0, 1)), c02 = T(T_c_r_(0, 2)), c03 = T(T_c_r_(0, 3));
            const T c10 = T(T_c_r_(1, 0)), c11 = T(T_c_r_(1, 1)), c12 = T(T_c_r_(1, 2)), c13 = T(T_c_r_(1, 3));
            const T c20 = T(T_c_r_(2, 0)), c21 = T(T_c_r_(2, 1)), c22 = T(T_c_r_(2, 2)), c23 = T(T_c_r_(2, 3));

            // Camera intrinsics (pre-converted to T type)
            const T fx = T(fx_), fy = T(fy_), cx = T(cx_), cy = T(cy_);

            // 3D point in world frame
            const T px = pwx[0], py = pwy[0], pz = pwz[0];

            // Observed 2D point in image frame
            const T ob_u = T(kp_.pt.x), ob_v = T(kp_.pt.y);

            // Robot pose parameters
            const T x = rx[0], y = ry[0], th = rth[0];

            // Precompute trigonometric values
            const T cth = cos(th);
            const T sth = sin(th);

            // Compute transformation of the 3D point to the camera frame
            T lamda_u = px * (cx * (c20 * cth - c21 * sth) + fx * (c00 * cth - c01 * sth)) +
                        py * (cx * (c21 * cth + c20 * sth) + fx * (c01 * cth + c00 * sth)) -
                        cx * (c20 * (x * cth + y * sth) - c23 + c21 * (y * cth - x * sth)) -
                        fx * (c00 * (x * cth + y * sth) - c03 + c01 * (y * cth - x * sth)) +
                        pz * (c22 * cx + c02 * fx);

            T lamda_v = px * (cy * (c20 * cth - c21 * sth) + fy * (c10 * cth - c11 * sth)) +
                        py * (cy * (c21 * cth + c20 * sth) + fy * (c11 * cth + c10 * sth)) -
                        cy * (c20 * (x * cth + y * sth) - c23 + c21 * (y * cth - x * sth)) -
                        fy * (c10 * (x * cth + y * sth) - c13 + c11 * (y * cth - x * sth)) +
                        pz * (c22 * cy + c12 * fy);

            T lamda = c23 + c22 * pz + px * (c20 * cth - c21 * sth) + py * (c21 * cth + c20 * sth) -
                      c20 * (x * cth + y * sth) - c21 * (y * cth - x * sth);

            // Project the point onto the image plane
            T u = lamda_u / lamda;
            T v = lamda_v / lamda;

            // Calculate the residuals
            T e1 = ob_u - u;
            T e2 = ob_v - v;
            residuals[0] = T(sqrt_info_(0, 0)) * e1 + T(sqrt_info_(0, 1)) * e2;
            residuals[1] = T(sqrt_info_(1, 0)) * e1 + T(sqrt_info_(1, 1)) * e2;

            return true;
        }

        static ceres::CostFunction *Create(const Eigen::Matrix4d &T_c_r,
                                           const double &fx, const double &fy, const double &cx, const double &cy,
                                           const cv::KeyPoint &kp, const Eigen::Matrix2d &sqrt_info)
        {
            return new ceres::AutoDiffCostFunction<ProjectionError, 2, 1, 1, 1, 1, 1, 1>(
                new ProjectionError(T_c_r, fx, fy, cx, cy, kp, sqrt_info));
        }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
        Eigen::Matrix4d T_c_r_;
        double fx_, fy_, cx_, cy_;
        cv::KeyPoint kp_;
        Eigen::Matrix2d sqrt_info_;
    };

    class EncoderFrame2FrameError
    {
    public:
        // Constructor initializing observed motion parameters and square root information matrix
        EncoderFrame2FrameError(const double &delta_x, const double &delta_y, const double &delta_theta, const Eigen::Matrix3d &sqrt_info)
            : delta_x_(delta_x), delta_y_(delta_y), delta_theta_(delta_theta), sqrt_info_(sqrt_info) {}

        template <typename T>
        bool operator()(const T *const ref_x, const T *const ref_y, const T *const ref_th,
                        const T *const cur_x, const T *const cur_y, const T *const cur_th,
                        T *residuals) const
        {
            // Reference pose (initial frame)
            T xr = ref_x[0];
            T yr = ref_y[0];
            T thr = ref_th[0];

            // Current pose (next frame)
            T xc = cur_x[0];
            T yc = cur_y[0];
            T thc = cur_th[0];

            // Observed motion differences
            T ob_dx = T(delta_x_);
            T ob_dy = T(delta_y_);
            T ob_dth = T(delta_theta_);

            // Compute the change in position and orientation
            T tmp_dx = xc - xr;
            T tmp_dy = yc - yr;

            // Rotate the change in position to the reference frame
            T dx = cos(thr) * tmp_dx + sin(thr) * tmp_dy;
            T dy = -sin(thr) * tmp_dx + cos(thr) * tmp_dy;

            // Compute the residual errors
            T ex = dx - ob_dx;
            T ey = dy - ob_dy;
            T eth = NormalizeAngle(thc - thr - ob_dth);

            // Apply the square root information matrix to compute the final residuals
            residuals[0] = T(sqrt_info_(0, 0)) * ex + T(sqrt_info_(0, 1)) * ey + T(sqrt_info_(0, 2)) * eth;
            residuals[1] = T(sqrt_info_(1, 0)) * ex + T(sqrt_info_(1, 1)) * ey + T(sqrt_info_(1, 2)) * eth;
            residuals[2] = T(sqrt_info_(2, 0)) * ex + T(sqrt_info_(2, 1)) * ey + T(sqrt_info_(2, 2)) * eth;

            return true;
        }

        // Static method to create a Ceres AutoDiffCostFunction object for this error
        static ceres::CostFunction *Create(const double &delta_x, const double &delta_y, const double &delta_theta, const Eigen::Matrix3d &sqrt_info)
        {
            // Create an AutoDiffCostFunction with the number of residuals set to 3 and each pose parameter set to 1 dimension
            return new ceres::AutoDiffCostFunction<EncoderFrame2FrameError, 3, 1, 1, 1, 1, 1, 1>(
                new EncoderFrame2FrameError(delta_x, delta_y, delta_theta, sqrt_info));
        }

    private:
        // Member variables for observed motion and square root information matrix
        const double delta_x_, delta_y_, delta_theta_;
        const Eigen::Matrix3d sqrt_info_;
    };

    void Optimizer::motionOnlyBA(Frame *frame, KeyFrame *ref_kf, EncoderIntegration &encoder_kf2f_)
    {
        // Parameters
        double &fx = cfg_->cam_fx_;
        double &fy = cfg_->cam_fy_;
        double &cx = cfg_->cam_cx_;
        double &cy = cfg_->cam_cy_;
        const Eigen::Matrix4d &Tcr = cfg_->Trc_.inverse().matrix();

        // Copy data
        Sophus::SE2 f_Twr2 = frame->getSE2Pose();
        double frame_pose[3] = {f_Twr2.translation()(0), f_Twr2.translation()(1), f_Twr2.so2().log()};
        Sophus::SE2 ref_kf_Twr = ref_kf->getSE2Pose();
        double ref_kf_pose[3] = {ref_kf_Twr.translation()(0), ref_kf_Twr.translation()(1), ref_kf_Twr.so2().log()};

        double mpts[frame->n_matched_ * 3];
        int idx = 0; // map point id.

        for (size_t i = 0; i < frame->mpts_.size(); i++)
        {
            MapPoint *mpt = frame->mpts_.at(i);
            if (mpt == nullptr)
            {
                continue;
            }
            Eigen::Vector3d ps = mpt->getPosition();
            mpts[idx * 3 + 0] = ps(0);
            mpts[idx * 3 + 1] = ps(1);
            mpts[idx * 3 + 2] = ps(2);
            idx++;
        }

        // Loss function
        auto visual_loss = std::make_unique<ceres::HuberLoss>(2.5);
        auto encoder_loss = std::make_unique<ceres::HuberLoss>(0.5);
        auto angle_local_parameterization = AngleLocalParameterization::Create();

        ceres::Problem problem;

        // Add projection errors.
        idx = 0;
        for (size_t i = 0; i < frame->mpts_.size(); i++)
        {
            MapPoint *mpt = frame->mpts_.at(i);
            cv::KeyPoint &kp = frame->kps_.at(i);
            if (mpt == nullptr)
            {
                continue;
            }

            Eigen::Matrix2d v_sqrt_info = getVisualSqrtInfo(kp.octave, cfg_->ret_ft_scale_factor_) * Eigen::Matrix2d::Identity();

            // Cost function
            auto cost_function = ProjectionError::Create(Tcr, fx, fy, cx, cy, kp, v_sqrt_info);

            problem.AddResidualBlock(
                cost_function,
                visual_loss.get(),
                frame_pose, frame_pose + 1, frame_pose + 2,
                mpts + 3 * idx, mpts + 3 * idx + 1, mpts + 3 * idx + 2);

            // Set map points fixed
            problem.SetParameterBlockConstant(mpts + (3 * idx));
            problem.SetParameterBlockConstant(mpts + (3 * idx + 1));
            problem.SetParameterBlockConstant(mpts + (3 * idx + 2));

            idx++;
        }

        // Local parameterization for angle
        problem.SetParameterization(frame_pose + 2, angle_local_parameterization);

        // Add encoder errors
        Eigen::Matrix3d o_sqrt_info = sqrtMatrix<Eigen::Matrix3d>(encoder_kf2f_.getCov().inverse());
        auto cost_function_encoder = EncoderFrame2FrameError::Create(
            encoder_kf2f_.getTrr().translation()(0),
            encoder_kf2f_.getTrr().translation()(1),
            encoder_kf2f_.getTrr().so2().log(),
            o_sqrt_info);
        problem.AddResidualBlock(
            cost_function_encoder,
            encoder_loss.get(),
            ref_kf_pose, ref_kf_pose + 1, ref_kf_pose + 2,
            frame_pose, frame_pose + 1, frame_pose + 2);

        // Local parameterization for angle of reference keyframe
        problem.SetParameterization(ref_kf_pose + 2, angle_local_parameterization);

        // Set reference keyframe pose fixed
        problem.SetParameterBlockConstant(ref_kf_pose);
        problem.SetParameterBlockConstant(ref_kf_pose + 1);
        problem.SetParameterBlockConstant(ref_kf_pose + 2);

        // Optimization settings
        ceres::Solver::Options options;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.minimizer_progress_to_stdout = false;
        options.gradient_tolerance = 1e-10;
        options.function_tolerance = 1e-10;
        options.parameter_tolerance = 1e-10;
        options.num_threads = 1;
        options.max_num_iterations = 20;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // Update frame pose
        frame->setPose(Sophus::SE2(frame_pose[2], Eigen::Vector2d(frame_pose[0], frame_pose[1])));
    } // motionOnlyBA

    void Optimizer::localBA(std::vector<KeyFrame *> &opt_kfs)
    {
        // Parameters
        double &fx = cfg_->cam_fx_;
        double &fy = cfg_->cam_fy_;
        double &cx = cfg_->cam_cx_;
        double &cy = cfg_->cam_cy_;
        const Eigen::Matrix4d &Tcr = cfg_->Trc_.inverse().matrix();

        // Min optimize keyframe ID.
        long int min_opt_kf_id = opt_kfs.back()->kfid_;

        // Loss functions
        auto visual_loss = std::make_unique<ceres::HuberLoss>(2.5);
        auto encoder_loss = std::make_unique<ceres::HuberLoss>(0.5);
        auto angle_local_parameterization = AngleLocalParameterization::Create();

        ceres::Problem problem;

        for (size_t i = 0; i < opt_kfs.size(); i++)
        {
            // Add visual error terms.
            KeyFrame *kf = opt_kfs.at(i);
            kf->cvtEigen2Double();

            // For all map points.
            for (size_t j = 0; j < kf->mpts_.size(); j++)
            {
                // Discard null map points.
                MapPoint *mpt = kf->mpts_.at(j);
                if (mpt == nullptr)
                {
                    continue;
                }
                // Discard single view map points.
                if (mpt->ob_kfs_.size() == 1)
                {
                    continue;
                }

                mpt->cvtEigen2Double();

                // For all observed keyframes.
                const std::map<KeyFrame *, int> &ob_kfs = mpt->ob_kfs_;
                for (const auto &[ob_kf, idx] : ob_kfs)
                {
                    ob_kf->cvtEigen2Double();

                    // Add visual error.
                    cv::KeyPoint &kp = ob_kf->kps_.at(idx);
                    Eigen::Matrix2d v_sqrt_info = getVisualSqrtInfo(kp.octave, cfg_->ret_ft_scale_factor_) * Eigen::Matrix2d::Identity();

                    // Cost function
                    auto cost_function = ProjectionError::Create(Tcr, fx, fy, cx, cy, kp, v_sqrt_info);

                    problem.AddResidualBlock(
                        cost_function,
                        visual_loss.get(),
                        ob_kf->Twrd_, ob_kf->Twrd_ + 1, ob_kf->Twrd_ + 2,
                        mpt->ptwd_, mpt->ptwd_ + 1, mpt->ptwd_ + 2);

                    if (ob_kf->kfid_ < min_opt_kf_id)
                    {
                        problem.SetParameterBlockConstant(ob_kf->Twrd_);
                        problem.SetParameterBlockConstant(ob_kf->Twrd_ + 1);
                        problem.SetParameterBlockConstant(ob_kf->Twrd_ + 2);
                    }
                } // for all viewed keyframes.

                // If the map point is created by a fixed keyframe, fix it.
                if (mpt->first_ob_kf_->kfid_ < min_opt_kf_id)
                {
                    problem.SetParameterBlockConstant(mpt->ptwd_);
                    problem.SetParameterBlockConstant(mpt->ptwd_ + 1);
                    problem.SetParameterBlockConstant(mpt->ptwd_ + 2);
                }

                const double range = 5.0;
                problem.SetParameterLowerBound(mpt->ptwd_, 0, *(mpt->ptwd_) - range);
                problem.SetParameterLowerBound(mpt->ptwd_ + 1, 0, *(mpt->ptwd_ + 1) - range);
                problem.SetParameterLowerBound(mpt->ptwd_ + 2, 0, *(mpt->ptwd_ + 2) - range);

                problem.SetParameterUpperBound(mpt->ptwd_, 0, *(mpt->ptwd_) + range);
                problem.SetParameterUpperBound(mpt->ptwd_ + 1, 0, *(mpt->ptwd_ + 1) + range);
                problem.SetParameterUpperBound(mpt->ptwd_ + 2, 0, *(mpt->ptwd_ + 2) + range);
            } // for all map points.

            // Add encoder error terms.
            KeyFrame *rkf = kf->getLastKeyFrameEdge();
            rkf->cvtEigen2Double();

            Eigen::Matrix3d o_sqrt_info = sqrtMatrix<Eigen::Matrix3d>(kf->covrr_.inverse());
            auto cost_function = EncoderFrame2FrameError::Create(
                kf->Trr_.translation()(0),
                kf->Trr_.translation()(1),
                kf->Trr_.so2().log(),
                o_sqrt_info);

            problem.AddResidualBlock(
                cost_function,
                encoder_loss.get(),
                rkf->Twrd_, rkf->Twrd_ + 1, rkf->Twrd_ + 2,
                kf->Twrd_, kf->Twrd_ + 1, kf->Twrd_ + 2);

            problem.SetParameterization(kf->Twrd_ + 2, angle_local_parameterization);
            problem.SetParameterization(rkf->Twrd_ + 2, angle_local_parameterization);

            if (rkf->kfid_ < min_opt_kf_id)
            {
                problem.SetParameterBlockConstant(rkf->Twrd_);
                problem.SetParameterBlockConstant(rkf->Twrd_ + 1);
                problem.SetParameterBlockConstant(rkf->Twrd_ + 2);
            }
        } // for all optimize keyframes.

        // Solve.
        ceres::Solver::Options options;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = 1;
        options.max_num_iterations = cfg_->sm_lm_lba_niter;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // Reset keyframe pose and single view keypoints.
        for (size_t i = 0; i < opt_kfs.size(); i++)
        {
            KeyFrame *kf = opt_kfs.at(i);

            // Reset the keyframe.
            kf->calculateRelativePosition();
            kf->cvtDouble2Eigen();
            kf->reCalculateSingleMpts();

            // Reset optimized map points.
            for (MapPoint *mpt : kf->mpts_)
            {
                if (mpt != nullptr)
                {
                    if (mpt->ob_kfs_.size() == 1)
                    {
                        continue;
                    }

                    if (mpt->first_ob_kf_->kfid_ < min_opt_kf_id)
                    {
                        continue;
                    }
                    mpt->cvtDouble2Eigen();
                }
            }
        } // for all opt_kfs
    } // localBA
    void Optimizer::motionOnlyBA(std::vector<cv::KeyPoint> &kps, std::vector<MapPoint *> &mpts, Sophus::SE2 &pose)
    {
        // Parameters
        double &fx = cfg_->cam_fx_;
        double &fy = cfg_->cam_fy_;
        double &cx = cfg_->cam_cx_;
        double &cy = cfg_->cam_cy_;
        const Eigen::Matrix4d &Tcr = cfg_->Trc_.inverse().matrix();

        // Loss function
        auto visual_loss = std::make_unique<ceres::HuberLoss>(5);

        // Local parameterization for angle
        auto angle_local_parameterization = AngleLocalParameterization::Create();

        double pd3[3] = {pose.translation()(0), pose.translation()(1), pose.so2().log()};

        ceres::Problem problem;

        for (size_t i = 0; i < mpts.size(); i++)
        {
            MapPoint *mpt = mpts.at(i);
            if (mpt == nullptr)
            {
                continue;
            }
            mpt->cvtEigen2Double();
            cv::KeyPoint &kp = kps.at(i);

            Eigen::Matrix2d v_sqrt_info = getVisualSqrtInfo(kp.octave, cfg_->ret_ft_scale_factor_) * Eigen::Matrix2d::Identity();

            // Cost function
            auto cost_function = ProjectionError::Create(Tcr, fx, fy, cx, cy, kp, v_sqrt_info);

            problem.AddResidualBlock(
                cost_function,
                visual_loss.get(),
                pd3, (pd3 + 1), (pd3 + 2),
                mpt->ptwd_, mpt->ptwd_ + 1, mpt->ptwd_ + 2);

            // Set map points as fixed
            problem.SetParameterBlockConstant(mpt->ptwd_);
            problem.SetParameterBlockConstant(mpt->ptwd_ + 1);
            problem.SetParameterBlockConstant(mpt->ptwd_ + 2);
        } // for all mpts

        problem.SetParameterization(pd3 + 2, angle_local_parameterization);

        // Solver options
        ceres::Solver::Options options;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = 1;
        options.max_num_iterations = 40;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        pose = Sophus::SE2(pd3[2], Eigen::Vector2d(pd3[0], pd3[1]));
    } // motionOnlyBA

    void Optimizer::poseGraphOptimization(KeyFrame *loop_kf)
    {
        std::map<long int, KeyFrame *> kfs = map_->getAllKeyFrames();
        auto it = std::next(kfs.begin()); // Start from the second KeyFrame

        auto loss = std::make_unique<ceres::HuberLoss>(0.5);
        auto angle_local_parameterization = AngleLocalParameterization::Create();

        ceres::Problem problem;

        for (; it != kfs.end(); ++it)
        {
            KeyFrame *kf = it->second;
            if (kf->kfid_ <= loop_kf->kfid_)
            {
                continue;
            }

            // Encoder edge
            KeyFrame *lkf = kf->getLastKeyFrameEdge();

            kf->cvtEigen2Double();
            lkf->cvtEigen2Double();

            auto cost_function = EncoderFrame2FrameError::Create(
                kf->Trr_.translation()(0),
                kf->Trr_.translation()(1),
                kf->Trr_.so2().log(),
                cfg_->sm_lc_encoder_edge_weight_ * Eigen::Matrix3d::Identity());

            problem.AddResidualBlock(
                cost_function,
                loss.get(),
                lkf->Twrd_, lkf->Twrd_ + 1, lkf->Twrd_ + 2,
                kf->Twrd_, kf->Twrd_ + 1, kf->Twrd_ + 2);

            problem.SetParameterization(kf->Twrd_ + 2, angle_local_parameterization);
            problem.SetParameterization(lkf->Twrd_ + 2, angle_local_parameterization);

            // Loop edge
            std::vector<KeyFrame *> loop_kfs;
            std::vector<Sophus::SE2> loop_delta_pose;
            kf->getLoopEdge(loop_kfs, loop_delta_pose);

            for (size_t nkf = 0; nkf < loop_kfs.size(); ++nkf)
            {
                KeyFrame *loop_kf = loop_kfs.at(nkf);
                loop_kf->cvtEigen2Double();

                // Check if there is a conflict with the consensus side
                std::set<KeyFrame *> ob_kfs = kf->getObKFs();
                ob_kfs.erase(loop_kf); // Remove the loop_kf if exists
                kf->setObKFs(ob_kfs);

                Sophus::SE2 delta_pose = loop_delta_pose.at(nkf);
                auto loop_cost_function = EncoderFrame2FrameError::Create(
                    delta_pose.translation()(0),
                    delta_pose.translation()(1),
                    delta_pose.so2().log(),
                    cfg_->sm_lc_loop_edge_weight_ * Eigen::Matrix3d::Identity());

                problem.AddResidualBlock(
                    loop_cost_function,
                    loss.get(),
                    loop_kf->Twrd_, loop_kf->Twrd_ + 1, loop_kf->Twrd_ + 2,
                    kf->Twrd_, kf->Twrd_ + 1, kf->Twrd_ + 2);

                problem.SetParameterization(loop_kf->Twrd_ + 2, angle_local_parameterization);
                problem.SetParameterization(kf->Twrd_ + 2, angle_local_parameterization);
            } // for all loop edges

            // Observation edge
            std::set<KeyFrame *> ob_kfs = kf->getVisualEdge();
            for (KeyFrame *okf : ob_kfs)
            {
                okf->cvtEigen2Double();

                Sophus::SE2 delta_pose = okf->getSE2Pose().inverse() * kf->getSE2Pose();

                auto obs_cost_function = EncoderFrame2FrameError::Create(
                    delta_pose.translation()(0),
                    delta_pose.translation()(1),
                    delta_pose.so2().log(),
                    cfg_->sm_lc_cov_edge_weight_ * Eigen::Matrix3d::Identity());

                problem.AddResidualBlock(
                    obs_cost_function,
                    loss.get(),
                    okf->Twrd_, okf->Twrd_ + 1, okf->Twrd_ + 2,
                    kf->Twrd_, kf->Twrd_ + 1, kf->Twrd_ + 2);

                problem.SetParameterization(okf->Twrd_ + 2, angle_local_parameterization);
                problem.SetParameterization(kf->Twrd_ + 2, angle_local_parameterization);
            } // for all observed KFs
        } // for all KFs

        // Fix the first KF
        loop_kf->cvtEigen2Double();
        problem.SetParameterBlockConstant(loop_kf->Twrd_);
        problem.SetParameterBlockConstant(loop_kf->Twrd_ + 1);
        problem.SetParameterBlockConstant(loop_kf->Twrd_ + 2);

        ceres::Solver::Options options;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = 1;
        options.max_num_iterations = cfg_->sm_lc_pgop_niter_;

        options.gradient_tolerance = 1e-20;
        options.function_tolerance = 1e-20;
        options.parameter_tolerance = 1e-20;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // Mutex
        std::unique_lock<std::mutex> lock(map_->update_mutex_);

        // Move all Mpts and KFs
        for (it = kfs.begin(); it != kfs.end(); ++it)
        {
            KeyFrame *kf = it->second;
            if (kf->kfid_ <= loop_kf->kfid_)
            {
                continue;
            }
            kf->calculateRelativePosition();
            kf->cvtDouble2Eigen();
            kf->reCalculateMpts();
        }
    } // poseGraphOptimization

} // namespace dre_slam
