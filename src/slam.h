#pragma once
#include <deque>
#include <memory>
#include "geometry.h"
#include "knn_search.h"
#include "normals.h"
namespace salami {
namespace slam {

// using search_t = nn::NanoFlannSearch;
using search_t = nn::GridSearch<2048>;

const flo fine_radius = 0.18;
const flo coarse_radius = 1.5;

struct Frame {
   public:
    Frame(std::shared_ptr<Points> points)
        : points_(points),
          normals_(normals::compute<60>(*points)),
          nn_fine_(fine_radius, points, normals_->valid_),
          nn_coarse_(coarse_radius, points, normals_->valid_),
          pose_(std::make_unique<SE3>(SE3::Identity())) {}
    std::shared_ptr<Points> points_;
    std::unique_ptr<normals::NormalsInformation> normals_;
    search_t nn_fine_;
    search_t nn_coarse_;
    std::unique_ptr<SE3> pose_;
};

inline void salamiRegister(Frame& frame, const std::deque<Frame>& history,
                           bool coarse = false) {
    const idx n = frame.normals_->features_.size();
    // because the kitti dataset can start while the car is moving,
    // we want the first iteration of the very first registration to be
    // coarse. However, afterwards, coarse can cause it to snap to wrong
    // local minima and is very slow.
    flo lambda = 1e-4;
    for (idx iteration = 0; iteration < 20; iteration++) {
        // const flo sigma = coarse ? 0.4 : 0.06;
        const flo sigma = coarse ? 0.4 : 0.06;
        const flo sigma_sq_inv = 1.0 / (sigma * sigma);
        // std::cerr << "Iteration: " << iteration << std::endl;
        Eigen::Matrix<flo, -1, 6> jacobian(n, 6);
        Eigen::Matrix<flo, -1, 1> residual(n, 1);

        jacobian.setZero();
        residual.setZero();

        // std::ofstream fout("points.txt");
        const Points curr_points = (*frame.pose_) * (*frame.points_);
        const Normals curr_normals =
            frame.pose_->linear() * frame.normals_->normals_;
        flo sum_weights = 0;
        idx inliers = 0;
        for (idx feature_index = 0; feature_index < n; feature_index++) {
            const idx i = frame.normals_->features_[feature_index];
            const Point curr_point = curr_points.col(i);
            // fout <<curr_point.transpose() << std::endl;
            const Vec3 curr_normal = curr_normals.col(i);
            if (curr_normal.squaredNorm() < num_eps) {
                continue;
            }

            flo weight = 0;
            // Point target = Point::Zero();
            Vec3 direction = Vec3::Zero();
            flo ipk = 0;
            for (const auto& old_frame : history) {
                const Point search_point =
                    (old_frame.pose_->inverse()) * curr_point;
                const auto indices =
                    coarse ? (old_frame.nn_coarse_.search(search_point))
                           : (old_frame.nn_fine_.search(search_point));

                for (const idx j : indices) {
                    const Point other_point =
                        (*old_frame.pose_) * old_frame.points_->col(j);
                    const Vec3 old_normal = old_frame.pose_->linear() *
                                            old_frame.normals_->normals_.col(j);
                    const flo direction_similarity =
                        old_normal.dot(curr_normal);
                    if (direction_similarity < num_eps) {
                        continue;
                    }
                    const flo w =
                        // direction_similarity *
                        std::exp(-(other_point - curr_point).squaredNorm() *
                                 sigma_sq_inv);
                    ipk += w * (curr_point - other_point).dot(old_normal);
                    weight += w;
                    // target = target + w * other_point;
                    direction = direction + w * old_normal;
                }
            }
            if (weight < num_eps) continue;
            sum_weights += weight;
            inliers++;

            // target = target / weight;
            direction.normalize();
            // const Point target = curr_point - ipk / weight * direction;

            /*
            fout << curr_point.transpose() << " " << target.transpose() << " "
                 << direction.transpose() << std::endl;
                 */
            Eigen::Matrix<flo, 3, 6> jacobian_block;
            jacobian_block.leftCols<3>() = -geometry::skewSym(curr_point);
            jacobian_block.rightCols<3>() = Mat3::Identity();
            jacobian.row(feature_index) =
                direction.transpose() * jacobian_block;

            residual(feature_index) = -ipk / weight;
        }
        // quick_exit(0);
        /*
        std::ofstream fout("problem.txt");
        Eigen::Matrix<flo, -1, 7> A(n, 7);
        A << jacobian, residual;
        fout << A << std::endl;
        quick_exit(0);
        */
        se3 update = (jacobian.transpose() * jacobian +
                      Eigen::Matrix<flo, 6, 6>::Identity() * lambda)
                         .inverse() *
                     (jacobian.transpose() * residual);
        if ((!coarse && update.squaredNorm() < eps) || iteration == 19) {
            std::cerr << "Score: " << sum_weights << ", " << residual.norm()
                      << ", " << inliers << std::endl;
            break;
        }
        // std::cerr << "Update: " << update.transpose() << std::endl;
        *frame.pose_ = geometry::exp(update) * (*frame.pose_);

        /*
        SE3 update_SE3 = geometry::exp(update);
        for (idx i = 0; i < 4; i++) {
            for (idx j = 0; j < 4; j++) {
                std::cerr << update_SE3(i, j) << "\t";
            }
            std::cerr << std::endl;
        }
        std::cerr << std::endl;
        for (idx i = 0; i < 4; i++) {
            for (idx j = 0; j < 4; j++) {
                std::cerr << (*frame.pose_)(i, j) << "\t";
            }
            std::cerr << std::endl;
        }
        */
        if (iteration > 3) {
            coarse = false;
        }
    }
}
}  // namespace slam
}  // namespace salami
