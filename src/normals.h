#pragma once
#include <Eigen/SVD>
#include <queue>
#include "geometry.h"

namespace salami {
namespace normals {

struct NormalsInformation {
    Normals normals_;
    std::vector<idx> features_;
    std::vector<idx> valid_;
};

template <idx K>
inline std::unique_ptr<NormalsInformation> compute(const Points& points,
                                                   const idx n_samples = 200) {
    const Eigen::Matrix<flo, -1, 3> points_t(points.transpose());
    const nanoflann::KDTreeEigenMatrixAdaptor<const Eigen::Matrix<flo, -1, 3>>
        kd(3, points_t);

    std::vector<std::priority_queue<std::pair<flo, idx>>> feature_responses(9);
    const idx n = points.cols();
    std::unique_ptr<NormalsInformation> normals_information =
        std::make_unique<NormalsInformation>();
    normals_information->normals_ = Normals::Zero(3, n);
    normals_information->valid_.reserve(n);
    for (idx i = 0; i < n; i++) {
        const Point p = points.col(i);
        if (p.squaredNorm() < 4.0) {
            continue;
        }

        std::array<idx, K> indices;
        std::array<flo, K> dist;

        kd.query(p.data(), K, indices.data(), dist.data());

        const idx m = indices.size();
        if (m < 3) {
            continue;
        }
        Points nearest_points(3, m);
        for (idx j = 0; j < m; j++) {
            nearest_points.col(j) = points.col(indices[j]);
        }
        Points nearest_points_centered =
            nearest_points.colwise() - p;  // nearest_points.rowwise().mean();
        Mat3 cov = geometry::covariance(nearest_points_centered);

        Eigen::SelfAdjointEigenSolver<Mat3> eigensolver(cov);
        // std::cerr << eigensolver.eigenvalues().transpose() << std::endl;
        Vec3 normal = eigensolver.eigenvectors().col(0);

        // HACK: fix normals on ground
        /*
        if (normal.dot(p) > 0) {
            normal = -normal;
        }
        */
        if (normal.dot(p + 0.2 * Vec3::UnitY()) > 0) {
            normal = -normal;
        }
        if (std::abs(normal.dot(Vec3::UnitY())) > 0.95 &&
            normal.dot(Vec3::UnitY()) < 0) {
            normal = -normal;
        }
        normals_information->normals_.col(i) = normal;
        if (m < K) continue;
        normals_information->valid_.push_back(i);

        Vec3 sigmas = eigensolver.eigenvalues().cwiseSqrt();

        const flo a2d = (sigmas(1) - sigmas(0)) / sigmas(2);
        const flo a2d_sq = a2d * a2d;

        for (idx dim = 0; dim < 3; dim++) {
            Vec3 direction = Vec3::Zero();
            direction(dim) = 1;
            const flo rotational_strength =
                a2d_sq * (p.cross(normal)).dot(direction);
            feature_responses[dim * 2].push(
                std::make_pair(rotational_strength, i));
            feature_responses[dim * 2 + 1].push(
                std::make_pair(-rotational_strength, i));

            const flo translational_strength =
                a2d_sq * std::abs(normal.dot(direction));
            feature_responses[dim + 6].push(
                std::make_pair(-translational_strength, i));
        }

        for (auto& pq : feature_responses) {
            if ((idx)pq.size() > n_samples) pq.pop();
        }
    }

    normals_information->features_.reserve(9 * n_samples);
    for (auto& pq : feature_responses) {
        while (!pq.empty()) {
            normals_information->features_.push_back(pq.top().second);
            pq.pop();
        }
    }
    return normals_information;
}
}  // namespace normals
}  // namespace salami
