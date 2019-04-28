/**
 * header-only library providing typedefs for Eigen objects
 * and common types and basic functions
 */

#pragma once
#include <cmath>
#include <limits>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>

namespace salami {
using flo = double;          // floating point type for math
using idx = std::ptrdiff_t;  // type for indexing arrays etc
constexpr double eps = std::numeric_limits<flo>::epsilon();
const double num_eps = std::sqrt(eps);  // ugh no constexpr for std::sqrt

using Vec3 = Eigen::Matrix<flo, 3, 1>;
using Point = Vec3;
using Points = Eigen::Matrix<flo, 3, -1>;
using Normals = Eigen::Matrix<flo, 3, -1>;
using Mat3 = Eigen::Matrix<flo, 3, 3>;
using Mat4 = Eigen::Matrix<flo, 4, 4>;
using SE3 = Eigen::Transform<flo, 3, Eigen::Affine>;
using se3 = Eigen::Matrix<flo, 6, 1>;
using Proj = Eigen::Transform<flo, 3, Eigen::Projective>;

namespace geometry {
inline Mat3 covariance(const Points& points) {
    return points * points.transpose() / points.rows();
}

inline std::tuple<Point, Mat3> meanAndCovariance(const Points& points) {
    Point mean = points.rowwise().mean();
    Points centered = points.colwise() - mean;
    Mat3 cov = covariance(centered);
    return std::make_tuple(mean, cov);
}

inline Mat3 skewSym(const Vec3& v) {
    Mat3 m;
    // clang-format off
    m <<      0, -v(2),  v(1),
           v(2),     0, -v(0),
          -v(1),  v(0),     0;
    // clang-format on
    return m;
}

inline Vec3 skewSym(const Mat3& m) { return Vec3(m(2, 1), m(0, 2), m(1, 0)); }
inline Mat4 skewSym(const se3& v) {
    Mat4 m = Mat4::Zero();
    m.block<3, 3>(0, 0) = skewSym(Vec3(v.head<3>()));
    m.block<3, 1>(0, 3) = v.tail<3>();
    return m;
}

// TODO: implement actual closed form method
inline SE3 exp(const se3& v) {
    const Mat4 m = skewSym(v);
    Mat4 result = Mat4::Zero();
    Mat4 a = m;
    flo fact = 1;
    for (idx i = 1; i < 30; i++) {
        fact /= i;
        result.array() += a.array() * fact;
        a = a * m;
    }
    result.array() += Mat4::Identity().array();
    /*
    std::cerr << v.transpose() << std::endl;
    std::cerr << m << std::endl;
    std::cerr << result << std::endl;
    SE3 output = (SE3)result;

    for (idx i = 0; i < 4; i++) {
        for (idx j = 0; j < 4; j++) {
            std::cerr << output(i, j) << "\t";
        }
        std::cerr << std::endl;
    }
    */
    return (SE3)result;
}

void reorthogonalize(SE3& a) {
    Eigen::JacobiSVD<Mat3> svd(a.linear(),
                               Eigen::ComputeThinU | Eigen::ComputeThinV);
    a.linear() = svd.matrixU() * svd.matrixV().transpose();
}

inline se3 log(const SE3& a) {
    std::cerr << "WARNING: log not implemented" << std::endl;
    return se3::Zero();
}
}  // namespace geometry
}  // namespace salami
