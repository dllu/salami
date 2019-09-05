/**
 * header-only library to perform radius search
 */
#pragma once
#include <memory>
#include <vector>

#include "geometry.h"
#include "nanoflann.hpp"

namespace salami {
namespace nn {
template <idx N = 256>
class GridSearch {
   public:
    GridSearch(const flo radius, std::shared_ptr<Points> points,
               const std::vector<idx>& ind)
        : radius_sq_(radius * radius),
          radius_inv_(1.0 / radius),
          points_(points),
          grid_(N) {
        const Eigen::Array<idx, 3, -1> points_integer =
            (*points * radius_inv_).array().floor().template cast<idx>() +
            N / 2;
        auto insertPoint = [&](idx i) {
            const idx x = points_integer(1, i);
            const idx y = points_integer(0, i);
            const idx z = points_integer(2, i);
            if (z >= 0 && z < N && y >= 0 && y < N && x >= 0 && x < N) {
                if (!grid_[y][z]) {
                    grid_[y][z] = std::make_unique<x_grid_t>();
                }
                if (!grid_[y][z]->at(x)) {
                    grid_[y][z]->at(x) = std::make_unique<cell_t>();
                }
                grid_[y][z]->at(x)->push_back(i);
            }
        };

        if (ind.empty()) {
            for (idx i = 0, n = points->cols(); i < n; i++) {
                insertPoint(i);
            }
        } else {
            for (idx i : ind) {
                insertPoint(i);
            }
        }
    }

    /**
     * perform radius search. We use template P to accept
     * various Eigen templates (columns, slices, etc)
     */
    template <class P>
    std::vector<idx> search(const P& point) const {
        std::vector<idx> indices;
        for (idx dx = -1; dx <= 1; dx++) {
            for (idx dy = -1; dy <= 1; dy++) {
                for (idx dz = -1; dz <= 1; dz++) {
                    const idx x =
                        std::floor(point(1) * radius_inv_) + N / 2 + dx;
                    const idx y =
                        std::floor(point(0) * radius_inv_) + N / 2 + dy;
                    const idx z =
                        std::floor(point(2) * radius_inv_) + N / 2 + dz;
                    if (z < 0 || z >= N || y < 0 || y >= N || x < 0 || x >= N) {
                        continue;
                    }
                    if (!grid_[y][z] || !grid_[y][z]->at(x)) {
                        continue;
                    }
                    for (idx i : *(grid_[y][z]->at(x))) {
                        if ((points_->col(i) - point).squaredNorm() <=
                            radius_sq_) {
                            indices.push_back(i);
                        }
                    }
                }
            }
        }
        return indices;
    }

   private:
    const flo radius_sq_;
    const flo radius_inv_;
    std::shared_ptr<Points> points_;

    using cell_t = std::vector<idx>;
    using x_grid_t = std::array<std::unique_ptr<cell_t>, N>;
    using zx_grid_t = std::array<std::unique_ptr<x_grid_t>, N>;
    using yzx_grid_t = std::vector<zx_grid_t>;
    yzx_grid_t grid_;
};

class NanoFlannSearch {
    using mat_t = Eigen::Matrix<flo, -1, 3>;

   public:
    NanoFlannSearch(const flo radius, std::shared_ptr<Points> points)
        : radius_sq_(radius * radius),
          points_t_(points->transpose()),
          kd_(3, points_t_) {}
    template <class P>
    std::vector<idx> search(const P& point) const {
        std::vector<std::pair<idx, flo>> indices;
        nanoflann::SearchParams params;
        kd_.index->radiusSearch(point.data(), radius_sq_, indices, params);

        std::vector<idx> only_indices;
        only_indices.reserve(indices.size());
        for (auto ir : indices) {
            /*
            std::cerr << ir.second << " "
                      << (points_t_.row(ir.first).transpose() - point).norm()
                      << std::endl;
                      */
            only_indices.push_back(ir.first);
        }
        return only_indices;
    }

   private:
    const flo radius_sq_;
    const mat_t points_t_;
    nanoflann::KDTreeEigenMatrixAdaptor<const mat_t> kd_;
};
}  // namespace nn
}  // namespace salami
