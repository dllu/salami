/**
 * header-only library to load KITTI data into Eigen things
 */

#pragma once
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

#include "geometry.h"
#include "velodyne_calibration.h"

namespace salami {
namespace kitti {
struct PointsRings {
    std::shared_ptr<Points> points;
    std::vector<idx> rings;
};

class Loader {
   public:
    Loader(const std::string& folder, const idx dataset)
        : folder_(folder), dataset_(dataset), frame_(0), n_frames_(0) {
        // load calibrations
        std::stringstream path_ss;
        path_ss << folder_ << "/" << std::setw(2) << std::setfill('0')
                << dataset_ << "/calib.txt";
        std::cerr << path_ss.str() << std::endl;
        std::ifstream calib_stream(path_ss.str());
        std::string tmp;
        for (idx camera = 0; camera < 4; camera++) {
            calib_stream >> tmp;
            camera_calibration_[camera] = std::make_unique<Proj>();
            for (idx i = 0; i < 3; i++) {
                for (int j = 0; j < 4; j++) {
                    calib_stream >> (*camera_calibration_[camera])(i, j);
                }
            }
        }

        velodyne_extrinsic_ = std::make_unique<SE3>(SE3::Identity());
        calib_stream >> tmp;
        for (idx i = 0; i < 3; i++) {
            for (idx j = 0; j < 4; j++) {
                calib_stream >> (*velodyne_extrinsic_)(i, j);
                std::cerr << (*velodyne_extrinsic_)(i, j) << "\t";
            }
            std::cerr << std::endl;
        }

        std::stringstream times_ss;
        times_ss << folder_ << "/" << std::setw(2) << std::setfill('0')
                 << dataset_ << "/times.txt";
        std::ifstream times_stream(times_ss.str());
        std::string times_string;
        while (std::getline(times_stream, times_string)) {
            n_frames_++;
        }

        std::fill(range_offset_.begin(), range_offset_.end(), 0);
    }

    PointsRings loadCloud(const idx frame) const {
        std::stringstream path_ss;
        path_ss << folder_ << "/" << std::setw(2) << std::setfill('0')
                << dataset_ << "/velodyne/" << std::setw(6) << frame << ".bin";
        const idx buf_n = 800'000;
        std::vector<float> buf(buf_n);
        FILE* points_stream = fopen(path_ss.str().c_str(), "rb");

        const idx read_n =
            fread(buf.data(), sizeof(float), buf_n, points_stream);
        fclose(points_stream);

        Points raw_points(3, 100'000);
        Point last_p = Point::Zero();

        std::vector<std::unique_ptr<Points>> calibrated_points;

        idx points_ind = 0;
        idx ring_id = 0;
        for (idx j = 0; j < read_n / 4; j++) {
            Point p(buf[j * 4], buf[j * 4 + 1], buf[j * 4 + 2]);
            raw_points.col(points_ind) = p;
            points_ind++;
            if (p(0) > eps && last_p(0) > eps && p(1) >= 0 && last_p(1) < 0) {
                calibrated_points.push_back(velodyne_calibration::calibrateRing(
                    raw_points.leftCols(points_ind), ring_id++, range_offset_));
                points_ind = 0;
            }
            last_p = p;
        }
        calibrated_points.push_back(velodyne_calibration::calibrateRing(
            raw_points.leftCols(points_ind), ring_id++, range_offset_));

        const idx good_n = std::accumulate(
            calibrated_points.begin(), calibrated_points.end(), idx(0),
            [](const idx a, const std::unique_ptr<Points>& b) {
                return (idx)(a + b->cols());
            });

        Points points(3, good_n);
        PointsRings points_ring;
        points_ring.rings.resize(good_n);
        for (idx ring = 0, k = 0; ring < (idx)calibrated_points.size();
             ring++) {
            const idx ring_n = calibrated_points[ring]->cols();
            points.block(0, k, 3, ring_n) = *(calibrated_points[ring]);
            std::fill(points_ring.rings.begin() + k,
                      points_ring.rings.begin() + k + ring_n, ring);
            k += ring_n;
        }

        std::cerr << "Frame " << frame << ", read: " << points.cols()
                  << " points with " << ring_id << " rings" << std::endl;
        points_ring.points =
            std::make_shared<Points>((*velodyne_extrinsic_) * points);
        return points_ring;
    }

    PointsRings loadNextCloud() { return loadCloud(frame_++); }

    idx frameCount() const { return n_frames_; }
    velodyne_calibration::RangeOffset range_offset_;

   private:
    const std::string folder_;
    const idx dataset_;
    idx frame_;
    idx n_frames_;
    std::array<std::unique_ptr<Proj>, 4> camera_calibration_;
    std::unique_ptr<SE3> velodyne_extrinsic_;
};
}  // namespace kitti
}  // namespace salami
