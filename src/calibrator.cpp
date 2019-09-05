#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "kitti.h"
#include "slam.h"

using namespace salami;

constexpr idx n_frames = 7;

class Calibrator {
   public:
    Calibrator() {
        std::fill(range_offset.begin(), range_offset.end(), 0);
        std::fill(population.begin(), population.end(), 0);
    }
    void estimateRangeBias(const kitti::PointsRings& points_rings,
                           const std::deque<slam::Frame>& history) {
        const SE3& pose = *(history[n_frames / 2].pose_);
        const Points curr_points = pose * (*points_rings.points);

        constexpr flo sigma = 0.09;
        constexpr flo sigma_sq_inv = 1.0 / (sigma * sigma);

        for (idx i = 0, n = points_rings.points->cols(); i < n; i++) {
            const idx ring = points_rings.rings[i];
            const Point curr_point = curr_points.col(i);

            flo weight = 0;
            Point target = Point::Zero();
            for (idx frame = 0; frame < n_frames; frame++) {
                if (frame == n_frames / 2) continue;
                const slam::Frame& old_frame = history[frame];
                const Point search_point =
                    (old_frame.pose_->inverse()) * curr_point;

                const auto indices = old_frame.nn_fine_.search(search_point);
                for (const idx j : indices) {
                    const Point other_point =
                        (*old_frame.pose_) * old_frame.points_->col(j);
                    const flo w =
                        std::exp(-(other_point - curr_point).squaredNorm() *
                                 sigma_sq_inv);
                    weight += w;
                    target = target + w * other_point;
                }
            }
            if (weight < num_eps) continue;
            target = target / weight;
            population[ring] += weight;

            const Point raw_point = points_rings.points->col(i);
            const Vec3 point_direction = raw_point / raw_point.norm();
            range_offset[ring] +=
                weight *
                (pose.inverse() * target - raw_point).dot(point_direction);
        }
    }

    velodyne_calibration::RangeOffset getCalibration() {
        velodyne_calibration::RangeOffset ro;
        const flo smoothing = 1e-3;
        for (idx i = 0; i < 64; i++) {
            ro[i] = range_offset[i] / (population[i] + smoothing);
            std::cerr << range_offset[i] << ", " << population[i] << std::endl;
        }
        return ro;
    }

   private:
    velodyne_calibration::RangeOffset range_offset;
    std::array<flo, 64> population;
};

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "USAGE: salami_calibrator /path/to/kitti /path/to/poses "
                     "/path/to/range_offset_input.txt "
                     "/path/to/range_offset_output.txt 0"
                  << std::endl;
        return 1;
    }

    const std::string kitti_path = argv[1];
    const std::string pose_path = argv[2];
    const std::string range_offset_input_path = argv[3];
    const std::string range_offset_output_path = argv[4];
    const idx kitti_dataset = std::atoi(argv[5]);

    std::ifstream pose_stream(pose_path);

    kitti::Loader kitti_loader(kitti_path, kitti_dataset);
    kitti_loader.range_offset_ =
        velodyne_calibration::loadRangeOffset(range_offset_input_path);

    std::deque<kitti::PointsRings> points_rings;
    std::deque<slam::Frame> frames;
    const idx frame_count = kitti_loader.frameCount();
    Calibrator calibrator;
    for (idx i = 0; i < std::min(frame_count, idx(100)); i++) {
        points_rings.push_back(kitti_loader.loadNextCloud());
        frames.emplace_back(points_rings.back().points);
        for (idx r = 0; r < 3; r++) {
            for (idx c = 0; c < 4; c++) {
                pose_stream >> (*frames.back().pose_)(r, c);
            }
        }

        if ((idx)(points_rings.size()) > n_frames) {
            points_rings.pop_front();
            frames.pop_front();
        } else {
            continue;
        }
        auto tic = std::chrono::steady_clock::now();
        calibrator.estimateRangeBias(points_rings[n_frames / 2], frames);
        auto toc = std::chrono::steady_clock::now();
        std::cerr << "Time elapsed: " << (toc - tic).count() << std::endl;
    }
    std::ofstream rout(range_offset_output_path);
    auto ro = calibrator.getCalibration();
    for (idx i = 0; i < 64; i++) {
        ro[i] += kitti_loader.range_offset_[i];
        rout << std::setprecision(10) << std::fixed << ro[i] << std::endl;
    }

    auto pr_before = kitti_loader.loadCloud(0);
    kitti_loader.range_offset_ = ro;
    auto pr_after = kitti_loader.loadCloud(0);
    std::fill(kitti_loader.range_offset_.begin(),
              kitti_loader.range_offset_.end(), 0);
    auto pr_orig = kitti_loader.loadCloud(0);
    std::ofstream pout_before("points_before.txt");
    pout_before << pr_before.points->transpose() << std::endl;
    std::ofstream pout_after("points_after.txt");
    pout_after << pr_after.points->transpose() << std::endl;
    std::ofstream pout_orig("points_orig.txt");
    pout_orig << pr_orig.points->transpose() << std::endl;
}
