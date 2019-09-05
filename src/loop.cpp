#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "kitti.h"
#include "slam.h"

using namespace salami;

constexpr idx n_frames = 7;

constexpr flo loop_perimeter_thresh = 200.0;
constexpr flo drift = 0.02;
constexpr flo loop_distance_thresh = 5.0;

flo pose_dist(const SE3& a, const SE3& b) {
    return (a.translation() - b.translation()).norm();
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "USAGE: salami_loop /path/to/kitti /path/to/poses/00.txt "
                     "/path/to/range_offset.txt 0"
                  << std::endl;
        return 1;
    }

    const std::string kitti_path = argv[1];
    const std::string pose_path = argv[2];
    const std::string range_offset_path = argv[3];
    const idx kitti_dataset = std::atoi(argv[4]);

    std::ifstream pose_stream(pose_path);
    kitti::Loader kitti_loader(kitti_path, kitti_dataset);
    kitti_loader.range_offset_ =
        velodyne_calibration::loadRangeOffset(range_offset_path);

    const idx frame_count = kitti_loader.frameCount();
    std::vector<SE3, Eigen::aligned_allocator<SE3>> poses(frame_count);
    std::vector<flo> distances(frame_count);
    distances[0] = 0;
    for (idx i = 0; i < frame_count; i++) {
        for (idx r = 0; r < 3; r++) {
            for (idx c = 0; c < 4; c++) {
                pose_stream >> poses[i](r, c);
            }
        }
        if (i > 0) {
            distances[i] = distances[i - 1] + pose_dist(poses[i], poses[i - 1]);
        }
    }
    for (idx i = 0; i < frame_count; i++) {
        for (idx j = i - 100; j >= 0; j--) {
            if (distances[i] - distances[j] < loop_perimeter_thresh) {
                continue;
            }
            flo loop_gap = pose_dist(poses[i], poses[j]);
            if (loop_gap <
                loop_distance_thresh + (distances[i] - distances[j]) * drift) {
                // loop detected!
                for (idx t = 0; t < 30; t++) {
                    if (i + 1 < frame_count &&
                        pose_dist(poses[i + 1], poses[j]) < loop_gap) {
                        i++;
                        loop_gap = pose_dist(poses[i], poses[j]);
                    }
                    if (i - 1 >= 0 &&
                        pose_dist(poses[i - 1], poses[j]) < loop_gap) {
                        i--;
                        loop_gap = pose_dist(poses[i], poses[j]);
                    }
                    if (j + 1 < i - 50 &&
                        pose_dist(poses[i], poses[j + 1]) < loop_gap) {
                        j++;
                        loop_gap = pose_dist(poses[i], poses[j]);
                    }
                    if (j - 1 >= 0 &&
                        pose_dist(poses[i], poses[j - 1]) < loop_gap) {
                        j--;
                        loop_gap = pose_dist(poses[i], poses[j]);
                    }
                }
                std::cerr << i << ", " << j << std::endl;
                i += 50;
                break;
            }
        }

        /*
        auto tic = std::chrono::steady_clock::now();
        auto toc = std::chrono::steady_clock::now();
        std::cerr << "Time elapsed: " << (toc - tic).count() << std::endl;
        */
    }
}
