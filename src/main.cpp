#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "kitti.h"
#include "slam.h"

using namespace salami;

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "USAGE: salami /path/to/kitti /path/to/range_offset.txt 0"
                  << std::endl;
        return 1;
    }
    const std::string kitti_path = argv[1];
    const idx kitti_dataset = std::atoi(argv[3]);
    kitti::Loader kitti_loader(kitti_path, kitti_dataset);
    kitti_loader.range_offset_ = velodyne_calibration::loadRangeOffset(argv[2]);

    std::deque<slam::Frame> frames;
    SE3 pose = SE3::Identity();
    std::stringstream pout_ss;
    pout_ss << "poses/" << std::setw(2) << std::setfill('0') << kitti_dataset
            << ".txt";
    std::ofstream pout(pout_ss.str());
    SE3 delta = SE3::Identity();

    const idx n_frames = kitti_loader.frameCount();
    for (idx i = 0; i < n_frames; i++) {
        auto tic = std::chrono::steady_clock::now();
        slam::Frame frame(kitti_loader.loadNextCloud().points);
        (*frame.pose_) = delta;
        if (i == 1) {
            frame.pose_->translation()(2) = 1.0;
        }
        auto tac = std::chrono::steady_clock::now();
        /*
        if (i == 0) {
            Eigen::Matrix<flo, 6, -1> zxcv(6, frame.points_->cols());
            zxcv.topRows(3) = *frame.points_;
            zxcv.bottomRows(3) = frame.normals_->normals_.cwiseAbs();
            std::ofstream fout("points.txt");
            fout << zxcv.transpose() << std::endl;
            quick_exit(0);
        }
        */

        if (frames.size() > 0) {
            fineRegister(frame, frames, i == 1);
        }
        delta = *frame.pose_;
        pose = pose * delta;
        if (i == 0 || delta.translation().squaredNorm() > 0.02) {
            frames.push_back(std::move(frame));
        }
        while (frames.size() > 20) {
            frames.pop_front();
        }

        for (const auto& f : frames) {
            (*f.pose_) = (*f.pose_) * delta.inverse();
        }
        if (i % 100 == 0) {
            for (const auto& f : frames) {
                geometry::reorthogonalize(*f.pose_);
            }
            geometry::reorthogonalize(pose);
            geometry::reorthogonalize(delta);
        }
        auto toc = std::chrono::steady_clock::now();
        std::cerr << "Time elapsed: " << (toc - tic).count() << ", "
                  << (toc - tac).count() << std::endl;
        pout << std::fixed << std::setprecision(10) << pose(0, 0) << " "
             << pose(0, 1) << " " << pose(0, 2) << " " << pose(0, 3) << " "
             << pose(1, 0) << " " << pose(1, 1) << " " << pose(1, 2) << " "
             << pose(1, 3) << " " << pose(2, 0) << " " << pose(2, 1) << " "
             << pose(2, 2) << " " << pose(2, 3) << std::endl;
    }
}
