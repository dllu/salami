#pragma once
#include "geometry.h"
namespace salami {
namespace velodyne_calibration {
constexpr flo min_range = 2.0;
constexpr flo altitude_correction = 0.22 * M_PI / 180.0;
using RangeOffset = std::array<flo, 64>;
RangeOffset loadRangeOffset(const std::string& filename) {
    std::ifstream range_offset_stream(filename);
    RangeOffset range_offset;
    for (idx i = 0; i < 64; i++) {
        range_offset_stream >> range_offset[i];
    }
    return range_offset;
}

// vertical (z-axis) intercept of ray vs velodyne coordinate frame, meters
constexpr std::array<flo, 64> v_offset{
    0.209141535130936, 0.208616367926389, 0.208276279650735, 0.207711958491218,
    0.207304518307663, 0.206773692939529, 0.206509890988135, 0.206049274521721,
    0.205544668001686, 0.205111174243929, 0.204769473906267, 0.204363854102242,
    0.203953242190651, 0.203572177805201, 0.203203076145211, 0.202822246430043,
    0.202505911613084, 0.202101993306034, 0.201821900171436, 0.201463664584844,
    0.201103263187787, 0.200813761483485, 0.200468377657280, 0.200152177093077,
    0.199836200113410, 0.199511138451715, 0.199256573723905, 0.198977234907476,
    0.198655220495721, 0.198328156491508, 0.198134514169738, 0.197831988167745,
    0.125823015705132, 0.125459223320442, 0.125068336964507, 0.124739532958647,
    0.124369808647134, 0.123951752200048, 0.123556650666698, 0.123237701285468,
    0.122927513839287, 0.122654488352438, 0.122284256166695, 0.121894780706677,
    0.121531585051245, 0.121218176189035, 0.120919183576432, 0.120664190919364,
    0.120310319555177, 0.119986888053215, 0.119657642476929, 0.119318228096141,
    0.119080118251604, 0.118854011507986, 0.118608086249998, 0.118279270497984,
    0.117955372733009, 0.117625167827451, 0.117401948153944, 0.117157390515708,
    0.116882532786530, 0.116547107500328, 0.116246988271735, 0.116001603658080};

// slope of ray coming from Velodyne lidar
constexpr std::array<flo, 64> v_slope{
    0.03381401460753143,  0.02748061339687925,  0.02277538431755451,
    0.01521253417895640,  0.01010350224169016,  0.00315655045419278,
    -0.00155092151295872, -0.00789051608194701, -0.01401319335287942,
    -0.02096834980366239, -0.02607844461538463, -0.03200425395257934,
    -0.03854981338249302, -0.04447595071271272, -0.05019832222371898,
    -0.05653429011086694, -0.06185131955733413, -0.06880095597137878,
    -0.07370837546484707, -0.08025212103023741, -0.08596968336944763,
    -0.09190250556892220, -0.09823816271940519, -0.10436998466429130,
    -0.11091462664554903, -0.11704380878937533, -0.12277164453831949,
    -0.12787709166478192, -0.13482537678478249, -0.14157038415989201,
    -0.14606846623624023, -0.15322364873210573, -0.15884889062956989,
    -0.16867085725152728, -0.17745553043346798, -0.18480903170692994,
    -0.19359663798416563, -0.20525842064868591, -0.21465349215988883,
    -0.22283002878331926, -0.23162163242438685, -0.23980200838782803,
    -0.25023021610836510, -0.26044246990591102, -0.27148596259898128,
    -0.28026938845171340, -0.29008587555751725, -0.29723647813257387,
    -0.30930278135165512, -0.31972725632433974, -0.33117376298250578,
    -0.34036036412991477, -0.35059747401069397, -0.35877359303160816,
    -0.36838593846097289, -0.37962335546000936, -0.39023793564316378,
    -0.40272038919431563, -0.41294273050011931, -0.42152258122880532,
    -0.43112240671723884, -0.44461125685792913, -0.45584671208797006,
    -0.46627535261180758};

// range offset, meters
/*
constexpr std::array<flo, 64> range_offset{
    -4.0673e-02, -2.5500e-02, -3.6442e-02, -8.5331e-03, -4.4933e-02,
    2.8335e-02,  -6.1920e-02, -2.9808e-02, -1.6979e-02, -3.3833e-02,
    -5.0795e-02, -3.0273e-02, -5.4886e-02, -4.0219e-02, -1.7165e-02,
    -1.1946e-02, -4.2163e-02, -2.5614e-02, -6.6968e-02, -3.1977e-02,
    -3.5535e-02, -1.6876e-02, -1.8821e-02, -1.8397e-02, -4.5171e-02,
    -2.5561e-02, -4.1792e-02, -7.9374e-03, -4.9073e-02, -2.4687e-02,
    -3.3296e-02, -2.8665e-02, -1.4196e-02, -1.0832e-02, -1.2535e-02,
    3.5567e-03,  1.2518e-03,  1.6872e-03,  -2.3664e-02, -2.2614e-02,
    -2.2188e-03, -1.8959e-02, 1.2989e-02,  5.2072e-03,  -1.0576e-02,
    4.8411e-03,  1.1213e-02,  1.2075e-02,  1.0523e-03,  -8.3421e-03,
    2.1203e-03,  3.1107e-03,  3.7737e-03,  -6.2003e-03, 6.1267e-03,
    1.6272e-03,  -8.6477e-03, -7.0274e-03, 8.2779e-04,  1.3744e-03,
    -3.5639e-03, -6.2852e-03, -5.7665e-03, -4.0192e-03};
    */

inline std::unique_ptr<Points> calibrateRing(const Eigen::Ref<Points>& input,
                                             const idx ring_guess,
                                             const RangeOffset& range_offset) {
    const idx n = input.cols();
    if (n < 2) {
        return std::make_unique<Points>(3, 0);
    }

    const Eigen::Matrix<flo, -1, 1> xy_ranges =
        input.topRows(2).colwise().norm();
    const Eigen::Matrix<flo, -1, 1> ranges = input.colwise().norm();
    /*
    Eigen::MatrixXd A(n, 2);
    A.col(0) = xy_ranges;
    A.col(1).setOnes();
    const Eigen::Vector2d slope_intercept =
        A.fullPivLu().solve(input.row(2).transpose());
        */

    idx ring = ring_guess;
    /*
    flo ring_score = 9999;
    for (idx i = 0; i < 64; i++) {
        const flo score = std::abs(slope_intercept[0] - v_slope[i]);
        if (score < ring_score) {
            ring = i;
            ring_score = score;
        }
    }
    std::cerr << ring_guess << ": " << ring << ", " << n << std::endl;
    */

    std::unique_ptr<Points> output = std::make_unique<Points>(3, n);

    for (idx i = 0; i < n; i++) {
        const flo azimuth = std::atan2(input(1, i), input(0, i));
        const flo altitude =
            std::asin((input(2, i) - v_offset[ring]) / ranges(i)) +
            altitude_correction;
        output->col(i) = (ranges(i) + range_offset[ring]) *
                             Point(std::cos(azimuth) * std::cos(altitude),
                                   std::sin(azimuth) * std::cos(altitude),
                                   std::sin(altitude)) +
                         v_offset[ring] * Point::UnitZ();
    }

    return output;
}
}  // namespace velodyne_calibration
}  // namespace salami
