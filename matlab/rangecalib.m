close all;
range_offsets = zeros(64, 1);
range_offset_n = zeros(64, 1);
%points = dlmread('/mnt/data/kitti/2011_09_26/2011_09_26_drive_0119_extract/velodyne_points/data/0000000010.txt')(:,1:3);

for frame = 0:4540
    fid = fopen(sprintf('/mnt/data/kitti/dataset/sequences/00/velodyne/%06d.bin',frame),'rb');
    points = fread(fid,[4 inf],'single')'(:,1:3);

    azimuth = atan2(points(:,2), points(:,1));
    xyrange = sqrt(points(:,2).^2 + points(:,1).^2);

    n = size(points, 1);
    laser_begin = ones(64, 1);
    laser_end = ones(64, 1);
    laser_begin(1) = 1;
    laser_end(64) = n;
    laser = 1;
    for ind = 2 : n
        if azimuth(ind) >= 0 && azimuth(ind - 1) < 0
            laser_begin(laser + 1) = ind + 1;
            laser_end(laser) = ind - 2;
            laser = laser + 1;
        end
    end

    v_offset = zeros(64, 1);
    v_slope = zeros(64, 1);
    for lind = 1:64
        ind = laser_begin(lind) : laser_end(lind);
        %plot(xyrange(ind), points(ind, 3), '.');
        A = [xyrange(ind), ones(size(ind, 2), 1)];
        b = points(ind, 3);
        x = A\b;
        bb = A * x;
        v_offset(lind) = x(2);
        v_slope(lind) = x(1);
    end


    points_corrected = [];

    n_regions = 64;
    wall_thresh = 0.2;
    for region = 1:n_regions
        region_end = -pi + 2 * pi * region / n_regions;
        region_start = -pi + 2 * pi * (region - 1) / n_regions;
        flat_wall_region = azimuth > region_start & azimuth <= region_end;
        median_xyrange = median(xyrange(flat_wall_region));
        flat_wall_region_good = flat_wall_region & (abs(xyrange - median_xyrange) < wall_thresh);
        flat_wall = points(flat_wall_region_good, :);
        if size(flat_wall, 1) < 100
            continue;
        end
        %plot3(flat_wall(:,1), flat_wall(:,2), flat_wall(:,3), '.');

        flat_wall_mean = mean(flat_wall);
        flat_wall_cov = cov(flat_wall, 1);
        [eigvec, eigval] = eig(flat_wall_cov);
        [~, idx] = min(diag(eigval));
        flat_wall_norm = eigvec(:, idx);
        flat_wall_offset = dot(flat_wall_mean, flat_wall_norm);

        for lind = 1:64
            ind = laser_begin(lind) : laser_end(lind);

            points_relevant = points(ind, :);
            points_relevant(:,3) = points_relevant(:,3) - v_offset(lind);
            azimuths_relevant = atan2(points_relevant(:,2), points_relevant(:,1));
            calib_ind = azimuths_relevant > region_start & azimuths_relevant <= region_end;

            ranges_relevant = sqrt(sum(points_relevant.^2, 2));

            ranges_for_calib = ranges_relevant(calib_ind);
            if ~isempty(ranges_for_calib)
                dir_for_calib = points_relevant(calib_ind, :);
                dir_for_calib = dir_for_calib ./ [ranges_for_calib, ranges_for_calib, ranges_for_calib];
                ranges_est = flat_wall_offset ./ (dir_for_calib * flat_wall_norm);
                %disp(['laser ' num2str(lind)]);
                range_offset = mean(ranges_est - ranges_for_calib);
                if abs(range_offset) < 0.2
                    ranges_corrected = ranges_relevant + range_offset;
                    range_offsets(lind) = range_offsets(lind) + range_offset;
                    range_offset_n(lind) = range_offset_n(lind) + 1;
                end
            end
        end
    end

    %{
    for lind = 1:64
        ind = laser_begin(lind) : laser_end(lind);

        points_relevant = points(ind, :);
        points_relevant(:,3) = points_relevant(:,3) - v_offset(lind);
        azimuths_relevant = atan2(points_relevant(:,2), points_relevant(:,1));

        ranges_relevant = sqrt(sum(points_relevant.^2, 2)) + range_offsets(lind) / range_offset_n(lind);
        ranges_relevant_corrected = ranges_relevant + range_offsets(lind) / range_offset_n(lind);
        points_relevant_corrected = bsxfun(@times, ranges_relevant_corrected ./ ranges_relevant, points_relevant);
        points_relevant_corrected(:,3) = points_relevant_corrected(:,3) + v_offset(lind);
        points_corrected = [points_corrected; points_relevant_corrected];
    end
    %}

    fclose(fid);
end
%{
figure;
plot3(points(:, 1), points(:, 2), points(:, 3), 'b.');
hold on;
plot3(points_corrected(:, 1), points_corrected(:, 2), points_corrected(:, 3), 'r.');
dlmwrite('/home/dllu/points_uncorrected.txt', points);
dlmwrite('/home/dllu/points_corrected2.txt', points_corrected);
axis equal;
%}
figure;
plot(range_offsets ./ range_offset_n, '.');
