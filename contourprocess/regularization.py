import cv2
import numpy as np
from .rdp_alg import rdp
from .cal_dist_ang import cal_angle, cal_dist, azimuthAngle
from .rotate_ang import Nrotation_angle_get_coor_coordinates, Srotation_angle_get_coor_coordinates
from .line_intersection import line, intersection, par_line_dist, point_in_line



def boundary_regularization(img, epsilon=6):
    h, w = img.shape[0:2]
    # 轮廓定位
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = np.squeeze(contours[0])
    # 轮廓精简（DP）
    contours = rdp(contours, epsilon=epsilon)
    contours[:, 1] = h - contours[:, 1]
    # 轮廓规则化
    dists = []
    azis = []
    azis_index = []
    # 获取每条边的长度和方位角
    for i in range(contours.shape[0]):
        cur_index = i
        next_index = i+1 if i < contours.shape[0]-1 else 0
        prev_index = i-1
        cur_point = contours[cur_index]
        nest_point = contours[next_index]
        prev_point = contours[prev_index]

        dist = cal_dist(cur_point, nest_point)
        azi = azimuthAngle(cur_point, nest_point)

        dists.append(dist)
        azis.append(azi)
        azis_index.append([cur_index, next_index])

    # 以最长的边的方向作为主方向
    longest_edge_idex = np.argmax(dists)
    main_direction = azis[longest_edge_idex]
    # 方向纠正，绕中心点旋转到与主方向垂直或者平行
    correct_points = []
    para_vetr_idxs = []  # 0平行 1垂直
    for i, (azi, (point_0_index, point_1_index)) in enumerate(zip(azis, azis_index)):

        if i == longest_edge_idex:
            correct_points.append([contours[point_0_index], contours[point_1_index]])
            para_vetr_idxs.append(0)
        else:
            # 确定旋转角度
            rotate_ang = main_direction - azi

            if np.abs(rotate_ang) < 180/4:
                rotate_ang = rotate_ang
                para_vetr_idxs.append(0)
            elif np.abs(rotate_ang) >= 90-180/4:
                rotate_ang = rotate_ang + 90
                para_vetr_idxs.append(1)

            # 执行旋转任务
            point_0 = contours[point_0_index]
            point_1 = contours[point_1_index]
            point_middle = (point_0 + point_1) / 2

            if rotate_ang > 0:
                rotate_point_0 = Srotation_angle_get_coor_coordinates(point_0, point_middle, np.abs(rotate_ang))
                rotate_point_1 = Srotation_angle_get_coor_coordinates(point_1, point_middle, np.abs(rotate_ang))
            elif rotate_ang < 0:
                rotate_point_0 = Nrotation_angle_get_coor_coordinates(point_0, point_middle, np.abs(rotate_ang))
                rotate_point_1 = Nrotation_angle_get_coor_coordinates(point_1, point_middle, np.abs(rotate_ang))
            else:
                rotate_point_0 = point_0
                rotate_point_1 = point_1
            correct_points.append([rotate_point_0, rotate_point_1])

    correct_points = np.array(correct_points)


    # 相邻边校正，垂直取交点，平行平移短边或者加线
    final_points = []
    final_points.append(correct_points[0][0])
    for i in range(correct_points.shape[0]-1):
        cur_index = i
        next_index = i + 1 if i < correct_points.shape[0] - 1 else 0

        cur_edge_point_0 = correct_points[cur_index][0]
        cur_edge_point_1 = correct_points[cur_index][1]
        next_edge_point_0 = correct_points[next_index][0]
        next_edge_point_1 = correct_points[next_index][1]

        cur_para_vetr_idx = para_vetr_idxs[cur_index]
        next_para_vetr_idx = para_vetr_idxs[next_index]

        if cur_para_vetr_idx != next_para_vetr_idx:
            # 垂直取交点
            L1 = line(cur_edge_point_0, cur_edge_point_1)
            L2 = line(next_edge_point_0, next_edge_point_1)

            point_intersection = intersection(L1, L2)
            final_points.append(point_intersection)

        elif cur_para_vetr_idx == next_para_vetr_idx:
            # 平行分两种，一种加短线，一种平移，取决于距离阈值
            L1 = line(cur_edge_point_0, cur_edge_point_1)
            L2 = line(next_edge_point_0, next_edge_point_1)
            marg = par_line_dist(L1, L2)

            if marg < 3:
                # 平移
                point_move = point_in_line(next_edge_point_0[0], next_edge_point_0[1], cur_edge_point_0[0], cur_edge_point_0[1], cur_edge_point_1[0], cur_edge_point_1[1])
                final_points.append(point_move)
                # 更新平移之后的下一条边
                correct_points[next_index][0] = point_move
                correct_points[next_index][1] = point_in_line(next_edge_point_1[0], next_edge_point_1[1], cur_edge_point_0[0], cur_edge_point_0[1], cur_edge_point_1[0], cur_edge_point_1[1])


            else:
                # 加线
                add_mid_point = (cur_edge_point_1 + next_edge_point_0) / 2
                add_point_1 = point_in_line(add_mid_point[0], add_mid_point[1], cur_edge_point_0[0], cur_edge_point_0[1], cur_edge_point_1[0], cur_edge_point_1[1])
                add_point_2 = point_in_line(add_mid_point[0], add_mid_point[1], next_edge_point_0[0], next_edge_point_0[1], next_edge_point_1[0], next_edge_point_1[1])
                final_points.append(add_point_1)
                final_points.append(add_point_2)


    final_points.append(final_points[0])
    final_points = np.array(final_points)

    final_points[:, 1] = h - final_points[:, 1]
    return final_points
