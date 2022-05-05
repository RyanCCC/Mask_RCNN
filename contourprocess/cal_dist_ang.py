import numpy as np
import math


# 计算两点的距离
def cal_dist(point_1, point_2):
    diff  = point_1-point_2
    dist = math.hypot(diff[0], diff[1])
    # dist = np.sqrt(np.sum(math.pow((point_1-point_2), 2)))
    return dist

# 计算两条直线的夹角  余弦定理
def cal_angle(point_1, point_2, point_3):
    '''
    根据三个坐标计算夹角
    :param point_1：点1坐标
    :param point_2：点2的坐标
    :param point_3：点3的坐标
    :return：返回任意角的夹角值
    '''
    a = math.sqrt(
        (point_2[0]-point_3[0])*(point_2[0]-point_3[0])+(point_2[1]-point_3[1])*(point_2[1] - point_3[1])
    )
    b = math.sqrt(
        (point_1[0]-point_3[0])*(point_1[0]-point_3[0])+(point_1[1]-point_3[1])*(point_1[1] - point_3[1])
    )
    c = math.sqrt(
        (point_1[0]-point_2[0])*(point_1[0]-point_2[0])+(point_1[1]-point_2[1])*(point_1[1] - point_2[1])
    )
    A=math.degrees(math.acos((a*a-b*b-c*c)/(-2*b*c)))
    B=math.degrees(math.acos((b*b-a*a-c*c)/(-2*a*c)))
    C=math.degrees(math.acos((c*c-a*a-b*b)/(-2*a*b)))
    return A, B, C

# 计算线条的方位角
def azimuthAngle(point_0, point_1):
    x1, y1 = point_0
    x2, y2 = point_1
    if x1 < x2:
        if y1 < y2:
            ang = math.atan((y2 - y1) / (x2 - x1))
            ang = ang * 180 / math.pi
            return ang
        elif y1 > y2:
            ang = math.atan((y1 - y2) / (x2 - x1))
            ang = ang * 180 / math.pi
            return 90 + (90 - ang)
        elif y1==y2:
            return 0
    elif x1 > x2:
        if y1 < y2:
            ang = math.atan((y2-y1)/(x1-x2))
            ang = ang*180/math.pi
            return 90+(90-ang)
        elif y1 > y2:
            ang = math.atan((y1-y2)/(x1-x2))
            ang = ang * 180 / math.pi
            return ang
        elif y1==y2:
            return 0

    elif x1==x2:
        return 90

if __name__ == '__main__':
    pass