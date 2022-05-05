import math
import numpy as np
'''
道格拉斯-普克抽稀算法
垂距限值抽稀算法
'''
# 设定阈值
THRESHOLD = 0.0001

# 计算点到直线的距离
def point2Line(point_a, point_b, point_c):
    '''
    计算点a到b, c之间的距离
    point_a: (x_a, y_a)
    point_b: (x_b, y_b)
    point_c: (x_c, y_c)
    '''
    # 计算bc的直线
    if point_b[0] == point_c[0]:
        return 9999
    # y = kx+b
    k = (point_b[1]-point_c[1])/(point_b[0]-point_c[0])
    b = point_c[1]-k*point_c[0]

    # 计算点到直线的距离
    distance = abs(k*point_a[0]-point_a[1]+b)/math.sqrt(1+k**2)
    return distance


class DouglasPeuker(object):

    def __init__(self):
        self._threshold=THRESHOLD
        self._qualify_list = []
        self._disqualify_list = []
    
    def diluting(self, point_list):
        '''
        抽稀算法
        : param point_list: 二维点列表
        : return 
        '''
        if len(point_list)<3:
            self._qualify_list.extend(point_list[::-1])
        else:
            # 找到首尾相连的两点
            max_distance_index, max_distance = 0, 0
            for index, point in enumerate(point_list):
                if index in [0, len(point_list) - 1]:
                    continue
                distance = point2Line(point, point_list[0], point_list[-1])
                if distance > max_distance:
                    max_distance_index = index
                    max_distance = distance

            # 若最大距离小于阈值，则去掉所有中间点。 反之，则将曲线按最大距离点分割
            if max_distance < self._threshold:
                self._qualify_list.append(point_list[-1])
                self._qualify_list.append(point_list[0])
            else:
                # 将曲线按最大距离的点分割成两段
                sequence_a = point_list[:max_distance_index]
                sequence_b = point_list[max_distance_index:]

                for sequence in [sequence_a, sequence_b]:
                    if len(sequence) < 3 and sequence == sequence_b:
                        self._qualify_list.extend(sequence[::-1])
                    else:
                        self._disqualify_list.append(sequence)
    def main(self, point_list):
        self.diluting(point_list)
        while len(self._disqualify_list) > 0:
            self.diluting(self._disqualify_list.pop())
        print(self._qualify_list)
        print(len(self._qualify_list))

class LimitVerticalDistance(object):
    def __init__(self):
        self._threshold = THRESHOLD
        self._qualify_list = []
    
    def diluting(self, point_list):
        self._qualify_list.append(point_list[0])
        check_index = 1
        while check_index<len(point_list)-1:
            distance = point2Line(point_list[check_index], self._qualify_list[-1], point_list[check_index+1])

            if distance<self._threshold:
                check_index+=1 
            else:
                self._qualify_list.append(point_list[check_index])
                check_index+=1
        return self._qualify_list
        



if __name__ == '__main__':
    d = DouglasPeuker()
    d.main([[104.066228, 30.644527], [104.066279, 30.643528], [104.066296, 30.642528], [104.066314, 30.641529],
            [104.066332, 30.640529], [104.066383, 30.639530], [104.066400, 30.638530], [104.066451, 30.637531],
            [104.066468, 30.636532], [104.066518, 30.635533], [104.066535, 30.634533], [104.066586, 30.633534],
            [104.066636, 30.632536], [104.066686, 30.631537], [104.066735, 30.630538], [104.066785, 30.629539],
            [104.066802, 30.628539], [104.066820, 30.627540], [104.066871, 30.626541], [104.066888, 30.625541],
            [104.066906, 30.624541], [104.066924, 30.623541], [104.066942, 30.622542], [104.066960, 30.621542],
            [104.067011, 30.620543], [104.066122, 30.620086], [104.065124, 30.620021], [104.064124, 30.620022],
            [104.063124, 30.619990], [104.062125, 30.619958], [104.061125, 30.619926], [104.060126, 30.619894],
            [104.059126, 30.619895], [104.058127, 30.619928], [104.057518, 30.620722], [104.057625, 30.621716],
            [104.057735, 30.622710], [104.057878, 30.623700], [104.057984, 30.624694], [104.058094, 30.625688],
            [104.058204, 30.626682], [104.058315, 30.627676], [104.058425, 30.628670], [104.058502, 30.629667],
            [104.058518, 30.630667], [104.058503, 30.631667], [104.058521, 30.632666], [104.057664, 30.633182],
            [104.056664, 30.633174], [104.055664, 30.633166], [104.054672, 30.633289], [104.053758, 30.633694],
            [104.052852, 30.634118], [104.052623, 30.635091], [104.053145, 30.635945], [104.053675, 30.636793],
            [104.054200, 30.637643], [104.054756, 30.638475], [104.055295, 30.639317], [104.055843, 30.640153],
            [104.056387, 30.640993], [104.056933, 30.641830], [104.057478, 30.642669], [104.058023, 30.643507],
            [104.058595, 30.644327], [104.059152, 30.645158], [104.059663, 30.646018], [104.060171, 30.646879],
            [104.061170, 30.646855], [104.062168, 30.646781], [104.063167, 30.646823], [104.064167, 30.646814],
            [104.065163, 30.646725], [104.066157, 30.646618], [104.066231, 30.645620], [104.066247, 30.644621]])

