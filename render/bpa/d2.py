import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

import taichi as ti
import numpy as np
ti.init(arch=ti.cpu)
from utils.dsu import DSU



@ti.data_oriented
class BPA:

    def __init__(self, points, radius):
        self.points = ti.Vector.field(2, dtype=ti.f64, shape=len(points))
        for i in range(len(points)):
            self.points[i] = points[i]
        dsu = DSU(points, radius)
        groups = dsu.getAllGroups()
        # The total count of groups
        self.group_count = len(groups)
        self.groups = ti.field(int)
        # 每个group中的数目
        self.group_num = ti.field(int)
        self.node = ti.root.dense(ti.i, len(groups))
        self.node.place(self.group_num)
        self.group_node = self.node.dense(ti.j, len(points))
        self.group_node.place(self.groups)
        for i in range(len(groups)):
            self.group_num[i] = len(groups[i])
            for j in range(len(groups[i])):
                self.groups[i, j] = groups[i][j]
        self.radius = radius
        self.len = len(points)
        # 点被visited过
        self.visited = ti.field(ti.i32, self.len)
        self.edge_node = ti.root.dynamic(ti.i, 1024, chunk_size=1024)
        self.edge = ti.field(ti.i32)
        self.edge_node.place(self.edge)
        self.edge_count = ti.field(ti.i32, shape=(1))
    @ti.func
    def get_angle(self, x: ti.f64, y: ti.f64) -> ti.f32:
        """Get the angle of (x, y) vector

        Args:
            x (ti.f64): x 
            y (ti.f64): y

        Returns:
            ti.f32: the angle
        """
        return ti.atan2(y, x) * 180 / np.pi

    @ti.func
    def get_angle_of_vector(self, x: ti.math.vec2, y: ti.math.vec2) -> ti.f32:
        """Get the clockwise angle of two vector

        Args:
            x (ti.math.vec2): the base vector
            y (ti.math.vec2): the target vector

        Returns:
            ti.f32: the angle
        """
        vector_prod = x[0] * y[0] + x[1] * y[1]
        cross_prod = x[0] * y[1] - x[1] * y[0]
        rho = -ti.math.degrees(ti.atan2(cross_prod, vector_prod))
        if rho < 0:
            rho = 360 + rho
        return rho

    @ti.func
    def get_next_point(self, group_index, point_index, circle_pos) -> ti.math.vec2:
        center = self.points[point_index]
        nextIndex = -1
        nextTheta = 360.0
        self.visited[point_index] = 1
        for i in range(self.group_num[group_index]):
            targetIndex = self.groups[group_index, i]
            if self.visited[targetIndex] == 1:
                continue
            baseVector = ti.math.vec2(circle_pos[0] - center[0], circle_pos[1] - center[1])
            targetVector = ti.math.vec2(self.points[targetIndex][0] - center[0], self.points[targetIndex][1] - center[1])
            theta = self.get_angle_of_vector(baseVector, targetVector)
            # print("base:{} circle: {} theta: {} targetPoint: {} baseV:{} targetV: {}".format(center,circle_pos, theta, self.points[targetIndex], baseVector, targetVector))
            if theta < nextTheta:
                nextIndex = targetIndex
                nextTheta = theta
        if nextIndex != -1:
            self.visited[nextIndex] = 1
        return ti.math.vec2(nextIndex, nextTheta)
    
    @ti.func
    def update_circle_pos(self, circle_pos, center_pos, theta) -> ti.math.vec2:
        """更新原点的信息
        Args:
            circle_pos (_type_): _description_
            center_pos (_type_): _description_
        """
        newx = (circle_pos[0] - center_pos[0])*ti.cos(theta/ti.math.pi) - (circle_pos[1] - center_pos[1])*ti.sin(theta/ti.math.pi) 
        newy = (circle_pos[0] - center_pos[0])*ti.sin(theta/ti.math.pi) - (circle_pos[1] - center_pos[1])*ti.cos(theta/ti.math.pi) 
        return ti.math.vec2(newx, newy)
    
    @ti.func
    def render_signle_group(self, index):
        # data = self.groups[index]
        # 选取开始的点，从最高处开始选择
        highest = 0
        for i in range(self.group_num[index]):
            j = self.groups[index, i]
            if self.points[j][1] > self.points[highest][1]:
                highest = j
        # 初始点必然是我们边界上的点
        self.edge.append(highest)
        self.edge_count[0] += 1
        next = highest
        print(self.points[highest])
        circle_pos = ti.math.vec2(self.points[next][0], self.points[next][1] + self.radius)
        while True:
            ans = self.get_next_point(index, next, circle_pos)
            circle_pos = self.update_circle_pos(circle_pos, self.points[next], ans[1])
            if ans[0] == -1:
                break
            next = ans[0]
            print("next ", self.points[next])
            self.edge.append(next)
            self.edge_count[0] += 1
            

    @ti.kernel
    def render(self):
        for i in range(self.group_count):
            self.render_signle_group(i)
        
        # self.get_neighbors()
        
num_rows = np.random.randint(50, 101)
num_cols = np.random.randint(50, 101)
random_array = np.random.rand(100, 2)
arr = random_array*(100 - 50) + 50
x = arr[:, 0]
y = arr[:, 1]
plt.scatter(x, y, marker='o', s=50, alpha=0.7, edgecolors='k')
plt.show()
# arr = np.array([[100, 50], [50, 100], [50, 50], [100, 100], [50, 200]])
bpa = BPA(arr, 50)
bpa.render()
edge_count = bpa.edge_count[0]

gui = ti.GUI("bpa", res=(400, 400))
result = np.zeros(shape=(edge_count,2))
for i in range(edge_count):
    insert = (1, 1)
    result[i] = arr[bpa.edge[i]]
    # result = np.append(result, insert)
base = arr[bpa.edge[0]]
# edge_count-=1
first = np.array([base for _ in range(0, edge_count-2)])
second =  np.array([result[i] for i in range(1, edge_count-1)])
third = np.array([result[i] for i in range(2, edge_count)])
print("first", first)
print("second", second)
print("third",third)
print(result)
while gui.running:
    # gui.circles(arr/500, radius=10, palette=[0x068587], palette_indices=[0, 0, 0, 0])

    gui.triangles(first/200, second/200, third/200, color=0xEEEEF0)
    # for i in range(1, bpa.edge_count[0]-1):
        # gui.triangle(a=result[i]/500,b=result[i+1]/500, c=result[i+2]/500, color=0xEEEEF0)
    gui.show()
