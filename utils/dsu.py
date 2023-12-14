import math

class DSU:
    """使用并查集算法进行对点进行分类, 距离小于r的点将被分为一类点
    """
    def __init__(self, points, r) -> None:
        self.points = points
        self.father = [i for i in range(len(points))]
        self.r = r
    
    def union(self, other) -> None:
        pass
    
    def getFa(self, i) -> int:
        if self.father[i] == i:
            return i
        self.father[i] = self.getFa(self.father[i])
        return self.father[i]
    
    def union(self, x, y) -> None:
        x_father = self.getFa(x)
        y_father = self.getFa(y)
        if x_father == y_father:
            return
        self.father[y_father] = x_father
    def get_distance(self, x, y) -> float:
        return math.sqrt((y[1] - x[1])*(y[1] - x[1]) + (y[0] - x[0])*(y[0] - x[0]))

    def getAllGroups(self):
        count = len(self.points)
        for i in range(count):
            for j in range(count):
                if i == j:
                    continue
                if self.get_distance(self.points[i], self.points[j]) < self.r:
                    self.union(i, j)
            
        for i in range(len(self.points)):
            self.father[i] = self.getFa(i)
        
        map = {}
        groups = []
        cnt = 0
        for i in range(len(self.points)):
            fai = self.father[i]
            if map.get(fai) == None:
                map[fai] = cnt
                groups.append([i])
                cnt += 1
            else:
                groups[map[fai]].append(i)
        return groups
    
