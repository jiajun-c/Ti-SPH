import math

center = [50, 50]
ans = []



for i in range(12):
    ans.append([center[0]+30*math.cos(i*math.pi/6), center[1]+30*math.sin(i*math.pi/6)])
    

print(ans)