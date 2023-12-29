import numpy as np
import taichi as ti
ti.init(arch=ti.cuda, debug=False)


data = np.array([1 , 2 ,3])

@ti.kernel
def helper(positions:ti.template(), prefix_sum:ti.template(),count: ti.template(), ids: ti.template(), ret:ti.template()):
    for i in ti.grouped(positions):
        offset = 0
        if ids[i] > 0:
            offset = prefix_sum[int(ids[i] - 1)]
        # print(ti.atomic_add(count[int(ids[i])], 1) + offset, i)
        ret[ti.atomic_add(count[int(ids[i])], 1)+ offset] = positions[i]

def resortObjects(postions, ids, object_counts, particle_num:int, object_num:int):
    prefix_sum = ti.field(dtype=ti.i32, shape=object_num)
    prefix_sum.from_numpy(object_counts)
    prefix_sum_executor = ti.algorithms.PrefixSumExecutor(particle_num)
    prefix_sum_executor.run(prefix_sum)
    ret = ti.Vector.field(3, dtype=ti.f32, shape=particle_num) # the particle position
    count =  ti.field(dtype=ti.i32, shape=object_num)
    helper(postions,prefix_sum,count, ids,ret)
    for i in range(particle_num):
        postions[i] = ret[i]



pos = ti.Vector.field(3, dtype=ti.f32, shape=6) # the particle position
pos_arr = np.array([[1, 1, 1], [1, 2, 2], [1, 3, 3], [1, 4, 4], [1, 5, 5], [1, 6, 6]])
ids = ti.field(dtype=ti.f32, shape=6) # the particle position
ids_arr = np.array([0, 2, 2, 2, 1, 1])
pos.from_numpy(pos_arr)
ids.from_numpy(ids_arr)
print(pos)
resortObjects(postions= pos,ids= ids,object_counts= data,particle_num= 6,object_num= 3 )
print(pos)
