import taichi as ti
import numpy as np
from functools import reduce


@ti.data_oriented
class ParticleSystem:
    def __init__(self, res):
        self.res = res
        self.dim = len(res)
        # 粒子系统的整体信息
        self.particle_num = ti.field(int, shape=())
        self.x = ti.Vector.field(self.dim, dtype=float)  # 粒子的位置
        self.v = ti.Vector.field(self.dim, dtype=float)  # 粒子的速度
        self.density = ti.field(dtype=float)  # 粒子的密度
        self.material = ti.field(dtype=float)  # 粒子的材质
        self.pressure = ti.field(dtype=float)  # 粒子受到的压力
        self.color = ti.field(dtype=int)  # 粒子的颜色
        self.material_fluid = 1  # 流体材质
        self.material_bound = 0  # 边界材质
        self.particle_radius = 0.05  # 粒子半径

    @ti.func
    def add_particle(self, p, x, v, density, pressure, material, color):
        self.x[p] = x
        self.v[p] = v
        self.density[p] = density
        self.pressure[p] = pressure
        self.material[p] = material
        self.color[p] = color

    @ti.kernel
    def add_particles(self, num: int,
                      particle_position: ti.types.ndarray(),
                      particle_velocity: ti.types.ndarray(),
                      particle_density: ti.types.ndarray(),
                      particle_pressure: ti.types.ndarray(),
                      particle_material: ti.types.ndarray(),
                      particle_color: ti.types.ndarray()):
        for i in range(num):
            v = ti.Vector.zero(float, self.dim)
            x = ti.Vector.zero(float, self.dim)
            for j in range(self.dim):
                v[j] = particle_velocity[i]
                x[j] = particle_position[i]
            self.add_particle(self.particle_num[None] + i, x, v,
                              particle_density[i],
                              particle_pressure[i],
                              particle_material[i],
                              particle_color[i])
        self.particle_num[None] += num

    def add_cube(self,
                 lower_corner,
                 cube_size,
                 color=0xFFFFFF,
                 velocity=None,
                 density=None,
                 pressure=None,
                 ):
        num_dim = []
        for i in range(self.dim):
            num_dim.append(np.arange(lower_corner[i], lower_corner[i] + cube_size[i], self.particle_radius))
        particle_num = reduce(lambda x, y: x * y, [len(n) for n in num_dim])
        positions = np.array(np.meshgrid(*num_dim,
                                         sparse=False,
                                         indexing='ij'),
                             dtype=np.float32)
        positions = positions.reshape(-1, reduce(lambda x, y: x * y, list(positions.shape[1:]))).transpose()
        velocity = np.full(positions.shape, fill_value=0 if velocity is None else velocity, dtype=np.float32)
        density = np.full(positions.shape, fill_value=1000.0 if density is None else density, dtype=np.float32)
        color = np.full(positions.shape, fill_value=color, dtype=np.float32)
        self.add_particles(particle_num,
                           positions,
                           velocity,
                           density,
                           pressure,
                           color)

