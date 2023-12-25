import taichi as ti

import core.const as const
from core.sph.sph_base import SPHBase


class WCSPH(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        self.exponent = 7.0
        self.stiffness = 50.0

        self.d_velocity = ti.Vector.field(self.ps.dim, dtype=float)
        particle_node = ti.root.dense(ti.i, self.ps.particle_max_num)
        particle_node.place(self.d_velocity)

    # 计算密度
    @ti.kernel
    def compute_densities(self):
        for p_i in range(self.ps.particle_num[None]):
            x_i = self.ps.x[p_i]
            self.ps.density[p_i] = 0.0
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                if self.ps.material[p_j] == self.ps.material_fluid:
                    self.ps.density[p_i] += self.ps.m_V * self.cubic_kernel((x_i - x_j).norm())
                    pass
            #     elif self.ps.material[p_j] == self.ps.material_boundary:
            #         self.ps.density[p_i] += self.density_0*self.ps.volume[p_j] * self.cubic_kernel((x_i - x_j).norm())
            #         # self.ps.density[p_i] += 0*self.cubic_kernel((x_i - x_j).norm)
            self.ps.density[p_i] *= self.density_0

    # 计算压力以及加速度(包括重力作用）
    @ti.kernel
    def compute_pressure_force(self):
        for p_i in range(self.ps.particle_num[None]):
            self.ps.density[p_i] = ti.max(self.ps.density[p_i], self.density_0)
            # wcsph 的关键
            self.ps.pressure[p_i] = self.stiffness * (ti.pow(self.ps.density[p_i] / self.density_0, self.exponent) - 1.0)

        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                d_v += self.pressure_force(p_i, p_j, self.ps.x[p_i] - self.ps.x[p_j])
            self.d_velocity[p_i] += d_v

    # 计算粘性力以及加速度
    @ti.kernel
    def compute_non_pressure_force(self):
        for p_i in range(self.ps.particle_num[None]):
            x_i = self.ps.x[p_i]
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
            d_v[self.ps.dim - 1] = const.g

            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                d_v += self.viscosity_force(p_i, p_j, x_i - x_j)
            self.d_velocity[p_i] = d_v

    @ti.kernel
    def advert(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_fluid:
                self.ps.v[p_i] += self.dt[None] * self.d_velocity[p_i]
                self.ps.x[p_i] += self.dt[None] * self.ps.v[p_i]

    def substep(self):
        self.compute_densities()
        self.compute_non_pressure_force()
        self.compute_pressure_force()
        self.advert()

