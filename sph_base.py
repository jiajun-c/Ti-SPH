import taichi as ti
import numpy as np

import const


@ti.data_oriented
class SPHBase:

    def __init__(self, particle_system):
        self.ps = particle_system  # 微分系统
        self.g = -9.80  # 重力
        self.viscosity = 0.05  # 粘性系数
        self.density_0 = 1000.0  # 水的密度
        self.dt = ti.field(float, shape=())
        self.dt[None] = 2e-4
        self.mass = self.ps.m_V * self.density_0 # 单位体积内的质量

    @ti.func
    def cubic_kernel(self, r_norm):
        res = 0.0
        h = self.ps.support_radius
        k = 1.0
        if self.ps.dim == 1:
            k = 4 / 3
        elif self.ps.dim == 2:
            k = 40 / (7 * np.pi)
        elif self.ps.dim == 3:
            k = 8 / np.pi
        k /= h ** self.ps.dim
        q = r_norm / h
        if q <= 1.0:
            if q <= 0.5:
                res = k * (6.0 * (q ** 3 - q ** 2) + 1)
            else:
                res = k * 2 * ((1 - q) ** 3)
        return res

    @ti.func
    def cubic_kernel_derivative(self, r):
        h = self.ps.support_radius
        res = ti.Vector([0.0 for _ in range(self.ps.dim)])
        k = 1.0
        if self.ps.dim == 1:
            k = 4 / 3
        elif self.ps.dim == 2:
            k = 40 / (7 * np.pi)
        elif self.ps.dim == 3:
            k = 8 / np.pi
        k = 6.0 * k / h ** self.ps.dim
        r_norm = r.norm()
        q = r_norm / h
        if r_norm < const.limit or q > 1.0:
            return res
        grad_q = r / (r_norm, h)
        if q <= 0.5:
            res = k * q * (3.0 * q - 2.0) * grad_q
        else:
            res = k * (1 - q) ** 2 * grad_q

    # 液体的压力
    @ti.func
    def pressure_force(self, p_i, p_j, r):
        res = -self.density_0 * self.ps.m_V * (self.ps.pressure[p_i] / self.ps.density[p_i] ** 2
                                               + self.ps.pressure[p_j] / self.ps.density[p_j] ** 2) \
              * self.cubic_kernel_derivative(r)
        return res

    # 液体的粘性力
    @ti.func
    def viscosity_force(self, p_i, p_j, r):
        # Compute the viscosity force contribution
        v_xy = (self.ps.v[p_i] -
                self.ps.v[p_j]).dot(r)
        res = 2 * (self.ps.dim + 2) * self.viscosity * (self.mass / (self.ps.density[p_j])) * v_xy / (
                r.norm() ** 2 + 0.01 * self.ps.support_radius ** 2) * self.cubic_kernel_derivative(
            r)
        return res
