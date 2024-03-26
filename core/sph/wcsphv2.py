import taichi as ti

import core.const as const
from core.sph.sph_basev2 import SPHBaseV2


class WCSPHV2(SPHBaseV2):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        self.exponent = 7.0
        self.stiffness = 50.0

        self.d_velocity = ti.Vector.field(self.ps.dim, dtype=float)
        particle_node = ti.root.dense(ti.i, self.ps.particle_max_num)
        particle_node.place(self.d_velocity)
        self.c_s = self.ps.configuration['c_s']

    @ti.func
    def compute_density_task(self, p_i, p_j, ret: ti.template()):
        x_i = self.ps.x[p_i]
        x_j = self.ps.x[p_j]
        if self.ps.material[p_i] == self.ps.material_fluid:
            ret += self.ps.mass[p_i] * self.cubic_kernel((x_i - x_j).norm())
        elif self.ps.material[p_i] == self.ps.material_boundary:
            ret += self.density_0*self.ps.volume[p_j] * self.cubic_kernel((x_i - x_j).norm())
            
    # 计算密度
    @ti.kernel
    def compute_densities(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_fluid:
                density = self.ps.mass[p_i] * self.cubic_kernel(0.0)
                self.ps.for_all_neighbors(p_i, self.compute_density_task, self.ps.density[p_i])
                self.ps.density[p_i] = density

    @ti.func
    def compute_pressure_force_task(self, p_i, p_j, ret: ti.template()):
        x_i  = self.ps.x[p_i]
        x_j = self.ps.x[p_j]
        ret += self.pressure_force(p_i, p_j, x_i - x_j)
        
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
            self.ps.for_all_neighbors(p_i, self.compute_pressure_force_task, d_v)
            self.d_velocity[p_i] += d_v
            # print(self.ps.pressure[p_i], self.ps.density[p_i])
    @ti.func
    def compute_non_pressure_force_task(self, p_i, p_j, ret:ti.template()):
        x_i = self.ps.x[p_i]
        x_j = self.ps.x[p_i]
        # surface tensor
        if self.ps.material[p_j] == self.ps.material_fluid:
            r_vec = self.ps.x[p_i] - self.ps.x[p_j]
            # if r_vec.norm() > self.ps.particle_diameter:
            ret -= 0.01 / self.ps.mass[p_i] * self.ps.mass[p_j] * r_vec * \
                self.cubic_kernel(r_vec.norm())
                
        # for fluid neighbours
        if self.ps.material[p_j] == self.ps.material_fluid:
            nu = 2*self.viscosity*self.ps.support_length * self.c_s / (self.ps.density[p_i] + self.ps.density[p_j])
            v_ij = self.ps.v[p_i] - self.ps.v[p_j]
            x_ij = self.ps.x[p_i] - self.ps.x[p_j]
            pi = -nu * ti.min(0, v_ij.dot(x_ij))/ (x_ij.dot(x_ij) + 0.01*self.ps.support_length**2)
            ret -= self.ps.mass[p_j] * pi * self.cubic_kernel_derivative(x_ij)
        else:
            sigma = 0.08
            nu = sigma * self.ps.support_length * self.c_s / (2 * self.ps.density[p_i])  # eq (14)
            v_ij = self.ps.v[p_i] - self.ps.v[p_j]
            x_ij = self.ps.x[p_i] - self.ps.x[p_j]
            pi = -nu * ti.min(v_ij.dot(x_ij), 0.0) / (x_ij.dot(x_ij) + 0.01 * self.ps.support_length ** 2)  # eq (11)
            ret -= self.ps.density0 * self.ps.volume[p_j] * pi * self.cubic_kernel_derivative(x_ij)  # eq (13)
        
    # 计算粘性力以及加速度
    @ti.kernel
    def compute_non_pressure_force(self):
        for p_i in range(self.ps.particle_num[None]):
            x_i = self.ps.x[p_i]
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
            for i in ti.static(range(3)):
                d_v[i] = self.g[i]
            self.ps.for_all_neighbors(p_i, self.compute_non_pressure_force_task, d_v)
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

