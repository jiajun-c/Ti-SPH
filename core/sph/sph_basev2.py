import taichi as ti
import numpy as np
from core.partice_system.partice_systemv4 import ParticleSystemV4
import core.const


@ti.data_oriented
class SPHBaseV2:

    def __init__(self, particle_system: ParticleSystemV4):
        self.ps = particle_system  # 微分系统
        self.viscosity = 0.05  # 粘性系数
        self.density_0 = 1000.0  # 水 的密度
        self.dt = ti.field(float, shape=())
        self.dt[None] = 2e-4
        self.g = self.ps.configuration['gravitation']
        # self.mass = self.ps.m_V * self.density_0  # 单位体积内的质量

    @ti.func
    def cubic_kernel(self, r_norm):
        res = 0.0
        h = self.ps.support_length
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
        h = self.ps.support_length
        # derivative of cubic spline smoothing kernel
        k = 1.0
        if self.ps.dim == 1:
            k = 4 / 3
        elif self.ps.dim == 2:
            k = 40 / 7 / np.pi
        elif self.ps.dim == 3:
            k = 8 / np.pi
        k = 6. * k / h ** self.ps.dim
        r_norm = r.norm()
        q = r_norm / h
        res = ti.Vector([0.0 for _ in range(self.ps.dim)])
        if r_norm > 1e-5 and q <= 1.0:
            grad_q = r / (r_norm * h)
            if q <= 0.5:
                res = k * q * (3.0 * q - 2.0) * grad_q
            else:
                factor = 1.0 - q
                res = k * (-factor * factor) * grad_q
        return res

    # 液体的压力
    @ti.func
    def pressure_force(self, p_i, p_j, r):
        # [fixed] support 2D/3D now
        res = ti.Vector([0.0 for _ in range(self.ps.dim)])
        p_rho_i = self.ps.pressure[p_i] / (self.ps.density[p_i] ** 2)
        
        if self.ps.material[p_j] == self.ps.material_fluid:
            # print(self.ps.mass[p_j])
            res = - self.ps.mass[p_j] * (self.ps.pressure[p_i] / self.ps.density[p_i] ** 2
                                               + self.ps.pressure[p_j] / self.ps.density[p_j] ** 2) \
              * self.cubic_kernel_derivative(r)
        elif self.ps.material[p_j] == self.ps.material_boundary:
            res = -self.density_0*self.ps.volume[p_j] * p_rho_i * self.cubic_kernel_derivative(r)
        # print(self.ps.pressure[p_i],self.ps.density[p_i] ,p_rho_i)
        return res

    # 液体的粘性力
    # @ti.func
    # def viscosity_force(self, p_i, p_j, r):
    #     # Compute the viscosity force contribution
    #     v_xy = (self.ps.v[p_i] - self.ps.v[p_j]).dot(r)
    #     res = 2 * (self.ps.dim + 2) * self.viscosity * (self.mass / (self.ps.density[p_j])) * v_xy / (
    #             r.norm() ** 2 + 0.01 * self.ps.support_radius ** 2) * self.cubic_kernel_derivative(
    #         r)
    #     return res

    def substep(self):
        pass

    @ti.func
    def simulate_collisions(self, p_i, vec, d):
        # Collision factor, assume roughly (1-c_f)*velocity loss after collision
        c_f = 0.5
        self.ps.x[p_i] += vec * d
        self.ps.v[p_i] -= (1.0 + c_f) * self.ps.v[p_i].dot(vec) * vec

    @ti.func
    def enforce_boundary_2D(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_fluid:
                pos = self.ps.x[p_i]
                if pos[0] < self.ps.padding:
                    self.simulate_collisions(
                        p_i, ti.Vector([1.0, 0.0]),
                        self.ps.padding - pos[0])
                if pos[0] > self.ps.bound[0] - self.ps.padding:
                    self.simulate_collisions(
                        p_i, ti.Vector([-1.0, 0.0]),
                        pos[0] - (self.ps.bound[0] - self.ps.padding))
                if pos[1] > self.ps.bound[1] - self.ps.padding:
                    self.simulate_collisions(
                        p_i, ti.Vector([0.0, -1.0]),
                        pos[1] - (self.ps.bound[1] - self.ps.padding))
                if pos[1] < self.ps.padding:
                    self.simulate_collisions(
                        p_i, ti.Vector([0.0, 1.0]),
                        self.ps.padding - pos[1])
    @ti.func
    def enforce_boundary_3D(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_fluid:
                pos = self.ps.x[p_i]
                if pos[0] < self.ps.padding:
                    self.simulate_collisions(
                        p_i, ti.Vector([1.0, 0.0, 0.0]),
                        self.ps.padding - pos[0])
                if pos[0] > self.ps.domain_end[0] - self.ps.padding:
                    self.simulate_collisions(
                        p_i, ti.Vector([-1.0, 0.0, 0.0]),
                        pos[0] - (self.ps.domain_end[0] - self.ps.padding))
                if pos[1] > self.ps.domain_end[1] - self.ps.padding:
                    self.simulate_collisions(
                        p_i, ti.Vector([0.0, -1.0, 0.0]),
                        pos[1] - (self.ps.domain_end[1] - self.ps.padding))
                if pos[1] < self.ps.padding:
                    self.simulate_collisions(
                        p_i, ti.Vector([0.0, 1.0, 0.0]),
                        self.ps.padding - pos[1])
                if pos[2] > self.ps.domain_end[2] - self.ps.padding:
                    self.simulate_collisions(
                        p_i, ti.Vector([0.0, 0.0, -1.0]),
                        pos[1] - (self.ps.domain_end[2] - self.ps.padding))
                if pos[2] < self.ps.padding:
                    self.simulate_collisions(
                        p_i, ti.Vector([0.0, 0.0, 1.0]),
                        self.ps.padding - pos[2])
                
    @ti.func
    def simulate_collisions_v1(self, p_i, vec):
        # Collision factor, assume roughly (1-c_f)*velocity loss after collision
        c_f = 0.5
        self.ps.v[p_i] -= (
            1.0 + c_f) * self.ps.v[p_i].dot(vec) * vec
        
    @ti.func
    def enforce_boundary_3D_v1(self, particle_type:int):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] == particle_type:
                pos = self.ps.x[p_i]
                collision_normal = ti.Vector([0.0, 0.0, 0.0])
                if pos[0] > self.ps.domain_size[0] - self.ps.padding:
                    collision_normal[0] += 1.0
                    self.ps.x[p_i][0] = self.ps.domain_size[0] - self.ps.padding
                if pos[0] <= self.ps.padding:
                    collision_normal[0] += -1.0
                    self.ps.x[p_i][0] = self.ps.padding

                if pos[1] > self.ps.domain_size[1] - self.ps.padding:
                    collision_normal[1] += 1.0
                    self.ps.x[p_i][1] = self.ps.domain_size[1] - self.ps.padding
                if pos[1] <= self.ps.padding:
                    collision_normal[1] += -1.0
                    self.ps.x[p_i][1] = self.ps.padding

                if pos[2] > self.ps.domain_size[2] - self.ps.padding:
                    collision_normal[2] += 1.0
                    self.ps.x[p_i][2] = self.ps.domain_size[2] - self.ps.padding
                if pos[2] <= self.ps.padding:
                    collision_normal[2] += -1.0
                    self.ps.x[p_i][2] = self.ps.padding

                collision_normal_length = collision_normal.norm()
                if collision_normal_length > 1e-6:
                    self.simulate_collisions_v1(
                            p_i, collision_normal / collision_normal_length)
            assert self.ps.x[p_i][1] < self.ps.domain_size[1] and self.ps.x[p_i][2] < self.ps.domain_size[2] and self.ps.x[p_i][0] < self.ps.domain_size[0]
    @ti.func
    def compute_boundary_volume_task(self, p_i, p_j, delta_bi):
        if self.ps.material[p_j] == self.ps.material_boundary:
            delta_bi += self.cubic_kernel((self.ps.x[p_i] - self.ps.x[p_j]).norm())

    @ti.kernel
    def compute_volume_of_boundary_particle(self):
        for i in range(self.ps.particle_num[None]):
            if self.ps.material[i] == self.ps.material_boundary:
                delta_bi = self.cubic_kernel(0.0)
                self.ps.for_all_neighbors(i, self.compute_boundary_volume_task, delta_bi)
                self.ps.volume[i] = 1.0/ delta_bi

    
    @ti.kernel
    def enforce_boundary(self):
        # if self.ps.dim == 2:
        #     self.enforce_boundary_2D()
        self.enforce_boundary_3D_v1(self.ps.material_fluid)

    def step(self):
        self.ps.update()
        self.compute_volume_of_boundary_particle()
        self.substep()
        self.enforce_boundary()
