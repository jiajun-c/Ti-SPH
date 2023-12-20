import taichi as ti
import numpy as np
from partice_system.partice_system import ParticleSystem
import const


@ti.data_oriented
class SPHBase:

    def __init__(self, particle_system: ParticleSystem):
        self.ps = particle_system  # 微分系统
        self.viscosity = 0.05  # 粘性系数
        self.density_0 = 1000.0  # 水 的密度
        self.dt = ti.field(float, shape=())
        self.dt[None] = 2e-4
        self.mass = self.ps.m_V * self.density_0  # 单位体积内的质量

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
        # support 2D only now
        res = ti.Vector([0.0, 0.0])
        p_rho_i = self.ps.pressure[p_i] / (self.ps.density[p_i] ** 2)
        if self.ps.material[p_j] == self.ps.material_fluid:
            res = -self.density_0 * self.ps.m_V * (self.ps.pressure[p_i] / self.ps.density[p_i] ** 2
                                               + self.ps.pressure[p_j] / self.ps.density[p_j] ** 2) \
              * self.cubic_kernel_derivative(r)
        elif self.ps.material[p_j] == self.ps.material_boundary:
            res = -self.density_0*self.ps.v[p_j] * p_rho_i * self.cubic_kernel_derivative(r)
        return res

    # 液体的粘性力
    @ti.func
    def viscosity_force(self, p_i, p_j, r):
        # Compute the viscosity force contribution
        v_xy = (self.ps.v[p_i] - self.ps.v[p_j]).dot(r)
        res = 2 * (self.ps.dim + 2) * self.viscosity * (self.mass / (self.ps.density[p_j])) * v_xy / (
                r.norm() ** 2 + 0.01 * self.ps.support_radius ** 2) * self.cubic_kernel_derivative(
            r)
        return res

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
        if self.ps.dim == 2:
            self.enforce_boundary_2D()

    def step(self):
        self.ps.init()
        self.compute_volume_of_boundary_particle()
        self.substep()
        self.enforce_boundary()
