import taichi as ti
from core.pbd.dynamic_rigid_particle_system import ParticleSystem

class PBD:
    def __init__(self, ps:ParticleSystem):
        self.ps = ps
        self.dt = ti.field(float, shape=())
        self.dt[None] = 2e-4
            
    def substep(self):
        pass
    
    def getNextposition(self):
        for p_i in range(self.ps.particle_max_num):
            self.ps.v[p_i] += self.dt[None]*self.ps.d_velocity[p_i]
            self.ps.x_next[p_i] += self.dt[None]*self.ps.v[p_i]

    @ti.kernel
    def get_centre(self, pos:ti.template(), centre:ti.template()):
        for i in range(self.ps.particle_max_num):
            rigid_id = self.ps.rigid_id[i] 
            centre[rigid_id] += pos[i] * self.ps.mass[i] 
            self.ps.rigid_sum[rigid_id] += self.ps.mass[i] 

        for i in range(self.ps.rigid_num):
            centre[i] = centre[i] / self.ps.rigid_sum[i]

    @ti.kernel
    def update_velocity(self):
        for i in range(self.ps.particle_max_num):
            self.ps.v[i] = (self.ps.x_next[i] - self.ps.x[i])/self.dt[None]
            self.ps.x[i] = self.ps.x_next[i]
        pass
    
    def substep(self):
        self.getNextposition()
        self.get_centre(self.ps.x_next, self.ps.rigid_center)
        self.update_velocity()
        
    @ti.kernel
    def shape_match(self):
        for i in range(self.ps.rigid_num):
            self.ps.rigidR[i] = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            self.ps.rigidT[i] = ti.Vector([0.0,0.0,0.0])
        for i in range(self.ps.particle_max_num):
            phase = self.ps.rigid_id[i] 
            x     = self.ps.x_zero[i]-self.rigidCentreZero[phase]
            y     = self.ps.x_next[i]-self.ps.rigid_center[phase]

            #cal S
            self.rigidR[phase] += self.mass[i] * x.outer_product(y)
            
        pass