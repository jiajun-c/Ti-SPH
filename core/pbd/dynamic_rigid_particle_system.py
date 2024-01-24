import taichi as ti
import numpy as np
from core.particle_system.particle_base import ParticleBase

@ti.data_oriented
class ParticleSystem(ParticleBase):
    def __init__(self, simulation_config) -> None:
        
        # material type 
        self.material_static_rigid = 0
        self.material_dynamic_rigid = 1
        self.material_fluid = 2
        self.rigid_num = 0 

        super.__init__(simulation_config)
        self.x_next = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.particle_max_num) # the particle next position
        self.phase = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.particle_max_num)
        self.rigid_sum = ti.field(dtype=ti.i32, shape=self.rigid_num)
        self.rigid_center = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.rigid_num)
        self.rigid_id = ti.field(dtype=ti.i32, shape=self.particle_max_num)
        self.rigidR = ti.Vector.field(self.dim,dtype=ti.f32, shape=self.rigid_num)
        self.rigidT = ti.Vector.field(self.dim,dtype=ti.f32, shape=self.rigid_num)
        pass
    
    def compute_particle_num(self):
        for fluid in self.fluidBlocksConfig:
            start = fluid['start']
            end = fluid['end']
            self.particle_max_num += self.compute_cube_particles_num(start, end)
            # print(self.particle_max_num)
        for rigid in self.rigidBodiesConfig:
            voxelized_points = self.load_rigid_body(rigid)
            particle_num = voxelized_points.shape[0]
            self.rigid_sum[self.rigid_num] = particle_num
            self.particle_max_num += particle_num
            self.rigid_num += 1
    # the mass should be set later
    def add_fluid_and_rigid(self):
        rigid_id = 0
        for rigid in self.rigidBodiesConfig:
            voxelized_points = self.load_rigid_body(rigid)
            particle_num = voxelized_points.shape[0]
            # print(self.particle_num[None])
            rigid['partice_num'] = particle_num
            rigid['voxelized_points'] = voxelized_points
            material = self.material_static_rigid
            """
            If a body is dynamic, it can be 
            """
            if rigid['dynamic']:
                material = self.material_dynamic_rigid
            material = np.full((particle_num, ), material, dtype=np.int32)
            velocity = rigid['velocity']
            velocity = np.tile(np.array(velocity, dtype=np.float32), (particle_num, 1))
            density = rigid['density']
            density = np.full_like(np.zeros(particle_num), density if density is not None else 1000.)
            pressure = np.full_like(np.zeros(particle_num), 0.)
            positions = voxelized_points
            # TODO: object id should not be rigid id
            self.add_particles(particle_num, 
                            positions,
                            velocity,
                            density,
                            pressure,
                            material,
                            self.rigid_num,
                            rigid_id)
            rigid_id += 1
    
    @ti.func
    def add_particle(self, p, x, v, density, pressure, material, id):
        # if p >= self.particle_max_num:
            # print("error:", self.particle_num[None])
        self.x[p] = x
        self.v[p] = v
        self.density[p] = density
        self.pressure[p] = pressure
        self.material[p] = material
        self.object_id[p] = id
        self.volume[p] = self.m_V0 # TODO: fluid volume need be compute later
        self.mass[p] = self.volume[p]*self.density[p]
