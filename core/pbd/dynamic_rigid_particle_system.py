import taichi as ti
from core.particle_system.particle_base import ParticleBase

@ti.data_oriented
class ParticleSystem(ParticleBase):
    def __init__(self, simulation_config) -> None:
        
        # material type 
        self.material_static_rigid = 0
        self.material_dynamic_rigid = 1
        self.material_fluid = 2
        
        super.__init__(simulation_config)
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
            self.particle_max_num += particle_num
