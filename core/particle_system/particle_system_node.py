import taichi as ti
from paritcle_system_base import ParticleSystemBase

@ti.data_oriented
class ParticleSystemNode(ParticleSystemBase):
    def __init__(self, simulation_config) -> None:
        super.__init__(simulation_config)
        self.particle_max_num_node = 0 
        self.start_index = 0
        self.end_index = 0
        