import taichi as ti
import trimesh as trim
import numpy as np
from functools import reduce

# 粒子系统的基类，存储其中的配置信息和共享方法
@ti.data_oriented
class ParticleSystemBase:
    def __init__(self, simulation_config) -> None:
        self.simulation_config = simulation_config
        self.configuation = self.simulation_config['configuration']
        self.rigidBodiesConfig = self.simulation_config['rigidBodies']
        self.fluidBodiesConfig = self.simulation_config['fluidBlocks']
        
        self.dim = self.configuation['dim']
        self.domin_start = np.array(self.configuation['domin_start'])
        self.domin_end = np.array(self.configuation['domin_end'])
        self.domin_size = self.domin_end - self.domin_start
        
        self.fluid = self.simulation_config['fluidBlocks']
        self.rigid = self.simulation_config['rigidBlocks']
        
        self.particle_radius = self.configuation['paricleRadius']
        self.particle_diameter = 2.0*self.paricle_radius
        self.support_radius = 4.0*self.paricle_radius
        
        self.grid_size = self.support_radius
        self.grid_num = np.ceil(self.domin_size/self.grid_size).astype(np.int32)
        # 一个切片的网格数目
        self.slice_size = self.grid_num[2]*self.grid_num[1]
        self.particle_num = 0
        
    @ti.func 
    def pos_to_index(self, pos):
        # 确保位置在空间内
        assert pos[0] < self.domin_size[0] and pos[1] < self.domin_size[1] and pos[2] < self.domin_size[2]
        return (pos / self.grid_size).cast(int)

    @ti.func
    def flatten_grid_index(self, grid_index):
        return grid_index[0]*self.grid_num[1]*self.grid_num[2] + grid_index[1]*self.grid_num[2] + grid_index[2]
    
    