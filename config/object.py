import numpy as np

class object_config:
    def __init__(self, f) -> None:
        self.object_id = f['objectId']
        self.scale = f['scale']
        self.translation = f['translation']
        self.color = f['color']
        self.particle_num = 0
        
    
class fluid_config(object_config):
    def __init__(self, f) -> None:
        super().__init__(f)
        self.start = f['start']
        self.end = f['end']
        self.density = f['density']

class rigid_config(object_config):
    def __init__(self, f) -> None:
        super().__init__(f)