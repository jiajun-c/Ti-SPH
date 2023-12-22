import taichi as ti
import json
from core.partice_system import partice_systemv2
ti.init(arch=ti.cuda)

if __name__ == "__main__":
    with open("./data/scenes/demo.json", "r") as f:
        simulation_config = json.load(f)
        
    config = simulation_config['configuration']
    ps = partice_systemv2.ParticleSystemV2((512, 512), config)
    