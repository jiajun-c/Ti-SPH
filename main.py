import taichi as ti
import json
from core.particle_system import partice_systemv2
from core.sph.wcsph import WCSPH
ti.init(arch=ti.gpu)

if __name__ == "__main__":
    with open("./data/scenes/demo.json", "r") as f:
        simulation_config = json.load(f)
        
    config = simulation_config['configuration']
    ps = partice_systemv2.ParticleSystemV2((512, 512), simulation_config)
    # add fluid and rigid
    ps.add_fluid_and_rigid()
    wcsph = WCSPH(ps)
    gui = ti.GUI(background_color=0xFFFFFF)
    while gui.running:
        for i in range(5):
            wcsph.step()
        particle_info = ps.dump()
        gui.circles(particle_info['position'] * ps.screen_to_world_ratio / 512,
                    radius=ps.particle_radius / 1.5 * ps.screen_to_world_ratio,
                    color=0x111113)
        gui.show()