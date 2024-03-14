import taichi as ti
import numpy as np
from core.particle_system.particle_system import ParticleSystem
from core.sph.wcsph import WCSPH

ti.init(arch=ti.gpu)

if __name__ == "__main__":
    ps = ParticleSystem((512, 512))
    ps.add_cube(lower_corner=[3, 1],
                cube_size=[3.0, 5.0],
                color=0x111111,
                velocity=[0, -20],
                density=1000.0,
                material=1)
    wcsph_solver = WCSPH(ps)
    gui = ti.GUI(background_color=0xFFFFFF)
    while gui.running:
        for i in range(5):
            wcsph_solver.step()
        particle_info = ps.dump()
        gui.circles(particle_info['position'] * ps.screen_to_world_ratio / 512,
                    radius=ps.particle_radius / 1.5 * ps.screen_to_world_ratio,
                    color=0x111113)
        gui.show()


