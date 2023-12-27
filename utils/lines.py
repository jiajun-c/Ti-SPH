import taichi as ti

def getlines(config):
    start = config['domainStart']
    end = config['domainEnd']
    points = ti.Vector.field(3, dtype=ti.f32, shape = 8)
    points[0] = ti.Vector([start[0], start[1], start[2]])
    points[1] = ti.Vector([start[0], end[1], start[2]])
    points[2] = ti.Vector([end[0], start[1], start[2]])
    points[3] = ti.Vector([end[0], end[1], start[2]])
    points[4] = ti.Vector([start[0], start[1], end[2]])
    points[5] = ti.Vector([start[0], end[1], end[2]])
    points[6] = ti.Vector([end[0], start[1], end[2]])
    points[7] = ti.Vector([end[0], end[1], end[2]])
    box_lines_indices = ti.field(int, shape=(2 * 12))
    for i, val in enumerate([0, 1, 0, 2, 1, 3, 2, 3, 4, 5, 4, 6, 5, 7, 6, 7, 0, 4, 1, 5, 2, 6, 3, 7]):
        box_lines_indices[i] = val
    return (points, box_lines_indices)