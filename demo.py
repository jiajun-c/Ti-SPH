import taichi as ti

ti.init()

# n: vector dimension; w: width; h: height
n, w, h = 3, 128, 64
vec_field = ti.Vector.field(n, dtype=float, shape=())

@ti.kernel
def fill_vector():
    vec_field[None][0] =10

fill_vector()
print(vec_field)