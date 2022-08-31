import math
# Define positions of the 5 microphone array
# (this is really the 3D printed 6-array, but with one unpopulated)
# Origin is the center of the ring, as you might expect
# y axis points towards the ethernet, z axis points down.
def p2c(theta, radius):
    x = math.sin(theta) * radius
    y = math.cos(theta) * radius
    z = 0
    return (x, y, z)

# meters
radius = 0.1175 / 2.0
array_x5 = [
    p2c(60 * math.pi / 180., radius),
    p2c(0 * math.pi / 180., radius),
    p2c(-60 * math.pi / 180., radius),
    p2c(120 * math.pi / 180., radius),
    p2c(180 * math.pi / 180., radius),
]

array_x6 = [
    p2c(60 * math.pi / 180., radius),
    p2c(0 * math.pi / 180., radius),
    p2c(-60 * math.pi / 180., radius),
    p2c(-120 * math.pi / 180., radius),
    p2c(120 * math.pi / 180., radius),
    p2c(180 * math.pi / 180., radius),
]
