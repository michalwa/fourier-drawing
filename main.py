import os

from typing import List, Tuple, Iterator

from math import sqrt, sin, cos, atan2
from math import pi as PI

import pygame
import pygame.display as display
import pygame.event as events
import pygame.draw as draw
import pygame.gfxdraw as gfxdraw
import pygame.image as image
import pygame.transform as transform
from pygame.time import Clock

from git_logo import git_logo


# A simple 2-d vector or point implementation for utility purposes
# Allows +, -, with other vectors or number pairs and *, / operations with numbers
class Vec2d:
    def __init__(self, x=.0, y=.0):
        self.x = x
        self.y = y

    def __str__(self):
        return '(%d, %d)' % (self.x, self.y)

    # Adds another vector or a pair of numbers
    def __add__(self, other):
        if isinstance(other, Vec2d):
            return Vec2d(self.x + other.x, self.y + other.y)
        elif len(other) == 2 and isinstance(other[0], (int, float)) and isinstance(other[1], (int, float)):
            return Vec2d(self.x + other[0], self.y + other[1])
        else:
            raise ValueError('Unsupported type for Vec2d `+` operator')

    # Subtracts another vector or a pair of numbers from this vector
    def __sub__(self, other):
        if isinstance(other, Vec2d):
            return Vec2d(self.x - other.x, self.y - other.y)
        elif len(other) == 2 and isinstance(other[0], (int, float)) and isinstance(other[1], (int, float)):
            return Vec2d(self.x - other[0], self.y - other[1])
        else:
            raise ValueError('Unsupported type for Vec2d `-` operator')

    # Multiplies each component of this vector by the given scalar
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vec2d(self.x * other, self.y * other)
        else:
            raise ValueError('Unsupported type for Vec2d `*` operator')

    # Divides each component of this vector by the given scalar
    def __div__(self, other):
        if isinstance(other, (int, float)):
            return Vec2d(self.x / other, self.y / other)
        else:
            raise ValueError('Unsupported type for Vec2d `/` operator')

    @property
    def int_tuple(self) -> Tuple[int, int]:
        return (int(self.x), int(self.y))

    @property
    def float_tuple(self) -> Tuple[float, float]:
        return (float(self.x), float(self.y))


# Maps a number from one range to another
def mapf(value: float, a=(0, 1), b=(0, 1)) -> float:
    return b[0] + (value - a[0]) / (a[1] - a[0]) * (b[1] - b[0])


# Colors
BLACK       = (   0,   0,   0 )
DARKER_GREY = (  31,  31,  31 )
DARK_GREY   = (  63,  63,  63 )
GREY        = ( 127, 127, 127 )
LIGHT_GREY  = ( 190, 190, 190 )
WHITE       = ( 255, 255, 255 )
RED         = ( 255,   0,   0 )
GREEN       = (   0, 255,   0 )
BLUE        = (   0,   0, 255 )
NICE_RED    = ( 220,  40,  40 )
NICE_GREEN  = (  40, 220, 110 )
NICE_BLUE   = (   0, 140, 230 )

# Display dimensions
WIDTH  = 800
HEIGHT = 800
# Window title
TITLE  = 'Fourier Drawing'
# Whether to output each frame to a file
EXPORT = False
# Frames per second
framerate = 30
# Number of rendered frames
frame = 0

# Constants
MINDIM = min(WIDTH, HEIGHT)
MIDDLE = Vec2d(WIDTH / 2, HEIGHT / 2)


#region Sketch

# Stores a single result of the discrete fourier transform function
class Wave:

    # Calculates and stores wave parameters based on the given frequency index
    # and complex result of the discrete fourier transform of that index
    def __init__(self, k: int, X_k: complex):
        self.freq = k
        self.amp = sqrt(X_k.real ** 2 + X_k.imag ** 2)
        self.phase = atan2(X_k.imag, X_k.real)

    def __str__(self):
        return 'Wave(freq=%.5f, amp=%.5f, phase=%.5f)' % (self.freq, self.amp, self.phase)


# Discrete Fourier Transform
def dft(x: List[complex]) -> List[complex]:

    # X_k = sum_{n=0}^{N-1} x_n * (cos(2*pi*kn/N) - i * sin(2*pi*kn/N))
    # https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Definition

    N = len(x)
    X = []

    for k in range(N):
        X_k = 0
        for n, x_n in enumerate(x):
            phi = 2.0 * PI * k * n / N
            X_k += x_n * complex(cos(phi), -sin(phi))

        X_k /= N  # take the average
        X.append(X_k)

    return X


def sample(func, n, full=False) -> Iterator:
    for i in range(n + (1 if full else 0)):
        yield func(i / n)


# Angle function for a square path
def square_t(t: float) -> Tuple[float, float]:
    t = t % 1 * PI * 2
    a = abs(cos(t))
    b = abs(sin(t))
    r = min(1 / (a if a != 0 else 1), 1 / (b if b != 0 else 1))
    return (r * cos(t), r * sin(t))


# Angle function for a circle path
def circle_t(t: float) -> Tuple[float, float]:
    t = t % 1 * PI * 2
    return (cos(t), sin(t))


#region Parameters

# Whether to perform a (poor quality) 2x zoom
ZOOM = False
# The radius of a path-tracing circle for amplitude 1
SCALE = MINDIM / 2 - 100
# How many trail points to keep at most
trail_length = 1
# The original sampled path to re-draw at the beginning of the draw loop
original_path = []
# The result of the discrete Fourier transform computed in setup
waves = []
# Points of the path traced by the epicycles
trail = []
# Elapsed time
t = 0
# Time step
dt = 0

#endregion


# Calculates stuff and sets things up
def setup() -> bool:
    global trail_length, original_path, waves, dt

    #samples = list(sample(square_t, 200))
    samples = git_logo
    trail_length = len(samples) - 20

    original_path = [(MIDDLE + Vec2d(*p) * SCALE).float_tuple for p in samples]

    fourier = dft([complex(*p) for p in samples])
    waves = [Wave(k, X_k) for k, X_k in enumerate(fourier)]
    waves = sorted(waves, key=lambda w: w.amp, reverse=True)

    dt = PI * 2 / len(waves)


# Called each frame to render and update the sketch
def show(screen):
    global t

    screen.fill(BLACK)

    # Original path
    draw.aalines(screen, DARKER_GREY, True, original_path)

    # Epicycles
    origin = MIDDLE  # type: Vec2d
    for wave in waves:

        # Circle
        radius = mapf(wave.amp, b=(0, SCALE))
        #draw.circle(screen, DARK_GREY, origin.int_tuple, int(max(2, radius)), 1)
        gfxdraw.aacircle(screen, int(origin.x), int(origin.y), int(max(2, radius)), DARK_GREY)

        # Line
        angle = t * wave.freq + wave.phase
        end_point = origin + (radius * cos(angle), radius * sin(angle))
        draw.aaline(screen, WHITE, origin.int_tuple, end_point.int_tuple)

        origin = end_point

    # Trail
    trail.append(origin.float_tuple)
    if len(trail) > trail_length:
        trail.pop(0)
    if len(trail) > 1:
        draw.aalines(screen, NICE_RED, False, trail)

    # Zoom
    if ZOOM:
        scaled = transform.scale2x(screen)
        screen.fill(BLACK)
        screen.blit(scaled, (WIDTH / 2 - origin.x * 2, HEIGHT / 2 - origin.y * 2))

    t += dt

#endregion


if __name__ == '__main__':

    # Setup
    pygame.init()
    screen = display.set_mode((WIDTH, HEIGHT))
    display.set_caption(TITLE)
    clock = Clock()
    setup()

    if EXPORT:
        try:
            os.mkdir('export')
        except FileExistsError:
            pass

    # Loop
    running = True
    while running:

        # Poll events
        clock.tick(framerate)
        for event in events.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Render
        show(screen)
        if EXPORT:
            image.save(screen, 'export/f_%04d.png' % frame)
        display.flip()

        frame += 1

    # Quit
    pygame.quit()
