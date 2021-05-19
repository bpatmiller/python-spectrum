# Project: python-spectrum
# Author: Brendan Miller
#
# Based off of Taichi "stable fluid" example code:
# https://github.com/taichi-dev/taichi/blob/master/examples/stable_fluid.py
# References:
# http://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html
# https://github.com/PavelDoGreat/WebGL-Fluid-Simulation
# https://www.bilibili.com/video/BV1ZK411H7Hc?p=4
# https://github.com/ShaneFX/GAMES201/tree/master/HW01

import sys
import pyaudio
import numpy as np
import taichi as ti
import struct
from audioop import rms
from scipy.fftpack import fft
from statistics import mean
from random import random
import wave
import colorsys

from audioReader import audioReader, audioReaderInput

use_mgpcg = False  # True to use multigrid-preconditioned conjugate gradients
res = 512
dt = 0.009
p_jacobi_iters = 40  # 40 for a quicker but less accurate result
f_strength = 10000.0
curl_strength = 12  # 7 for unrealistic visual enhancement
dye_decay = 0.99
force_radius = res / 3.0
debug = False
paused = False

ti.init(arch=ti.gpu)
_velocities = ti.Vector.field(2, float, shape=(res, res))
_intermedia_velocities = ti.Vector.field(2, float, shape=(res, res))
_new_velocities = ti.Vector.field(2, float, shape=(res, res))
velocity_divs = ti.field(float, shape=(res, res))
velocity_curls = ti.field(float, shape=(res, res))
_pressures = ti.field(float, shape=(res, res))
_new_pressures = ti.field(float, shape=(res, res))
_dye_buffer = ti.Vector.field(3, float, shape=(res, res))
_intermedia_dye_buffer = ti.Vector.field(3, float, shape=(res, res))
_new_dye_buffer = ti.Vector.field(3, float, shape=(res, res))
pixels = ti.Vector.field(3, float, shape=(res, res))


class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


velocities_pair = TexPair(_velocities, _new_velocities)
pressures_pair = TexPair(_pressures, _new_pressures)
dyes_pair = TexPair(_dye_buffer, _new_dye_buffer)


@ti.func
def sample(qf, u, v):
    I = ti.Vector([int(u), int(v)])
    I = max(0, min(res - 1, I))
    return qf[I]


@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)


@ti.func
def bilerp(vf, p):
    u, v = p
    s, t = u - 0.5, v - 0.5
    # floor
    iu, iv = ti.floor(s), ti.floor(t)
    # fract
    fu, fv = s - iu, t - iv
    a = sample(vf, iu, iv)
    b = sample(vf, iu + 1, iv)
    c = sample(vf, iu, iv + 1)
    d = sample(vf, iu + 1, iv + 1)
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)


@ti.func
def sample_minmax(vf, p):
    u, v = p
    s, t = u - 0.5, v - 0.5
    # floor
    iu, iv = ti.floor(s), ti.floor(t)
    a = sample(vf, iu, iv)
    b = sample(vf, iu + 1, iv)
    c = sample(vf, iu, iv + 1)
    d = sample(vf, iu + 1, iv + 1)
    return min(a, b, c, d), max(a, b, c, d)


@ti.func
def backtrace_rk1(vf: ti.template(), p, dt: ti.template()):
    p -= dt * bilerp(vf, p)
    return p


@ti.func
def backtrace_rk2(vf: ti.template(), p, dt: ti.template()):
    p_mid = p - 0.5 * dt * bilerp(vf, p)
    p -= dt * bilerp(vf, p_mid)
    return p


@ti.func
def backtrace_rk3(vf: ti.template(), p, dt: ti.template()):
    v1 = bilerp(vf, p)
    p1 = p - 0.5 * dt * v1
    v2 = bilerp(vf, p1)
    p2 = p - 0.75 * dt * v2
    v3 = bilerp(vf, p2)
    p -= dt * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)
    return p


backtrace = backtrace_rk3


@ti.kernel
def advect_semilag(vf: ti.template(), qf: ti.template(), new_qf: ti.template(),
                   intermedia_qf: ti.template()):
    for i, j in vf:
        p = ti.Vector([i, j]) + 0.5
        p = backtrace(vf, p, dt)
        new_qf[i, j] = bilerp(qf, p)


@ti.kernel
def advect_bfecc(vf: ti.template(), qf: ti.template(), new_qf: ti.template(),
                 intermedia_qf: ti.template()):
    for i, j in vf:
        p = ti.Vector([i, j]) + 0.5
        p = backtrace(vf, p, dt)
        intermedia_qf[i, j] = bilerp(qf, p)

    for i, j in vf:
        p = ti.Vector([i, j]) + 0.5
        # star means the temp value after a back tracing (forward advection)
        # two star means the temp value after a forward tracing (reverse
        # advection)
        p_two_star = backtrace(vf, p, -dt)
        p_star = backtrace(vf, p, dt)
        q_star = intermedia_qf[i, j]
        new_qf[i, j] = bilerp(intermedia_qf, p_two_star)

        new_qf[i, j] = q_star + 0.5 * (qf[i, j] - new_qf[i, j])

        min_val, max_val = sample_minmax(qf, p_star)
        cond = min_val < new_qf[i, j] < max_val
        for k in ti.static(range(cond.n)):
            if not cond[k]:
                new_qf[i, j][k] = q_star[k]


advect = advect_bfecc


@ti.kernel
def apply_impulse(vf: ti.template(), dyef: ti.template(),
                  bands: ti.ext_arr(), time: ti.f32, nbands: ti.i32):
    for i, j in vf:
        target_band = int((i / float(res)) * nbands)
        value = bands[target_band]
        target_top_pixel = int(value * res)

        # omx, omy = i , 0
        mdir = ti.Vector([(random() - 0.5), value / 8.0])
        dx, dy = random(), random()
        d2 = dx * dx + dy * dy
        # dv = F * dt
        factor = ti.exp(-d2 / force_radius) / 8.0
        momentum = mdir * f_strength * dt * factor
        v = vf[i, j]
        vf[i, j] = v + momentum
        # add dye
        dc = dyef[i, j]
        if value > 1.0:
            dc += ti.exp(-d2 * (4 / (res / 15)**2)) * ti.Vector(
                hsv2rgb(ti.cos(time), 0.2 + 0.02 * value, 0.1))
        dc *= dye_decay
        dyef[i, j] = dc


@ti.kernel
def divergence(vf: ti.template()):
    for i, j in vf:
        vl = sample(vf, i - 1, j).x
        vr = sample(vf, i + 1, j).x
        vb = sample(vf, i, j - 1).y
        vt = sample(vf, i, j + 1).y
        vc = sample(vf, i, j)
        if i == 0:
            vl = 0
        if i == res - 1:
            vr = 0
        if j == 0:
            vb = 0
        if j == res - 1:
            vt = 0
        velocity_divs[i, j] = (vr - vl + vt - vb) * 0.5


@ti.kernel
def vorticity(vf: ti.template()):
    for i, j in vf:
        vl = sample(vf, i - 1, j).y
        vr = sample(vf, i + 1, j).y
        vb = sample(vf, i, j - 1).x
        vt = sample(vf, i, j + 1).x
        vc = sample(vf, i, j)
        velocity_curls[i, j] = (vr - vl - vt + vb) * 0.5


@ti.kernel
def pressure_jacobi_single(pf: ti.template(), new_pf: ti.template()):
    for i, j in pf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        div = velocity_divs[i, j]
        new_pf[i, j] = (pl + pr + pb + pt - div) * 0.25


@ti.kernel
def pressure_jacobi_dual(pf: ti.template(), new_pf: ti.template()):
    for i, j in pf:
        pcc = sample(pf, i, j)
        pll = sample(pf, i - 2, j)
        prr = sample(pf, i + 2, j)
        pbb = sample(pf, i, j - 2)
        ptt = sample(pf, i, j + 2)
        plb = sample(pf, i - 1, j - 1)
        prb = sample(pf, i + 1, j - 1)
        plt = sample(pf, i - 1, j + 1)
        prt = sample(pf, i + 1, j + 1)
        div = sample(velocity_divs, i, j)
        divl = sample(velocity_divs, i - 1, j)
        divr = sample(velocity_divs, i + 1, j)
        divb = sample(velocity_divs, i, j - 1)
        divt = sample(velocity_divs, i, j + 1)
        new_pf[i,
               j] = (pll + prr + pbb + ptt - divl - divr - divb - divt - div +
                     (plt + prt + prb + plb) * 2 + pcc * 4) * 0.0625


pressure_jacobi = pressure_jacobi_single

if pressure_jacobi == pressure_jacobi_dual:
    p_jacobi_iters //= 2


@ti.kernel
def subtract_gradient(vf: ti.template(), pf: ti.template()):
    for i, j in vf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        vf[i, j] -= 0.5 * ti.Vector([pr - pl, pt - pb])


@ti.kernel
def enhance_vorticity(vf: ti.template(), cf: ti.template()):
    # anti-physics visual enhancement...
    for i, j in vf:
        cl = sample(cf, i - 1, j)
        cr = sample(cf, i + 1, j)
        cb = sample(cf, i, j - 1)
        ct = sample(cf, i, j + 1)
        cc = sample(cf, i, j)
        force = ti.Vector([abs(ct) - abs(cb),
                           abs(cl) - abs(cr)]).normalized(1e-3)
        force *= curl_strength * cc
        vf[i, j] = min(max(vf[i, j] + force * dt, -1e3), 1e3)


def step(imp_data, time):
    advect(velocities_pair.cur, velocities_pair.cur, velocities_pair.nxt,
           _intermedia_velocities)
    advect(velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt,
           _intermedia_dye_buffer)
    velocities_pair.swap()
    dyes_pair.swap()

    apply_impulse(
        velocities_pair.cur,
        dyes_pair.cur,
        imp_data,
        time,
        imp_data.size)

    divergence(velocities_pair.cur)

    if curl_strength:
        vorticity(velocities_pair.cur)
        enhance_vorticity(velocities_pair.cur, velocity_curls)

    if use_mgpcg:
        mgpcg.init(velocity_divs, -1)
        mgpcg.solve(max_iters=10)
        mgpcg.get_result(pressures_pair.cur)

    else:
        for _ in range(p_jacobi_iters):
            pressure_jacobi(pressures_pair.cur, pressures_pair.nxt)
            pressures_pair.swap()

    subtract_gradient(velocities_pair.cur, pressures_pair.cur)

    if debug:
        divergence(velocities_pair.cur)
        div_s = np.sum(velocity_divs.to_numpy())
        print(f'divergence={div_s}')

    draw_spectrum(time, sum(imp_data) / len(bands), imp_data, imp_data.size)


def reset():
    # velocities_pair.cur.fill(0)
    # pressures_pair.cur.fill(0)
    dyes_pair.cur.fill(1)


@ti.func
def hsv2rgb(h, s, v, time=0.0):
    r = 0.0
    b = 0.0
    g = 0.0
    h *= 0.4
    h = (h + time) % 1
    frac = (h * 3) - (int(h * 3))
    if (h <= 1.0 / 3):
        r = 1.0 - frac
        g = frac
    elif (h <= 2.0 / 3):
        g = 1.0 - frac
        b = frac
    else:
        b = 1.0 - frac
        r = frac

    g /= 1.5
    s = max(0.0, s - 0.1)
    avg = v * (r + g + b) / 3.0
    return (s * (r * v) + (1 - s) * avg,
            s * (g * v) + (1 - s) * avg,
            s * (b * v) + (1 - s) * avg)


@ti.kernel
def draw_spectrum(
        time: float,
        mag: float,
        bands: ti.ext_arr(),
        nbands: ti.i32):
    for i, j in dyes_pair.cur:
        target_band = int((i / float(res)) * nbands)
        value = bands[target_band]
        target_top_pixel = int(value * res / 2.0)
        if (j < target_top_pixel):
            rgb = hsv2rgb(
                (float(i) / float(res)),
                0.35 + 0.1 * ti.cos(time),
                0.85 + 0.1 * value, time / 20)
            dyes_pair.cur[i, j] = ti.Vector(rgb)

            pixels[i, j] = dyes_pair.cur[i, j] * 1.3
        elif (j == target_top_pixel):
            pixels[i, j].fill(0)
        else:
            pixels[i, j] = dyes_pair.cur[i, j]


@ti.kernel
def flash_pixels():
    for i, j in pixels:
        pixels[i, j][0] *= 1.25
        pixels[i, j][1] *= 1.25
        pixels[i, j][2] *= 1.25


@ti.kernel
def flip_velocity_field():
    for i, j in velocities_pair.cur:
        velocities_pair.cur[i, j] = - velocities_pair.cur[i, j]
        velocities_pair.nxt[i, j] = - velocities_pair.nxt[i, j]


gui = ti.GUI("python-spectrum", res=(res, res))

deviceName = sys.argv[1]
ar = audioReader(audioReaderInput.device, 24, deviceName)

time = 0.0
scale = 0.1
burstScale = 0.6
running_avg_rms = 1000.0
avg_frames_per_reset_target = 100
avg_frames_per_reset = avg_frames_per_reset_target
frame = 0
resets = 1

try:
    while gui.running:
        for e in gui.get_events():
            if e.key == 'q':
                gui.running = False
                print("exiting")
            elif e.key == ti.GUI.UP:
                scale = min(scale + 0.05, 10.0)
                print(scale)
            elif e.key == ti.GUI.DOWN:
                scale = max(scale - 0.05, 0.05)
                print(scale)
            elif e.key == ti.GUI.RIGHT:
                burstScale += 0.01
                print(burstScale)
            elif e.key == ti.GUI.LEFT:
                burstScale -= 0.01
                print(burstScale)

        frame += 1
        bands = ar.getFrameFFT()
        mag = sum(bands)
        willReset = mag * burstScale > running_avg_rms * 2.0
        avg_frames_per_reset = 1 / \
            (avg_frames_per_reset * .99 + (1.0 if willReset else 0.0))

        running_avg_rms = running_avg_rms * .96 + mag * .04
        if (willReset):
            print("avg_frames_per_reset: {} burstScale: {} reset {} :".format(
                avg_frames_per_reset, burstScale, resets), time)
            resets += 1
            np.random.shuffle(bands)
            if random() < 0.1:
                print("reset dye fields")
                reset()
            step(bands * scale * 2.0, time)
            flash_pixels()
            flip_velocity_field()
            gui.set_image(pixels)
            gui.show()
            continue

        time += dt
        step(bands * scale, time)

        gui.set_image(pixels)
        gui.show()

except KeyboardInterrupt:
    print("exiting")

ar.close()

