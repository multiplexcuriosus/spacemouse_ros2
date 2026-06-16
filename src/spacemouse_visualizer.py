#!/usr/bin/env python3

import time
import argparse

import pyspacemouse

import pygame
from pygame.locals import (
    DOUBLEBUF,
    OPENGL,
    QUIT,
    KEYDOWN,
    K_ESCAPE,
    K_r,
    K_EQUALS,
    K_MINUS,
)

from OpenGL.GL import *
from OpenGL.GLU import *


def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def deadzone(x, dz):
    return 0.0 if abs(x) < dz else x


def sm_to_world_translation(x_raw, y_raw, z_raw):
    """
    Map SpaceMouse raw translation to semantic/world Z-up coordinates.

    Semantic/world convention:
      +X = red   = right
      +Y = green = forward/back depth axis
      +Z = blue  = up

    For your current observation:
      raw x -> world X
      raw y -> world Y
      raw z -> world Z

    If one axis is inverted, change the sign here.
    """
    world_x = x_raw
    world_y = y_raw
    world_z = z_raw

    return world_x, world_y, world_z


def sm_to_world_rotation(roll_raw, pitch_raw, yaw_raw):
    """
    Map SpaceMouse raw rotations to semantic/world rotations.

    rot[0] = roll  around world X / red
    rot[1] = pitch around world Y / green
    rot[2] = yaw   around world Z / blue

    You said rotation around X is already correct, so this keeps roll unchanged.
    """
    roll = pitch_raw
    pitch = roll_raw
    yaw = yaw_raw

    return roll, pitch, yaw


def world_to_gl_position(x, y, z):
    """
    Convert semantic/world Z-up coordinates to OpenGL display coordinates.

    World:
      X right
      Y depth
      Z up

    OpenGL display:
      GL X = world X
      GL Y = world Z
      GL Z = world Y
    """
    return x, z, y


def draw_cube():
    vertices = [
        (-0.5, -0.5, -0.5),
        ( 0.5, -0.5, -0.5),
        ( 0.5,  0.5, -0.5),
        (-0.5,  0.5, -0.5),
        (-0.5, -0.5,  0.5),
        ( 0.5, -0.5,  0.5),
        ( 0.5,  0.5,  0.5),
        (-0.5,  0.5,  0.5),
    ]

    faces = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (1, 2, 6, 5),
        (0, 3, 7, 4),
    ]

    colors = [
        (0.8, 0.2, 0.2),
        (0.2, 0.8, 0.2),
        (0.2, 0.2, 0.8),
        (0.8, 0.8, 0.2),
        (0.8, 0.2, 0.8),
        (0.2, 0.8, 0.8),
    ]

    glBegin(GL_QUADS)
    for face, color in zip(faces, colors):
        glColor3f(*color)
        for idx in face:
            glVertex3f(*vertices[idx])
    glEnd()

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    glColor3f(0.0, 0.0, 0.0)
    glLineWidth(2.0)
    glBegin(GL_LINES)
    for a, b in edges:
        glVertex3f(*vertices[a])
        glVertex3f(*vertices[b])
    glEnd()


def draw_axes(length=2.0):
    """
    Draw semantic/world axes with Z-up:

      X = red   = right
      Y = green = depth
      Z = blue  = up
    """
    glLineWidth(3.0)
    glBegin(GL_LINES)

    # World X -> GL X
    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(0, 0, 0)
    glVertex3f(length, 0, 0)

    # World Y -> GL Z
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, length)

    # World Z -> GL Y
    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, length, 0)

    glEnd()


def draw_axis_bars(values, width, height):
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, width, 0, height, -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glDisable(GL_DEPTH_TEST)

    x0 = 30
    y0 = height - 40
    bar_w = 180
    bar_h = 12
    gap = 24

    for i, value in enumerate(values):
        y = y0 - i * gap
        center = x0 + bar_w / 2.0
        v = clamp(value)

        # Background
        glColor3f(0.25, 0.25, 0.25)
        glBegin(GL_QUADS)
        glVertex2f(x0, y)
        glVertex2f(x0 + bar_w, y)
        glVertex2f(x0 + bar_w, y + bar_h)
        glVertex2f(x0, y + bar_h)
        glEnd()

        # Center line
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_LINES)
        glVertex2f(center, y - 2)
        glVertex2f(center, y + bar_h + 2)
        glEnd()

        # Value bar
        if v >= 0:
            x_start = center
            x_end = center + v * bar_w / 2.0
        else:
            x_start = center + v * bar_w / 2.0
            x_end = center

        glColor3f(0.2, 0.7, 1.0)
        glBegin(GL_QUADS)
        glVertex2f(x_start, y)
        glVertex2f(x_end, y)
        glVertex2f(x_end, y + bar_h)
        glVertex2f(x_start, y + bar_h)
        glEnd()

    glEnable(GL_DEPTH_TEST)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deadzone", type=float, default=0.04)
    parser.add_argument(
        "--translation_sensitivity",
        type=float,
        default=1.0,
        help="meters-ish per second at full SpaceMouse deflection",
    )
    parser.add_argument(
        "--rotation_sensitivity",
        type=float,
        default=120.0,
        help="degrees per second at full SpaceMouse deflection",
    )

    # Optional sign flips, without changing the axis mapping.
    parser.add_argument("--invert_x", action="store_true")
    parser.add_argument("--invert_y", action="store_true")
    parser.add_argument("--invert_z", action="store_true")
    parser.add_argument("--invert_roll", action="store_true")
    parser.add_argument("--invert_pitch", action="store_true")
    parser.add_argument("--invert_yaw", action="store_true")

    args = parser.parse_args()

    device = pyspacemouse.open()
    if not device:
        raise RuntimeError("Failed to open SpaceMouse device.")

    pygame.init()
    width, height = 1000, 700
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("SpaceMouse sensitivity visualizer")

    glEnable(GL_DEPTH_TEST)
    glClearColor(0.08, 0.08, 0.1, 1.0)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, width / height, 0.1, 100.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Semantic/world pose: X right, Y depth, Z up.
    pos = [0.0, 0.0, 0.0]
    rot = [0.0, 0.0, 0.0]  # roll, pitch, yaw in degrees

    translation_sensitivity = args.translation_sensitivity
    rotation_sensitivity = args.rotation_sensitivity

    last_t = time.time()
    last_print_t = 0.0

    print()
    print("Controls:")
    print("  SpaceMouse: translate / rotate cube")
    print("  r         : reset cube pose")
    print("  + / -     : increase / decrease sensitivity")
    print("  ESC       : quit")
    print()
    print("World convention:")
    print("  +X red   = right")
    print("  +Y green = depth")
    print("  +Z blue  = up")
    print()

    running = True
    while running:
        now = time.time()
        dt = now - last_t
        last_t = now

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_r:
                    pos = [0.0, 0.0, 0.0]
                    rot = [0.0, 0.0, 0.0]
                elif event.key == K_EQUALS:
                    translation_sensitivity *= 1.2
                    rotation_sensitivity *= 1.2
                    print(
                        f"sensitivity: "
                        f"trans={translation_sensitivity:.3f}, "
                        f"rot={rotation_sensitivity:.1f}"
                    )
                elif event.key == K_MINUS:
                    translation_sensitivity /= 1.2
                    rotation_sensitivity /= 1.2
                    print(
                        f"sensitivity: "
                        f"trans={translation_sensitivity:.3f}, "
                        f"rot={rotation_sensitivity:.1f}"
                    )

        state = pyspacemouse.read()

        display_values = [0.0] * 6

        if state:
            x_raw = deadzone(float(state.x), args.deadzone)
            y_raw = deadzone(float(state.y), args.deadzone)
            z_raw = deadzone(float(state.z), args.deadzone)

            roll_raw = deadzone(float(state.roll), args.deadzone)
            pitch_raw = deadzone(float(state.pitch), args.deadzone)
            yaw_raw = deadzone(float(state.yaw), args.deadzone)

            x, y, z = sm_to_world_translation(x_raw, y_raw, z_raw)
            roll, pitch, yaw = sm_to_world_rotation(roll_raw, pitch_raw, yaw_raw)

            if args.invert_x:
                x *= -1.0
            if args.invert_y:
                y *= -1.0
            if args.invert_z:
                z *= -1.0

            if args.invert_roll:
                roll *= -1.0
            if args.invert_pitch:
                pitch *= -1.0
            if args.invert_yaw:
                yaw *= -1.0

            display_values = [x, y, z, roll, pitch, yaw]

            # Treat SpaceMouse values as velocity commands.
            pos[0] += x * translation_sensitivity * dt
            pos[1] += y * translation_sensitivity * dt
            pos[2] += z * translation_sensitivity * dt

            rot[0] += roll * rotation_sensitivity * dt
            rot[1] += pitch * rotation_sensitivity * dt
            rot[2] += yaw * rotation_sensitivity * dt

            if now - last_print_t > 0.15:
                last_print_t = now
                print(
                    f"\rworld "
                    f"x={x:+.3f} y={y:+.3f} z={z:+.3f} "
                    f"roll={roll:+.3f} pitch={pitch:+.3f} yaw={yaw:+.3f} "
                    f"| raw "
                    f"x={x_raw:+.3f} y={y_raw:+.3f} z={z_raw:+.3f} "
                    f"| pos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}) "
                    f"rot=({rot[0]:+.1f},{rot[1]:+.1f},{rot[2]:+.1f})",
                    end="",
                    flush=True,
                )

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Camera
        glTranslatef(0.0, 0.0, -8.0)
        #glRotatef(25.0, 1.0, 0.0, 0.0)
        #glRotatef(-35.0, 0.0, 1.0, 0.0)

        # World axes
        draw_axes(length=2.0)

        # Object transform
        glPushMatrix()

        # Position: semantic/world Z-up -> OpenGL display coordinates.
        gl_x, gl_y, gl_z = world_to_gl_position(pos[0], pos[1], pos[2])
        glTranslatef(gl_x, gl_y, gl_z)

        # Rotation:
        # roll  around world X -> GL X
        # pitch around world Y -> GL Z
        # yaw   around world Z -> GL Y
        glRotatef(rot[2], 0.0, 1.0, 0.0)  # yaw around blue/up world Z
        glRotatef(rot[1], 0.0, 0.0, 1.0)  # pitch around green world Y
        glRotatef(rot[0], 1.0, 0.0, 0.0)  # roll around red world X

        draw_cube()
        glPopMatrix()

        draw_axis_bars(display_values, width, height)

        pygame.display.flip()
        pygame.time.wait(10)

    pygame.quit()
    print("\nclosed.")


if __name__ == "__main__":
    main()