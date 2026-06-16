#!/usr/bin/env python3

"""
SpaceMouse navigation trainer.

A very simple OpenGL/Pygame trainer:
  - Move a controlled cube through a cubic workspace.
  - Avoid red obstacle boxes.
  - Reach the yellow target box.
  - Uses a Z-up semantic/world convention:
      +X = red   = right
      +Y = green = depth / view direction
      +Z = blue  = up

Notes:
  - This is intentionally kinematic, not physics-based.
  - The camera is front-on: the green face of the cube is initially
    parallel to the viewing plane. Movement along semantic Y is movement
    into/out of the screen.
"""

import time
import argparse
import random
import os
import glob
import select
import threading
import numpy as np

import pyspacemouse

import pygame
from pygame.locals import (
    DOUBLEBUF,
    OPENGL,
    QUIT,
    KEYDOWN,
    K_ESCAPE,
    K_r,
    K_n,
    K_EQUALS,
    K_MINUS,
)

from OpenGL.GL import *
from OpenGL.GLU import *


# -----------------------------
# Small utilities
# -----------------------------

def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def deadzone(x, dz):
    return 0.0 if abs(x) < dz else x


def distance(a, b):
    return (
        (a[0] - b[0]) ** 2
        + (a[1] - b[1]) ** 2
        + (a[2] - b[2]) ** 2
    ) ** 0.5


class HidrawButtonReader:
    """
    Low-level reader for SpaceMouse Compact button reports.

    From observed reports:
      03 00 00 -> no button
      03 01 00 -> button 0 pressed
      03 02 00 -> button 1 pressed
      03 03 00 -> both buttons pressed

    Therefore:
      report[0] == 0x03
      report[1] is the button bitmask
        bit 0 = button 0
        bit 1 = button 1
    """

    def __init__(self, vendor_id="256f", product_id="c635"):
        self.vendor_id = vendor_id.lower()
        self.product_id = product_id.lower()
        self.path = self._find_hidraw()
        self.fd = None
        self.mask = 0
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def _find_hidraw(self):
        for path in sorted(glob.glob("/dev/hidraw*")):
            name = os.path.basename(path)
            uevent_path = f"/sys/class/hidraw/{name}/device/uevent"

            try:
                text = open(uevent_path, "r").read().lower()
            except OSError:
                continue

            if self.vendor_id in text and self.product_id in text:
                return path

        return None

    def start(self):
        if self.path is None:
            print("warning: no SpaceMouse hidraw button device found")
            return False

        self.fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

        print(f"button reader: using {self.path}")
        return True

    def stop(self):
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=0.5)

        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def _loop(self):
        while self.running:
            try:
                readable, _, _ = select.select([self.fd], [], [], 0.05)
            except (OSError, ValueError):
                break

            if not readable:
                continue

            try:
                data = os.read(self.fd, 64)
            except BlockingIOError:
                continue
            except OSError:
                break

            if len(data) >= 2 and data[0] == 0x03:
                with self.lock:
                    self.mask = data[1]

    def get_mask(self):
        with self.lock:
            return self.mask

    def button0(self):
        return bool(self.get_mask() & 0x01)

    def button1(self):
        return bool(self.get_mask() & 0x02)


def get_cube_button_color(button_0_pressed, button_1_pressed):
    """
    Return a temporary cube color override while buttons are held.

    None means: use the normal multi-colored cube.
    """
    if button_0_pressed and button_1_pressed:
        return (1.0, 1.0, 1.0)      # both buttons: white
    if button_0_pressed:
        return (1.0, 0.55, 0.05)    # button 0: orange
    if button_1_pressed:
        return (0.55, 0.35, 1.0)    # button 1: purple
    return None


# -----------------------------
# SpaceMouse mapping
# -----------------------------

def sm_to_world_translation(x_raw, y_raw, z_raw):
    """
    Map SpaceMouse raw translation to semantic/world Z-up coordinates.

    Semantic/world convention:
      +X = red   = right
      +Y = green = depth / view direction
      +Z = blue  = up

    Current mapping:
      raw x -> world X
      raw y -> world Y
      raw z -> world Z

    If one axis is inverted, use the CLI flags:
      --invert_x
      --invert_y
      --invert_z
    """
    world_x = x_raw
    world_y = y_raw
    world_z = z_raw
    return world_x, world_y, world_z


def sm_to_world_rotation(roll_raw, pitch_raw, yaw_raw):
    """
    Map SpaceMouse raw rotations to semantic/world rotations.

    Semantic/world:
      roll  = rotation around world X / red
      pitch = rotation around world Y / green
      yaw   = rotation around world Z / blue

    For your current device observation:
      raw pitch -> semantic roll
      raw roll  -> semantic pitch
      raw yaw   -> semantic yaw
    """
    roll = pitch_raw
    pitch = roll_raw
    yaw = yaw_raw
    return roll, pitch, yaw


def world_to_gl_point(p):
    """
    Convert semantic/world Z-up coordinates to OpenGL display coordinates.

    World:
      X right
      Y depth
      Z up

    OpenGL:
      GL X = world X
      GL Y = world Z
      GL Z = world Y
    """
    x, y, z = p
    return x, z, y


def rot_x(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s,  c],
    ], dtype=float)


def rot_y(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [ c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=float)


def rot_z(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def reorthonormalize(R):
    """
    Keep the accumulated rotation matrix numerically well-behaved.
    """
    u, _, vh = np.linalg.svd(R)
    return u @ vh


def world_rotation_to_gl_matrix4(R_world_from_cube):
    """
    Convert a semantic/world rotation matrix to an OpenGL 4x4 matrix.

    Semantic/world axes:
      X = right
      Y = depth
      Z = up

    OpenGL axes:
      X = world X
      Y = world Z
      Z = world Y

    This is equivalent to:
      R_gl = P * R_world * P^T
    where P maps world coordinates to OpenGL coordinates.
    """
    P = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ], dtype=float)

    R_gl = P @ R_world_from_cube @ P.T

    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R_gl.astype(np.float32)

    # OpenGL expects column-major memory order.
    return M.T


# -----------------------------
# Collision helpers
# -----------------------------

def clamp_to_room(pos, room_min, room_max, margin):
    return [
        max(room_min[0] + margin, min(room_max[0] - margin, pos[0])),
        max(room_min[1] + margin, min(room_max[1] - margin, pos[1])),
        max(room_min[2] + margin, min(room_max[2] - margin, pos[2])),
    ]


def aabb_overlap(center_a, half_a, center_b, half_b):
    return (
        abs(center_a[0] - center_b[0]) <= half_a[0] + half_b[0]
        and abs(center_a[1] - center_b[1]) <= half_a[1] + half_b[1]
        and abs(center_a[2] - center_b[2]) <= half_a[2] + half_b[2]
    )


def collides_with_obstacles(pos, cube_half, obstacles):
    cube_half_vec = (cube_half, cube_half, cube_half)

    for obs in obstacles:
        obs_center = obs["center"]
        obs_size = obs["size"]
        obs_half = (
            obs_size[0] / 2.0,
            obs_size[1] / 2.0,
            obs_size[2] / 2.0,
        )

        if aabb_overlap(pos, cube_half_vec, obs_center, obs_half):
            return True

    return False


def is_position_free(pos, cube_half, room_min, room_max, obstacles):
    clamped = clamp_to_room(pos, room_min, room_max, cube_half)
    if any(abs(clamped[i] - pos[i]) > 1e-6 for i in range(3)):
        return False
    return not collides_with_obstacles(pos, cube_half, obstacles)


def random_free_target(room_min, room_max, cube_half, obstacles, min_dist_from=None):
    for _ in range(1000):
        p = [
            random.uniform(room_min[0] + 0.8, room_max[0] - 0.8),
            random.uniform(room_min[1] + 0.8, room_max[1] - 0.8),
            random.uniform(room_min[2] + 0.8, room_max[2] - 0.8),
        ]

        if not is_position_free(p, cube_half, room_min, room_max, obstacles):
            continue

        if min_dist_from is not None and distance(p, min_dist_from) < 2.0:
            continue

        return p

    # Fallback should basically never happen with the simple map below.
    return [2.3, 2.3, 2.3]


# -----------------------------
# Drawing
# -----------------------------

def draw_cube(color_override=None):
    """
    Draw the controlled cube.

    Important for the front-on view:
      the green face is the local +GL-Z face.
      Since GL-Z corresponds to semantic/world Y, the green face is initially
      parallel to the viewing plane.
    """
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
        (0, 1, 2, 3),  # -GL Z
        (4, 5, 6, 7),  # +GL Z = semantic/world +Y, green
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (1, 2, 6, 5),
        (0, 3, 7, 4),
    ]

    default_colors = [
        (0.8, 0.2, 0.2),  # red
        (0.2, 0.8, 0.2),  # green
        (0.2, 0.2, 0.8),  # blue
        (0.8, 0.8, 0.2),  # yellow
        (0.8, 0.2, 0.8),  # magenta
        (0.2, 0.8, 0.8),  # cyan
    ]

    if color_override is None:
        colors = default_colors
    else:
        colors = [color_override] * len(faces)

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


def draw_world_box(center, size, color=(1.0, 1.0, 1.0), wire=True):
    """
    Draw an axis-aligned box in semantic/world coordinates.
    center = (x, y, z)
    size   = (sx, sy, sz)
    """
    cx, cy, cz = center
    sx, sy, sz = size

    x0, x1 = cx - sx / 2.0, cx + sx / 2.0
    y0, y1 = cy - sy / 2.0, cy + sy / 2.0
    z0, z1 = cz - sz / 2.0, cz + sz / 2.0

    vertices_world = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]

    vertices_gl = [world_to_gl_point(p) for p in vertices_world]

    faces = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (1, 2, 6, 5),
        (0, 3, 7, 4),
    ]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    glColor3f(*color)

    if wire:
        glLineWidth(1.5)
        glBegin(GL_LINES)
        for a, b in edges:
            glVertex3f(*vertices_gl[a])
            glVertex3f(*vertices_gl[b])
        glEnd()
    else:
        glBegin(GL_QUADS)
        for face in faces:
            for idx in face:
                glVertex3f(*vertices_gl[idx])
        glEnd()

        glColor3f(0.0, 0.0, 0.0)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        for a, b in edges:
            glVertex3f(*vertices_gl[a])
            glVertex3f(*vertices_gl[b])
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


def setup_projection(width, height, projection_mode, fov_deg, ortho_scale):
    """
    Configure the camera projection.

    projection_mode:
      perspective  = normal 3D perspective projection
      orthographic = no perspective distortion, useful for X/Y/Z views
    """
    aspect = width / float(height)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    if projection_mode == "perspective":
        gluPerspective(fov_deg, aspect, 0.1, 100.0)
    elif projection_mode == "orthographic":
        if aspect >= 1.0:
            glOrtho(
                -ortho_scale * aspect,
                +ortho_scale * aspect,
                -ortho_scale,
                +ortho_scale,
                -100.0,
                +100.0,
            )
        else:
            glOrtho(
                -ortho_scale,
                +ortho_scale,
                -ortho_scale / aspect,
                +ortho_scale / aspect,
                -100.0,
                +100.0,
            )
    else:
        raise ValueError(f"unknown projection mode: {projection_mode}")

    glMatrixMode(GL_MODELVIEW)


def setup_camera_view(view_mode, camera_distance):
    """
    Configure view direction.

    Reminder: all actual geometry has already been converted to OpenGL coords:
      world X -> GL X
      world Y -> GL Z
      world Z -> GL Y

    View modes:
      y   : look along semantic/world Y. Screen = X-Z plane.
            This is the old front-on view. Green/depth points into screen.
      x   : look along semantic/world X. Screen = Y-Z plane.
      z   : look along semantic/world Z. Screen = X-Y plane. Top-down view.
      iso : oblique 3D view.
    """
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    if view_mode == "y":
        # Look along -GL Z = semantic/world Y depth.
        # Screen horizontal: world X / red
        # Screen vertical:   world Z / blue
        glTranslatef(0.0, 0.0, -camera_distance)

    elif view_mode == "x":
        # Look along semantic/world X.
        # Rotate scene so world X/GL X becomes view depth.
        # Screen approximately:
        #   horizontal = world Y / green
        #   vertical   = world Z / blue
        glTranslatef(0.0, 0.0, -camera_distance)
        glRotatef(-90.0, 0.0, 1.0, 0.0)

    elif view_mode == "z":
        # Look along semantic/world Z from above.
        # Semantic/world Z is GL Y, so rotate scene into view depth.
        # Screen approximately:
        #   horizontal = world X / red
        #   vertical   = world Y / green
        glTranslatef(0.0, 0.0, -camera_distance)
        glRotatef(90.0, 1.0, 0.0, 0.0)

    elif view_mode == "iso":
        # Oblique 3D view.
        glTranslatef(0.0, 0.0, -camera_distance)
        glRotatef(25.0, 1.0, 0.0, 0.0)
        glRotatef(-35.0, 0.0, 1.0, 0.0)

    else:
        raise ValueError(f"unknown camera view: {view_mode}")


# -----------------------------
# Main
# -----------------------------

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

    # Optional sign flips, without changing axis mapping.
    parser.add_argument("--invert_x", action="store_true")
    parser.add_argument("--invert_y", action="store_true")
    parser.add_argument("--invert_z", action="store_true")
    parser.add_argument("--invert_roll", action="store_true")
    parser.add_argument("--invert_pitch", action="store_true")
    parser.add_argument("--invert_yaw", action="store_true")

    # Rotation-axis blocking.
    #
    # These zero individual rotation command components after mapping/inversion,
    # before integration into the cube orientation matrix.
    parser.add_argument(
        "--block_roll",
        action="store_true",
        help="Block roll command, i.e. rotation around local/world X depending on frame mode.",
    )
    parser.add_argument(
        "--block_pitch",
        action="store_true",
        help="Block pitch command, i.e. rotation around local/world Y depending on frame mode.",
    )
    parser.add_argument(
        "--block_yaw",
        action="store_true",
        help="Block yaw command, i.e. rotation around local/world Z depending on frame mode.",
    )
    parser.add_argument(
        "--translation_only",
        action="store_true",
        help="Convenience flag: block all rotation axes.",
    )
    parser.add_argument(
        "--yaw_only",
        action="store_true",
        help="Convenience flag: block roll and pitch, keep only yaw rotation.",
    )

    # Environment options.
    parser.add_argument("--room_half_extent", type=float, default=3.0)
    parser.add_argument("--goal_radius", type=float, default=0.65)
    parser.add_argument("--no_obstacles", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--hidraw_vendor_id", type=str, default="256f")
    parser.add_argument("--hidraw_product_id", type=str, default="c635")
    parser.add_argument(
        "--world_frame_translation",
        action="store_true",
        help="Apply translation in world/camera frame instead of local cube frame.",
    )
    parser.add_argument(
        "--world_frame_rotation",
        action="store_true",
        help="Apply rotation increments around world axes instead of local cube axes.",
    )

    # Camera / projection options.
    parser.add_argument(
        "--view",
        choices=["x", "y", "z", "iso"],
        default="y",
        help=(
            "Camera view direction. "
            "x/y/z are orthogonal axis views; "
            "iso is an oblique 3D view. "
            "Default y keeps the green face parallel to the screen."
        ),
    )
    parser.add_argument(
        "--projection",
        choices=["perspective", "orthographic"],
        default="perspective",
        help="Use perspective projection or orthographic projection.",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=45.0,
        help="Perspective field of view in degrees.",
    )
    parser.add_argument(
        "--ortho_scale",
        type=float,
        default=4.2,
        help="Orthographic half-height scale.",
    )
    parser.add_argument(
        "--camera_distance",
        type=float,
        default=8.0,
        help="Camera distance from the scene.",
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    device = pyspacemouse.open()
    if not device:
        raise RuntimeError("Failed to open SpaceMouse device.")

    button_reader = HidrawButtonReader(
        vendor_id=args.hidraw_vendor_id,
        product_id=args.hidraw_product_id,
    )
    button_reader.start()

    pygame.init()
    width, height = 1000, 700
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("SpaceMouse navigation trainer")

    glEnable(GL_DEPTH_TEST)
    glClearColor(0.08, 0.08, 0.1, 1.0)

    setup_projection(
        width=width,
        height=height,
        projection_mode=args.projection,
        fov_deg=args.fov,
        ortho_scale=args.ortho_scale,
    )

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Semantic/world pose: X right, Y depth, Z up.
    start_pos = [-2.35, -2.35, -2.35]
    pos = start_pos.copy()

    # Cube orientation as a semantic/world rotation matrix.
    #
    # R_world_from_cube maps vectors expressed in the cube's local frame
    # into the semantic/world frame.
    #
    # This is the important difference from the older Euler-angle version:
    # incremental rotations can now be post-multiplied, so "yaw" means
    # "rotate around the cube's current local Z axis" rather than around
    # global/world Z.
    R_world_from_cube = np.eye(3, dtype=float)

    # Only used for the terminal status printout.
    rot_display = [0.0, 0.0, 0.0]  # integrated local roll, pitch, yaw commands in degrees

    # Simple cubic training environment.
    h = args.room_half_extent
    room_min = [-h, -h, -h]
    room_max = [ h,  h,  h]

    cube_half = 0.5
    goal_radius = args.goal_radius

    if args.no_obstacles:
        obstacles = []
    else:
        obstacles = [
            # A vertical-ish pillar. Offset from the reset position.
            {"center": [ 0.0,  0.0,  0.0], "size": [0.7, 0.7, 2.7]},

            # Front/back depth blockers.
            {"center": [-1.55,  1.1,  1.0], "size": [0.8, 1.5, 0.8]},
            {"center": [ 1.55, -1.1, -1.0], "size": [0.8, 1.5, 0.8]},

            # A horizontal obstacle near the top right.
            {"center": [ 1.1,  1.3,  1.9], "size": [1.5, 0.8, 0.45]},
        ]

    target_pos = random_free_target(
        room_min=room_min,
        room_max=room_max,
        cube_half=cube_half,
        obstacles=obstacles,
        min_dist_from=pos,
    )

    score = 0
    collisions = 0

    translation_sensitivity = args.translation_sensitivity
    rotation_sensitivity = args.rotation_sensitivity

    last_t = time.time()
    last_print_t = 0.0

    print()
    print("Controls:")
    print("  SpaceMouse : translate / rotate cube")
    print("  r          : reset cube pose")
    print("  n          : new random target")
    print("  SpaceMouse buttons:")
    print("               button 0 -> cube orange while held")
    print("               button 1 -> cube purple while held")
    print("               both     -> cube white while held")
    print("  + / -      : increase / decrease sensitivity")
    print("  ESC        : quit")
    print()
    print("World convention:")
    print("  +X red   = right")
    print("  +Y green = depth / view direction")
    print("  +Z blue  = up")
    print()
    print("Task:")
    print("  Move the colored cube to the yellow target.")
    print("  Avoid the red obstacle boxes.")
    print()
    print("Frame convention:")
    print("  Default: translation and rotation increments are applied in the")
    print("           cube's current LOCAL/body frame.")
    print("  Use --world_frame_translation and/or --world_frame_rotation to")
    print("  recover the older global/world-frame behavior.")
    print()
    print("Rotation-axis blocking:")
    print("  --block_roll      block rotation around X / red")
    print("  --block_pitch     block rotation around Y / green")
    print("  --block_yaw       block rotation around Z / blue")
    print("  --yaw_only        keep only yaw, block roll + pitch")
    print("  --translation_only block all rotations")
    print()
    print("Camera:")
    print("  --view y          orthogonal/front view along world Y; screen = X-Z")
    print("  --view x          orthogonal side view along world X; screen = Y-Z")
    print("  --view z          orthogonal top view along world Z; screen = X-Y")
    print("  --view iso        oblique 3D view")
    print("  --projection perspective|orthographic")
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
                    pos = start_pos.copy()
                    R_world_from_cube = np.eye(3, dtype=float)
                    rot_display = [0.0, 0.0, 0.0]
                    print("\nreset pose")
                elif event.key == K_n:
                    target_pos = random_free_target(
                        room_min=room_min,
                        room_max=room_max,
                        cube_half=cube_half,
                        obstacles=obstacles,
                        min_dist_from=pos,
                    )
                    print(f"\nnew target: {target_pos}")
                elif event.key == K_EQUALS:
                    translation_sensitivity *= 1.2
                    rotation_sensitivity *= 1.2
                    print(
                        f"\nsensitivity: "
                        f"trans={translation_sensitivity:.3f}, "
                        f"rot={rotation_sensitivity:.1f}"
                    )
                elif event.key == K_MINUS:
                    translation_sensitivity /= 1.2
                    rotation_sensitivity /= 1.2
                    print(
                        f"\nsensitivity: "
                        f"trans={translation_sensitivity:.3f}, "
                        f"rot={rotation_sensitivity:.1f}"
                    )

        state = pyspacemouse.read()

        display_values = [0.0] * 6

        # Button state comes from raw HID reports, because pyspacemouse
        # does not expose button presses for this device/driver combination.
        button_mask = button_reader.get_mask()
        button_0_pressed = bool(button_mask & 0x01)
        button_1_pressed = bool(button_mask & 0x02)
        cube_button_color = get_cube_button_color(button_0_pressed, button_1_pressed)

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

            # Optional rotation-axis blocking.
            #
            # Important:
            #   With the default local-frame integration, these axes refer to
            #   the cube's CURRENT LOCAL axes:
            #     roll  -> local X / red
            #     pitch -> local Y / green
            #     yaw   -> local Z / blue
            #
            #   With --world_frame_rotation, they refer to world axes instead.
            if args.translation_only:
                roll = 0.0
                pitch = 0.0
                yaw = 0.0

            if args.yaw_only:
                roll = 0.0
                pitch = 0.0

            if args.block_roll:
                roll = 0.0
            if args.block_pitch:
                pitch = 0.0
            if args.block_yaw:
                yaw = 0.0

            display_values = [x, y, z, roll, pitch, yaw]

            # Translation update.
            #
            # Default behavior:
            #   translation increments are expressed in the cube's LOCAL frame.
            #   So after the cube is rotated, pushing "local Z" moves along the
            #   cube's current local Z direction.
            #
            # Optional:
            #   --world_frame_translation keeps translation in world/camera axes.
            local_delta = np.array([
                x * translation_sensitivity * dt,
                y * translation_sensitivity * dt,
                z * translation_sensitivity * dt,
            ], dtype=float)

            if args.world_frame_translation:
                world_delta = local_delta
            else:
                world_delta = R_world_from_cube @ local_delta

            candidate_pos = [
                pos[0] + float(world_delta[0]),
                pos[1] + float(world_delta[1]),
                pos[2] + float(world_delta[2]),
            ]

            candidate_pos = clamp_to_room(candidate_pos, room_min, room_max, cube_half)

            if not collides_with_obstacles(candidate_pos, cube_half, obstacles):
                pos = candidate_pos
            else:
                collisions += 1

            # Rotation update.
            #
            # Default behavior:
            #   rotation increments are applied in the cube's LOCAL frame:
            #     R_new = R_old @ dR_local
            #
            # This means:
            #   yaw   -> current local cube Z / blue axis
            #   pitch -> current local cube Y / green axis
            #   roll  -> current local cube X / red axis
            #
            # Optional:
            #   --world_frame_rotation applies increments around world axes:
            #     R_new = dR_world @ R_old
            d_roll = np.deg2rad(roll * rotation_sensitivity * dt)
            d_pitch = np.deg2rad(pitch * rotation_sensitivity * dt)
            d_yaw = np.deg2rad(yaw * rotation_sensitivity * dt)

            dR = rot_z(d_yaw) @ rot_y(d_pitch) @ rot_x(d_roll)

            if args.world_frame_rotation:
                R_world_from_cube = dR @ R_world_from_cube
            else:
                R_world_from_cube = R_world_from_cube @ dR

            R_world_from_cube = reorthonormalize(R_world_from_cube)

            # Terminal-only approximate command integration.
            rot_display[0] += roll * rotation_sensitivity * dt
            rot_display[1] += pitch * rotation_sensitivity * dt
            rot_display[2] += yaw * rotation_sensitivity * dt

            # Target reached.
            if distance(pos, target_pos) < goal_radius:
                score += 1
                print(f"\nTARGET REACHED! score={score}, collisions={collisions}")
                target_pos = random_free_target(
                    room_min=room_min,
                    room_max=room_max,
                    cube_half=cube_half,
                    obstacles=obstacles,
                    min_dist_from=start_pos,
                )
                pos = start_pos.copy()
                R_world_from_cube = np.eye(3, dtype=float)
                rot_display = [0.0, 0.0, 0.0]

            if now - last_print_t > 0.15:
                last_print_t = now
                print(
                    f"\rscore={score:03d} collisions={collisions:04d} | "
                    f"world "
                    f"x={x:+.3f} y={y:+.3f} z={z:+.3f} "
                    f"roll={roll:+.3f} pitch={pitch:+.3f} yaw={yaw:+.3f} "
                    f"| pos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}) "
                    f"cmd_rot=({rot_display[0]:+.1f},{rot_display[1]:+.1f},{rot_display[2]:+.1f}) "
                    f"target=({target_pos[0]:+.2f},{target_pos[1]:+.2f},{target_pos[2]:+.2f}) "
                    f"buttons=({int(button_0_pressed)},{int(button_1_pressed)})",
                    end="",
                    flush=True,
                )

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        setup_projection(
            width=width,
            height=height,
            projection_mode=args.projection,
            fov_deg=args.fov,
            ortho_scale=args.ortho_scale,
        )

        setup_camera_view(
            view_mode=args.view,
            camera_distance=args.camera_distance,
        )

        # World axes
        draw_axes(length=2.0)

        # Room
        room_center = [
            (room_min[0] + room_max[0]) / 2.0,
            (room_min[1] + room_max[1]) / 2.0,
            (room_min[2] + room_max[2]) / 2.0,
        ]
        room_size = [
            room_max[0] - room_min[0],
            room_max[1] - room_min[1],
            room_max[2] - room_min[2],
        ]
        draw_world_box(room_center, room_size, color=(0.6, 0.6, 0.6), wire=True)

        # Obstacles
        for obs in obstacles:
            draw_world_box(obs["center"], obs["size"], color=(0.9, 0.15, 0.15), wire=False)

        # Target
        draw_world_box(target_pos, (0.45, 0.45, 0.45), color=(1.0, 0.9, 0.1), wire=False)

        # Controlled cube
        glPushMatrix()

        gl_x, gl_y, gl_z = world_to_gl_point(pos)
        glTranslatef(gl_x, gl_y, gl_z)

        # Orientation from accumulated rotation matrix.
        #
        # With default settings, increments were integrated in the cube's
        # local/body frame. Thus, after arbitrary reorientation, yaw is around
        # the cube's current local Z axis, not global/world Z.
        glMultMatrixf(world_rotation_to_gl_matrix4(R_world_from_cube))

        draw_cube(color_override=cube_button_color)
        glPopMatrix()

        draw_axis_bars(display_values, width, height)

        pygame.display.flip()
        pygame.time.wait(10)

    button_reader.stop()
    pygame.quit()
    print("\nclosed.")


if __name__ == "__main__":
    main()
