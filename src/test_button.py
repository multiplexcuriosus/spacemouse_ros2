#!/usr/bin/env python3

import os
import glob
import select
import time

VID = "256F"
PID = "C635"


def is_spacemouse_hidraw(path):
    name = os.path.basename(path)
    uevent_path = f"/sys/class/hidraw/{name}/device/uevent"

    try:
        text = open(uevent_path, "r").read().upper()
    except OSError:
        return False

    return VID in text and PID in text


paths = sorted(glob.glob("/dev/hidraw*"))
paths = [p for p in paths if is_spacemouse_hidraw(p)]

print("matching hidraw devices:")
for p in paths:
    print(" ", p)

if not paths:
    raise SystemExit("No matching SpaceMouse hidraw device found.")

fds = {}
for p in paths:
    fd = os.open(p, os.O_RDONLY | os.O_NONBLOCK)
    fds[fd] = p

print()
print("Listening for raw HID reports.")
print("Press/release the two buttons.")
print("Also try HOLDING a button while moving the cap.")
print("Ctrl+C to stop.")
print()

last_by_path = {}

while True:
    readable, _, _ = select.select(list(fds.keys()), [], [], 1.0)

    for fd in readable:
        path = fds[fd]

        try:
            data = os.read(fd, 64)
        except BlockingIOError:
            continue

        if not data:
            continue

        hex_bytes = " ".join(f"{b:02x}" for b in data)

        if last_by_path.get(path) != data:
            last_by_path[path] = data
            print(f"{time.time():.3f} {path}: {hex_bytes}")