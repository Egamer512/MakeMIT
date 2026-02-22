#!/usr/bin/env python3
"""
BMO Desk Companion - OLED Face Animator
128x64 SSD1309 / SSD1306 display via luma.oled + Pillow

Faces: idle, happy, sad, listening, thinking, speaking
"""

import time
import math
import argparse
import signal
import sys
from PIL import Image, ImageDraw

# ── luma.oled display setup ────────────────────────────────────────────────────
def create_device(interface='i2c', spi_port=0, spi_device=0, i2c_port=1, i2c_address=0x3C, driver='ssd1309'):
    """Create and return the OLED device."""
    from luma.core.interface.serial import i2c, spi
    from luma.oled.device import ssd1306, ssd1309
    from luma.oled.device import sh1106

    drivers = {
        'ssd1306': ssd1306,
        'ssd1309': ssd1309,
        'sh1106':  sh1106,
    }
    DevClass = drivers.get(driver, ssd1309)

    if interface == 'spi':
        serial = spi(port=spi_port, device=spi_device)
    else:
        serial = i2c(port=i2c_port, address=i2c_address)

    return DevClass(serial, width=128, height=64)


# ── Drawing helpers ────────────────────────────────────────────────────────────
W, H = 128, 64
ON  = 'white'
DIM = '#606060'    # dimmer shade (shows as dimmer pixels on real OLED)
OFF = 'black'

def new_canvas():
    """Return a fresh (image, draw) pair."""
    img = Image.new('1', (W, H), 0)   # mode '1' = 1-bit for luma
    draw = ImageDraw.Draw(img)
    return img, draw

def fill_ellipse(draw, cx, cy, rx, ry, fill=ON):
    draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=fill)

def draw_ellipse(draw, cx, cy, rx, ry, outline=ON, width=1):
    for i in range(width):
        draw.ellipse([cx-rx+i, cy-ry+i, cx+rx-i, cy+ry-i], outline=outline)

def arc_xy(cx, cy, r, a0, a1, col=ON, draw=None, thick=1):
    """Draw an arc by plotting points."""
    step = max(0.5, 1/r)
    for deg in [a0 + d*step for d in range(int((a1-a0)/step)+1)]:
        rad = math.radians(deg)
        for t in range(thick):
            x = round(cx + (r-t) * math.cos(rad))
            y = round(cy + (r-t) * math.sin(rad))
            draw.point((x, y), fill=col)

def smile(draw, cx, cy, w=13, col=ON):
    """Draw a smile (arc opening upward = lower half of circle below cy)."""
    arc_xy(cx, cy, w, 15, 165, col=col, draw=draw, thick=2)

def frown(draw, cx, cy, w=13, col=ON):
    """Draw a frown (arc opening downward = upper half above cy)."""
    arc_xy(cx, cy, w, 195, 345, col=col, draw=draw, thick=2)

def eye(draw, cx, cy, rx=10, ry=8, closed=False, squint=0):
    """Draw one BMO-style eye."""
    if closed:
        # Closed: horizontal line with tiny curve
        draw.line([(cx-rx+2, cy), (cx+rx-2, cy)], fill=ON, width=2)
        draw.point((cx-rx+2, cy+1), fill=ON)
        draw.point((cx+rx-2, cy+1), fill=ON)
        return
    ry2 = max(1, ry - squint)
    fill_ellipse(draw, cx, cy, rx, ry2)
    # Black pupil
    draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill=OFF)
    # Eye shine
    draw.point((cx-2, cy-2), fill=ON)

def brow_flat(draw, cx, cy, col=ON):
    draw.line([(cx-7, cy), (cx+7, cy)], fill=col, width=2)

def brow_sad(draw, cx, cy, side='left', col=ON):
    """Furrowed sad brow - inner end goes down."""
    if side == 'left':
        draw.line([(cx-7, cy), (cx+7, cy+4)], fill=col, width=2)
    else:
        draw.line([(cx-7, cy+4), (cx+7, cy)], fill=col, width=2)

def brow_up(draw, cx, cy, col=ON):
    """Raised brow."""
    draw.line([(cx-7, cy-2), (cx+7, cy-2)], fill=col, width=2)

def brow_angry(draw, cx, cy, side='left', col=ON):
    """Furrowed angry brow - inner end goes up."""
    if side == 'left':
        draw.line([(cx-7, cy+3), (cx+7, cy)], fill=col, width=2)
    else:
        draw.line([(cx-7, cy), (cx+7, cy+3)], fill=col, width=2)


# ── Face positions ─────────────────────────────────────────────────────────────
EL  = (38, 27)   # left eye center
ER  = (90, 27)   # right eye center
EW  = 10         # eye x-radius
EH  = 8          # eye y-radius
BLY = EL[1]-13   # brow y (left)
BRY = ER[1]-13   # brow y (right)
MCX = 64         # mouth center x
MCY = 47         # mouth center y


# ── FACE STATES ────────────────────────────────────────────────────────────────

class FaceState:
    def render(self, phase: int) -> Image.Image:
        raise NotImplementedError


class IdleFace(FaceState):
    def render(self, phase):
        img, draw = new_canvas()
        # Blink every ~4 seconds (at 15fps → 60 frames)
        cycle = phase % 80
        closed = (cycle < 3)

        eye(draw, *EL, EW, EH, closed=closed)
        eye(draw, *ER, EW, EH, closed=closed)
        brow_flat(draw, EL[0], BLY)
        brow_flat(draw, ER[0], BRY)

        # Gentle neutral mouth
        smile(draw, MCX, MCY, w=10)

        # Blush marks
        draw.ellipse([EL[0]-14, EL[1]+4, EL[0]-6, EL[1]+10], outline=DIM)
        draw.ellipse([ER[0]+6,  ER[1]+4, ER[0]+14, ER[1]+10], outline=DIM)
        return img


class HappyFace(FaceState):
    def render(self, phase):
        img, draw = new_canvas()
        # Slight vertical bounce
        bounce = round(math.sin(phase * 0.18) * 1)

        eye(draw, EL[0], EL[1]+bounce, EW, EH+1)
        eye(draw, ER[0], ER[1]+bounce, EW, EH+1)
        brow_up(draw, EL[0], BLY+bounce)
        brow_up(draw, ER[0], BRY+bounce)

        # Big smile
        smile(draw, MCX, MCY+bounce, w=14)
        # Teeth line
        draw.line([(MCX-11, MCY+bounce-2), (MCX+11, MCY+bounce-2)], fill=ON, width=1)

        # Rosy cheeks
        for cx, cy in [(EL[0]-4, EL[1]+8+bounce), (ER[0]+4, ER[1]+8+bounce)]:
            draw.ellipse([cx-7, cy-3, cx+7, cy+3], fill=DIM)

        # Sparkle dots (animated)
        sp = (phase // 6) % 5
        for i, (sx, sy) in enumerate([(10,4),(113,4),(6,11),(117,11),(7,52)]):
            if i == sp:
                draw.rectangle([sx, sy, sx+2, sy+2], fill=ON)
        return img


class SadFace(FaceState):
    def render(self, phase):
        img, draw = new_canvas()

        eye(draw, EL[0], EL[1]+2, EW-1, EH-2)
        eye(draw, ER[0], ER[1]+2, EW-1, EH-2)
        brow_sad(draw, EL[0], BLY+2, side='left')
        brow_sad(draw, ER[0], BRY+2, side='right')

        # Frown
        frown(draw, MCX, MCY+8, w=12)

        # Animated tear drop
        tear_progress = (phase % 55)
        tear_start = ER[1] + EH + 2
        tear_end   = min(tear_start + tear_progress, H - 5)
        draw.line([(ER[0]+3, tear_start), (ER[0]+3, tear_end)], fill=ON, width=1)
        if tear_end > tear_start + 4:
            draw.ellipse([ER[0], tear_end-2, ER[0]+6, tear_end+3], fill=ON)
        return img


class ListeningFace(FaceState):
    def render(self, phase):
        img, draw = new_canvas()
        pulse = round(math.sin(phase * 0.22) * 1)

        eye(draw, EL[0], EL[1]+pulse, EW+1, EH+1)
        eye(draw, ER[0], ER[1]+pulse, EW+1, EH+1)
        brow_flat(draw, EL[0], BLY+pulse)
        brow_up(draw,   ER[0], BRY+pulse-1)   # one brow raised (curious)

        # Slightly open O mouth
        draw_ellipse(draw, MCX, MCY, 6, 4, width=2)

        # Audio bars (left side)
        bar_heights = [3, 5, 7, 5, 4]
        for i, bh in enumerate(bar_heights):
            animated_h = bh + round(math.sin(phase * 0.3 + i * 1.1) * 2)
            x = 4 + i * 5
            draw.rectangle([x, 56-animated_h, x+3, 56], fill=ON)
        return img


class ThinkingFace(FaceState):
    def render(self, phase):
        img, draw = new_canvas()

        # Squinting eyes (looking up-right)
        eye(draw, EL[0], EL[1], EW-2, EH-3)
        eye(draw, ER[0], ER[1], EW-2, EH-3)
        brow_angry(draw, EL[0], BLY, side='left')
        brow_angry(draw, ER[0], BRY, side='right')

        # Pursed mouth
        draw.line([(MCX-9, MCY), (MCX+9, MCY)], fill=ON, width=2)
        draw.point((MCX-9, MCY+1), fill=ON)
        draw.point((MCX+9, MCY+1), fill=ON)

        # Thought bubble (top-right)
        dot_phase = (phase // 18) % 4
        for i, (bx, by, br) in enumerate([(100, 12, 4),(107, 7, 3),(113, 3, 2)]):
            if dot_phase > i:
                draw.ellipse([bx-br, by-br, bx+br, by+br], fill=ON)
            else:
                draw.ellipse([bx-br, by-br, bx+br, by+br], outline=ON)

        # Sweat drop (top-left)
        draw.line([(14, 10), (14, 17)], fill=ON, width=1)
        draw.ellipse([11, 17, 17, 23], fill=ON)
        return img


class SpeakingFace(FaceState):
    def render(self, phase):
        img, draw = new_canvas()
        m_phase = (phase // 7) % 4
        m_open  = [0, 4, 7, 4][m_phase]

        eye(draw, *EL, EW, EH)
        eye(draw, *ER, EW, EH)
        brow_flat(draw, EL[0], BLY)
        brow_flat(draw, ER[0], BRY)

        if m_open == 0:
            # Closed mouth
            draw.line([(MCX-10, MCY), (MCX+10, MCY)], fill=ON, width=2)
        else:
            # Open mouth (oval)
            draw.ellipse([MCX-10, MCY-3, MCX+10, MCY+m_open], fill=ON)
            # Teeth
            draw.line([(MCX-8, MCY-1), (MCX+8, MCY-1)], fill=OFF, width=2)
            # Tongue if wide open
            if m_open >= 6:
                draw.ellipse([MCX-4, MCY+1, MCX+4, MCY+m_open-1], fill=DIM)

        # Sound wave arcs (right side)
        for i, r in enumerate([6, 10, 14]):
            alpha = round(math.sin(phase * 0.3 + i) * 1)
            arc_xy(120, 32, r+alpha, 240, 300, col=ON if i % 2 == 0 else DIM, draw=draw, thick=1)
        return img


# ── State machine ──────────────────────────────────────────────────────────────

FACES = {
    'idle':      IdleFace(),
    'happy':     HappyFace(),
    'sad':       SadFace(),
    'listening': ListeningFace(),
    'thinking':  ThinkingFace(),
    'speaking':  SpeakingFace(),
}

AUTO_SEQUENCE = ['idle', 'happy', 'listening', 'thinking', 'speaking', 'sad']


def transition(device, from_face, to_face, frames=12):
    """Crossfade-style wipe transition between two faces."""
    for i in range(frames + 1):
        t = i / frames
        # Vertical wipe: reveal to_face column by column
        split = round(t * W)
        img_from = from_face.render(0)
        img_to   = to_face.render(0)
        combined = Image.new('1', (W, H), 0)
        combined.paste(img_from.crop((split, 0, W, H)), (split, 0))
        combined.paste(img_to.crop((0, 0, split, H)),   (0,    0))
        device.display(combined)
        time.sleep(0.04)


def run(device, state='idle', auto=False, fps=15):
    """Main animation loop."""
    frame_time = 1.0 / fps
    phase = 0
    auto_idx = AUTO_SEQUENCE.index(state) if state in AUTO_SEQUENCE else 0
    auto_hold = 0
    AUTO_HOLD_FRAMES = fps * 4   # 4 seconds per state in auto mode

    current_face = FACES[state]
    prev_face    = current_face

    running = True
    def _sig(s, f): nonlocal running; running = False
    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    print(f"  Running '{state}' face — Ctrl+C to quit")

    while running:
        t0 = time.monotonic()

        # Auto mode cycling
        if auto:
            auto_hold += 1
            if auto_hold >= AUTO_HOLD_FRAMES:
                auto_hold = 0
                prev_face = current_face
                auto_idx  = (auto_idx + 1) % len(AUTO_SEQUENCE)
                state     = AUTO_SEQUENCE[auto_idx]
                current_face = FACES[state]
                print(f"  → {state}")
                transition(device, prev_face, current_face)
                phase = 0

        img = current_face.render(phase)
        device.display(img)
        phase += 1

        elapsed = time.monotonic() - t0
        sleep   = frame_time - elapsed
        if sleep > 0:
            time.sleep(sleep)

    device.cleanup()
    print("Bye!")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='BMO OLED Desk Companion')
    parser.add_argument('--interface', choices=['i2c','spi'],  default='i2c',    help='Bus interface (default: i2c)')
    parser.add_argument('--driver',    choices=['ssd1309','ssd1306','sh1106'], default='ssd1309', help='Display driver chip (default: ssd1309)')
    parser.add_argument('--i2c-port',    type=int, default=1,    help='I2C bus number (default: 1)')
    parser.add_argument('--i2c-address', type=lambda x: int(x,0), default=0x3C, help='I2C address (default: 0x3C)')
    parser.add_argument('--spi-port',   type=int, default=0,    help='SPI port (default: 0)')
    parser.add_argument('--spi-device', type=int, default=0,    help='SPI device (default: 0)')
    parser.add_argument('--state',  choices=list(FACES.keys()), default='idle',  help='Face to display (default: idle)')
    parser.add_argument('--auto',   action='store_true',                         help='Cycle through all faces automatically')
    parser.add_argument('--fps',    type=int, default=15,                        help='Frames per second (default: 15)')
    args = parser.parse_args()

    print("BMO Desk Companion")
    print(f"  Interface : {args.interface.upper()}")
    print(f"  Driver    : {args.driver}")

    try:
        device = create_device(
            interface   = args.interface,
            spi_port    = args.spi_port,
            spi_device  = args.spi_device,
            i2c_port    = args.i2c_port,
            i2c_address = args.i2c_address,
            driver      = args.driver,
        )
    except Exception as e:
        print(f"\nERROR: Could not open display — {e}")
        print("Check wiring and that I2C/SPI is enabled (raspi-config).")
        sys.exit(1)

    run(device, state=args.state, auto=args.auto, fps=args.fps)


if __name__ == '__main__':
    main()
