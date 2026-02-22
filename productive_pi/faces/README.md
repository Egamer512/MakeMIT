# BMO OLED Desk Companion

Animated BMO-style faces on a 128×64 OLED (SSD1309/SSD1306) connected to a Raspberry Pi.

---

## Wiring

### I2C (most common, 4 wires)

```
OLED Pin   →   Raspberry Pi
─────────────────────────────
VCC        →   Pin 1  (3.3V)
GND        →   Pin 6  (GND)
SCL / CLK  →   Pin 5  (GPIO 3 / SCL)
SDA / DIN  →   Pin 3  (GPIO 2 / SDA)
```

> **Note:** Some modules have a CS or DC pin — leave them floating or tied high for I2C mode.
> Check your module's back for I2C/SPI solder jumpers and make sure I2C is selected.

### SPI (faster, 6+ wires)

```
OLED Pin   →   Raspberry Pi
─────────────────────────────
VCC        →   Pin 1  (3.3V)
GND        →   Pin 6  (GND)
CLK        →   Pin 23 (GPIO 11 / SCLK)
DIN / MOSI →   Pin 19 (GPIO 10 / MOSI)
CS         →   Pin 24 (GPIO 8  / CE0)
DC         →   Pin 18 (GPIO 24) ← any free GPIO, note the number
RST        →   Pin 22 (GPIO 25) ← any free GPIO
```

---

## Setup

```bash
# 1. Copy files to your Pi (from your computer):
scp -r bmo_face/ pi@raspberrypi.local:~/

# 2. SSH in and run setup:
ssh pi@raspberrypi.local
cd ~/bmo_face
bash setup.sh
```

The setup script will:
- Enable I2C and SPI in raspi-config
- Install system and Python dependencies
- Scan the I2C bus to verify your display is detected

---

## Running

```bash
# Default: idle face over I2C
python3 face.py

# Specific face
python3 face.py --state happy
python3 face.py --state sad
python3 face.py --state listening
python3 face.py --state thinking
python3 face.py --state speaking

# Auto-cycle through all faces (4 sec each)
python3 face.py --auto

# SSD1306 display (common 0.96" modules)
python3 face.py --driver ssd1306

# SPI interface
python3 face.py --interface spi

# Custom I2C address (some modules use 0x3D)
python3 face.py --i2c-address 0x3D

# All options
python3 face.py --help
```

---

## Auto-start on Boot (optional)

To make the face start automatically when the Pi powers on:

```bash
# Edit the service file if your username isn't 'pi'
nano bmo-face.service   # change 'pi' to your username if needed

# Install and enable the service
sudo cp bmo-face.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable bmo-face
sudo systemctl start bmo-face

# Check it's running
sudo systemctl status bmo-face

# View logs
journalctl -u bmo-face -f
```

---

## Face States

| State      | Description                                      |
|------------|--------------------------------------------------|
| `idle`     | Neutral face, slow blink, blush marks            |
| `happy`    | Big smile, raised brows, blush, sparkles, bounce |
| `sad`      | Frown, sad brows, animated falling tear          |
| `listening`| Wide eyes, open mouth, animated audio bars       |
| `thinking` | Squint, pursed lips, thought bubble, sweat drop  |
| `speaking` | Talking mouth animation, sound wave arcs         |

---

## Troubleshooting

**Display not detected (I2C):**
```bash
# Check I2C is enabled
ls /dev/i2c*

# Scan for devices
sudo i2cdetect -y 1

# Should show 3c or 3d in the grid
```

**"No module named luma" error:**
```bash
pip3 install --break-system-packages luma.oled pillow
```

**Display shows garbage / wrong driver:**
- 2.42" OLED modules commonly use one of three chips. Try each until it works:
  ```bash
  python3 face.py --driver ssd1309   # most common for 2.42"
  python3 face.py --driver sh1106    # also very common for 2.42"
  python3 face.py --driver ssd1306   # common on smaller 0.96" modules
  ```
- The chip name is usually printed in tiny text on the driver IC on the back of the board

**Permission denied on SPI/I2C:**
```bash
sudo usermod -aG i2c,spi $(whoami)
# then log out and back in
```

---

## Integrating with Your Project

You can import and control faces from your own Python code:

```python
from face import create_device, FACES, run

device = create_device(interface='i2c')

# Render a single frame
img = FACES['happy'].render(phase=0)
device.display(img)

# Or run the full animation loop
run(device, state='happy', auto=False, fps=15)
```
