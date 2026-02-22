#!/bin/bash
# BMO Desk Companion - Setup Script
# Run with: bash setup.sh

set -e

echo "================================================"
echo "  BMO OLED Desk Companion - Setup"
echo "================================================"
echo ""

# 1. Enable I2C and SPI
echo "[1/4] Enabling I2C and SPI interfaces..."
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_spi 0
echo "  ✓ I2C and SPI enabled"

# 2. System dependencies
echo ""
echo "[2/4] Installing system packages..."
sudo apt-get update -q
sudo apt-get install -y -q \
    python3-pip \
    python3-pil \
    python3-smbus \
    i2c-tools \
    libopenjp2-7
echo "  ✓ System packages installed"

# 3. Python packages
echo ""
echo "[3/4] Installing Python packages..."
pip3 install --break-system-packages \
    luma.oled \
    pillow
echo "  ✓ Python packages installed"

# 4. Verify I2C device detected
echo ""
echo "[4/4] Scanning I2C bus for your display..."
sudo i2cdetect -y 1
echo ""
echo "  If you see 0x3C or 0x3D above, your display is detected!"
echo "  If nothing shows, check your wiring."
echo ""
echo "================================================"
echo "  Setup complete! Run the face with:"
echo ""
echo "  python3 face.py                    # idle face"
echo "  python3 face.py --state happy      # happy face"
echo "  python3 face.py --state sad        # sad face"
echo "  python3 face.py --auto             # cycle all faces"
echo ""
echo "  SPI users:"
echo "  python3 face.py --interface spi --driver ssd1309"
echo "================================================"
