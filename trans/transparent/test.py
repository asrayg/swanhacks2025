# --- SIMPLE STATIC HELLO ---

import time
from PIL import Image, ImageDraw, ImageFont
import logging
from driver import OLED_1in51   # fix this import to match your file name

OLED_WIDTH = 128
OLED_HEIGHT = 64

logging.basicConfig(level=logging.INFO)

disp = OLED_1in51()
disp.Init()

# Create a blank image (white = off / transparent)
image = Image.new("1", (OLED_WIDTH, OLED_HEIGHT), "white")
draw = ImageDraw.Draw(image)

# Load font
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
except:
    font = ImageFont.load_default()

# Draw solid blue text -> fill=0 (blue pixel ON)
draw.text((10, 20), "Hello", font=font, fill=0)

# Convert to buffer & show
buf = disp.getbuffer(image)
disp.ShowImage(buf)

logging.info("Hello displayed.")

# Keep on screen forever
while True:
    time.sleep(1)
