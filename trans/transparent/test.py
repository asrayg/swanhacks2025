import time
from PIL import Image, ImageDraw, ImageFont
import logging
from driver import OLED_1in51  

OLED_WIDTH = 128
OLED_HEIGHT = 64

logging.basicConfig(level=logging.INFO)

disp = OLED_1in51()
disp.Init()

image = Image.new("1", (OLED_WIDTH, OLED_HEIGHT), "white")
draw = ImageDraw.Draw(image)

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
except:
    font = ImageFont.load_default()

draw.text((10, 20), "Hello", font=font, fill=0)

buf = disp.getbuffer(image)
disp.ShowImage(buf)

logging.info("Hello displayed.")

while True:
    time.sleep(1)
