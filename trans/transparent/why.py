#!/usr/bin/python3
# -*- coding:utf-8 -*-

import time
import board
import digitalio
import busio
from PIL import Image, ImageDraw, ImageFont
import logging

logging.basicConfig(level=logging.INFO)

OLED_WIDTH = 128
OLED_HEIGHT = 64


class OLED_1in51:
    def __init__(self):
        self.spi = busio.SPI(board.SCLK, MOSI=board.MOSI)

        self.dc = digitalio.DigitalInOut(board.D24)
        self.rst = digitalio.DigitalInOut(board.D25)
        self.cs = digitalio.DigitalInOut(board.D8)

        self.dc.direction = digitalio.Direction.OUTPUT
        self.rst.direction = digitalio.Direction.OUTPUT
        self.cs.direction = digitalio.Direction.OUTPUT

        self.width = OLED_WIDTH
        self.height = OLED_HEIGHT

        while not self.spi.try_lock():
            pass
        self.spi.configure(baudrate=8000000, phase=0, polarity=0)
        self.spi.unlock()

    def command(self, cmd):
        self.dc.value = 0
        self.cs.value = 0
        self.spi.write(bytes([cmd]))
        self.cs.value = 1

    def data(self, data):
        self.dc.value = 1
        self.cs.value = 0
        self.spi.write(bytes(data))
        self.cs.value = 1

    def Init(self):
        logging.info("Initializing transparent OLED...")

        self.reset()

        self.command(0xAE) 
        self.command(0xD5); self.data([0xA0])
        self.command(0xA8); self.data([0x3F])
        self.command(0xD3); self.data([0x00])
        self.command(0x40)
        self.command(0xA1)
        self.command(0xC8)
        self.command(0xDA); self.data([0x12])
        self.command(0x81); self.data([0x7F])
        self.command(0xA4)
        self.command(0xA6)
        self.command(0xD9); self.data([0xF1])
        self.command(0xDB); self.data([0x40])
        self.command(0xAF) 

        logging.info("OLED init OK.")

    def reset(self):
        self.rst.value = 0
        time.sleep(0.05)
        self.rst.value = 1
        time.sleep(0.05)

    def getbuffer(self, image):
        buf = [0x00] * (self.width * self.height // 8)
        img = image.convert("1")
        pixels = img.load()

        for y in range(self.height):
            for x in range(self.width):
                if pixels[x, y] == 0: 
                    buf[x + (y // 8) * self.width] |= (1 << (y % 8))

        return buf

    def ShowImage(self, buf):
        for page in range(8):
            self.command(0xB0 + page)
            self.command(0x00)
            self.command(0x10)
            start = page * self.width
            end = (page + 1) * self.width
            self.data(buf[start:end])

    def clear(self):
        self.ShowImage([0x00] * (self.width * self.height // 8))


if __name__ == "__main__":
    disp = OLED_1in51()
    disp.Init()

    image = Image.new("1", (OLED_WIDTH, OLED_HEIGHT), "white")  
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    draw.text((20, 18), "HELLO", fill="black")

    buf = disp.getbuffer(image)
    disp.ShowImage(buf)

    logging.info("Displayed HELLO in blue! Static image.")
    while True:
        time.sleep(1)
