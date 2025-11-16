# import time
# import board
# import digitalio
# import busio
# from PIL import Image, ImageDraw, ImageFont

# print("\n=== OLED DIAGNOSTIC TOOL ===")
# print("This will test:")
# print(" - SPI stability")
# print(" - Control pins (DC, RST, CS)")
# print(" - Panel power / flicker conditions")
# print(" - Pixel addressing and orientation")
# print("================================\n")


# OLED_WIDTH = 128
# OLED_HEIGHT = 64

# class OLED_1in51:
#     def __init__(self):
#         # SPI
#         self.spi = busio.SPI(board.SCLK, MOSI=board.MOSI)

#         # Pins
#         self.dc = digitalio.DigitalInOut(board.D24)
#         self.rst = digitalio.DigitalInOut(board.D25)
#         self.cs = digitalio.DigitalInOut(board.D8)
#         for pin in [self.dc, self.rst, self.cs]:
#             pin.direction = digitalio.Direction.OUTPUT

#         # Configure SPI
#         while not self.spi.try_lock():
#             pass
#         self.spi.configure(baudrate=8000000, phase=0, polarity=0)
#         self.spi.unlock()

#         self.width = OLED_WIDTH
#         self.height = OLED_HEIGHT

#     def command(self, cmd):
#         self.dc.value = 0
#         self.cs.value = 0
#         self.spi.write(bytes([cmd]))
#         self.cs.value = 1

#     def data(self, data):
#         self.dc.value = 1
#         self.cs.value = 0
#         self.spi.write(data)
#         self.cs.value = 1

#     def reset(self):
#         print("Testing RST pin (should cause one clean flash)...")
#         self.rst.value = 0
#         time.sleep(0.2)
#         self.rst.value = 1
#         time.sleep(0.2)

#     def Init(self):
#         print("Sending OLED init sequence...")
#         self.reset()
#         cmds = [
#             0xAE, 0xD5, 0xA0, 0xA8, 0x3F, 0xD3, 0x00,
#             0x40, 0xA1, 0xC8, 0xDA, 0x12, 0x81, 0x7F,
#             0xA4, 0xA6, 0xD9, 0xF1, 0xDB, 0x40, 0xAF
#         ]
#         for c in cmds:
#             self.command(c)
#         print("Init complete.\n")

#     def getbuffer(self, image):
#         buf = [0x00] * (self.width * self.height // 8)
#         img = image.convert("1")
#         pixels = img.load()
#         for y in range(self.height):
#             for x in range(self.width):
#                 if pixels[x, y] == 0:
#                     buf[x + (y // 8) * self.width] |= (1 << (y % 8))
#         return buf

#     def ShowImage(self, buf):
#         for page in range(8):
#             self.command(0xB0 + page)
#             self.command(0x00)
#             self.command(0x10)
#             start = page * 128
#             end = start + 128
#             self.data(bytes(buf[start:end]))

#     def display_text_upside_down(self, text, font_size=20):
#         # Create blank white canvas
#         image = Image.new("1", (self.width, self.height), "white")
#         draw = ImageDraw.Draw(image)

#         # Load font
#         font = ImageFont.truetype(
#             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
#         )

#         # Center text
#         w, h = draw.textsize(text, font=font)
#         draw.text(((self.width - w) // 2, (self.height - h) // 2),
#                 text, fill="black", font=font)

#         # ROTATE 180°  (UPSIDE DOWN)
#         image = image.rotate(180)

#         # Convert to buffer + push to screen
#         buf = self.getbuffer(image)
#         self.ShowImage(buf)
