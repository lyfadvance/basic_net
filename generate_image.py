import pygame
import os
import PIL.Image,PIL.ImageDraw,PIL.ImageFont
pygame.init()
text="test"
#font=pygame.font.SysFont('Microsoft YhHei',64)
im=PIL.Image.new("RGB",(300,300),(0,255,255))
dr=PIL.ImageDraw.Draw(im)
font=PIL.ImageFont.truetype("/usr/share/fonts/opentye/freefont/FreeSans.otf",64)

dr.text((100,100),text,font=font,fill="#ff0000")
im.show()
im.save("test/img_5.jpg")

