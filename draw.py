import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

x = 0
y = 900

im = Image.new("RGB", (1000, 1000))
draw = ImageDraw.Draw(im)

# draw shapes
draw.rectangle(xy=(x, y, x+100, y+100), fill="white")
draw.ellipse((x, y, x+100, y+100), fill = 'blue', outline ='blue')

# draw text
fnt = ImageFont.truetype("Verdana.ttf", 72)
draw.text((10,50), "Hello", font=fnt, fill=(255,255,255,128))

image = im.resize((400, 400))
cv2.imshow("image", np.array(image))
cv2.waitKey(1000)