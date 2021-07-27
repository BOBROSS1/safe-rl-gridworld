import numpy as np
import cv2
from PIL import Image

layout = '''xxxxxxxxx
		    xx-------
		    xxxxxxxxx
		    xx--xxxxx
		    xxxxxxxxx
		    xxxxx-x--
		    x---x-xxx
		    x-x-xx-xx
		    xxxxxxxxx
		'''

def generate_env(layout, SIZE):
	env = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
	walls = []
	layout = layout.split()
	for idy, y in enumerate(layout):
		for x in range(len(y)):
			if y[x] == "-":
				env[idy][x] = (255,255,255, 1)
				walls.append((idy, x))
	return env, walls
	
# env, walls = generate_env(layout, SIZE)
# img = Image.fromarray(env, "RGBA")
# img = img.resize((300,300))
# cv2.imshow("", np.array(img))
# cv2.waitKey(5000)