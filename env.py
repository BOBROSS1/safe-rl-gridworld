import numpy as np
import cv2
from PIL import Image

#
# SIZE = 9
# c = {1: (255, 175, 0, 1),
# 	 2: (0, 255, 0, 1),
# 	 3: (0, 0, 255, 1),
# 	 "white": (244,255,255, 1)}


# env = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
# env[3][8] = d[1]
# env[4][4] = d[2]
# env[7][8] = d[3]

layout = '''xxxxxxxxx\
		    xx-------\
		    xxxxxxxxx\
		    xx--xxxxx\
		    xxxxxxxxx\
		    xxxxx-x--\
		    x---x-xxx\
		    x-x-xx-xx\
		    xxxxxxxxx\
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
# print(walls)

# img = Image.fromarray(env, "RGBA")
# img = img.resize((300,300))
# cv2.imshow("", np.array(img))
# cv2.waitKey(5000)

# walls = [(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),
# 		   (3,1),(3,2),
# 		   (5,5),(5,7),(5,8)
# 		   (6,1),(6,2),(6,3),(6,5)
# 		   (7,1),(7,3),(7,6)]