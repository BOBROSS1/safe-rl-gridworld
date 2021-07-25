import numpy as np
import cv2
from PIL import Image

#
SIZE = 9
d = {1: (255, 175, 0, 1),
	 2: (0, 255, 0, 1),
	 3: (0, 0, 255, 1),
	 4: (244,255,255, 1)}


env = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
env[3][8] = d[1]
env[4][4] = d[2]
env[7][8] = d[3]

for i in [1]:
	for j in range(2, 9):
		env[i][j] = d[4]

# for i in range(3, 9):
# 	for j in [6]:
# 		env[i][j] = d[4]

# walls
env[1][1] = d[4]
env[1][2] = d[4]
env[1][3] = d[4]
env[1][4] = d[4]
env[1][5] = d[4]
env[1][6] = d[4]
env[1][7] = d[4]
env[3][1] = d[4]
env[3][2] = d[4]
env[6][1] = d[4]
env[7][1] = d[4]
env[6][2] = d[4]
env[6][3] = d[4]
env[7][3] = d[4]
env[5][5] = d[4]
env[6][5] = d[4]
env[7][6] = d[4]
env[5][7] = d[4]
env[5][8] = d[4]


# for i in [3]:
# 	for j in range(2, 4):
# 		env[i][j] = d[4]

# for i in [6]:
# 	for j in range(1, 3):
# 		env[i][j] = d[4]

# for i in range(5, 6):
# 	for j in [5]:
# 		env[i][j] = d[4]

# for i in range(6, 7):
# 	for j in [7]:
# 		env[i][j] = d[4]

img = Image.fromarray(env, "RGBA")
img = img.resize((300,300))

cv2.imshow("", np.array(img))
cv2.waitKey(5000)

# print(env)


