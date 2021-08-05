import numpy as np
import cv2
from PIL import Image

# x = free
# - = wall
layout_original = '''xxxxxxxxx
		    		 xx-------
					 xxxxxxxxx
			 		 xx--xxxxx
					 xxxxxxxxx
					 xxxxx-x--
					 x---x-xxx
					 x-x-xx-xx
					 xxxxxxxxx
		 		 '''

layout_nowalls = '''xxxxxxxxx
				 	xxxxxxxxx
					xxxxxxxxx
				 	xxxxxxxxx
				 	xxxxxxxxx
				 	xxxxxxxxx
				 	xxxxxxxxx
				 	xxxxxxxxx
				 	xxxxxxxxx
				 '''

class gridworld:
	def __init__(self, layout, size):
		env = np.zeros((size, size, 4), dtype=np.uint8)
		walls = []
		layout = layout.split()
		if len(layout[0]) != size:
			raise Exception('Layout must be the same width and height as SIZE (SIZExSIZE)')
		for idy, y in enumerate(layout):
			for x in range(len(y)):
				if y[x] == "-":
					env[idy][x] = (255,255,255, 1)
					walls.append((idy, x))
		#return env, walls
		self.walls = walls
		self.env = env

	def render(self, player, food):
		self.env[player.y][player.x] = (255, 175, 0, 1) #blue
		self.env[food.y][food.x] = (0, 255, 0, 1) #green
		# env[enemy.y][enemy.x] = (0, 0, 255, 1) #red

		img = Image.fromarray(self.env, "RGBA")
		img = img.resize((400, 400))
		cv2.imshow("image", np.array(img))
		cv2.waitKey(1)