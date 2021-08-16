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

layout_original_walled = '''-----------
					  		-xxxxxxxxx-
		    		    	-xx--------
					  		-xxxxxxxxx-
			 		  		-xx--xxxxx-
					  		-xxxxxxxxx-
					  		-xxxxx-x---
					  		-x---x-xxx-
					  		-x-x-x--xx-
					  		-xxxxxxxxx-
					  		-----------
		 	    		'''

layout_original_walled_big = '''----------------------
					  			-xxxxxxxxx-xxxxxxxxxx-
		    		    		-xx-----xxxxx-xxxx--x-
					  			-xxxxxxxxxxxxxxx---xx-
			 		  			-xx--xxxxxxxxx---xxxx-
					  			-xxxxxxxxx-xxxxxxx--x-
					  			-xxxxx-x-xxxxxx--xxxx-
					  			-x---x-xxx------xxxxx-
					  			-x-x-x--xxxxx---xxxxx-
					  			-xxxxxxxxx-xxxxxxx--x-
					  			--xxxx--x----xxx----x-
								--xx----x----x-x----x-
					  			-xxxxxxxxx-xxxxxxxxxx-
		    		    		-xx-----xxxxx-xxxx----
					  			-xxxxxxxxxxxxxxx---x--
			 		  			-xx--xxxxxxxxx---xxxx-
					  			-xxxxxxxxx-xxxxxxx--x-
					  			-xxxxx-x-xxxxxx--xxxx-
					  			-x---x-xxx------xx-xx-
					  			-x-x-x--xxxxx---x--xx-
					  			-xxxxxxxxxxxxxxxxxxxx-
					  			----------------------
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
		self.walls = walls
		self.env = env

	def render(self, player, target, step, reward):
		# make screen red/green if episode is reset/reward
		wait = 1
		if step == 99:
			self.env = np.full(self.env.shape, (0, 0, 255, 1), dtype=np.uint8)
			wait = 100
		elif reward > 0:
			self.env = np.full(self.env.shape, (0, 255, 0, 1), dtype=np.uint8)
			wait = 100

		self.env[player.y][player.x] = (255, 175, 0, 1) #blue
		self.env[target.y][target.x] = (0, 255, 0, 1) #green
		# env[enemy.y][enemy.x] = (0, 0, 255, 1) #red
		img = Image.fromarray(self.env, "RGBA")
		
		img = img.resize((400, 400))
		cv2.imshow("image", np.array(img))
		cv2.waitKey(wait)