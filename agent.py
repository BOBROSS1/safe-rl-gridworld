import random

class Agent:
    def __init__(self, places_no_walls, walls, SHIELD_ON, SIZE, x=None, y=None, random_init=True):
        if random_init:
            position = random.choice(places_no_walls)
            self.y = int(position[0])
            self.x = int(position[1])
            places_no_walls.remove(position)
        else:
            self.x = x
            self.y = y

        self.walls = walls
        self.size = SIZE
        self.SHIELD_ON = SHIELD_ON

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)
	
    def state(self):
        return (self.y, self.x)

    def action(self, choice):
        if choice == 0:
            # up
            self.move(self.walls, self.size, self.SHIELD_ON, y=-1, x=0)
        elif choice == 1:
			# down
            self.move(self.walls, self.size, self.SHIELD_ON, y=1, x=0)
        elif choice == 2:
            # left
            self.move(self.walls, self.size, self.SHIELD_ON, y=0, x=-1)
        elif choice == 3:
            # right
            self.move(self.walls, self.size, self.SHIELD_ON, y=0, x=1)
        elif choice == 4:
			# down left
            self.move(self.walls, self.size, self.SHIELD_ON, y=1, x=-1)
        elif choice == 5:
			# down right
            self.move(self.walls, self.size, self.SHIELD_ON, y=1, x=1)
        elif choice == 6:
			# up left
            self.move(self.walls, self.size, self.SHIELD_ON, y=-1, x=-1)
        elif choice ==7:
			# up right
            self.move(self.walls, self.size, self.SHIELD_ON, y=-1, x=1)
        elif choice == 8:
			# standing still (has to be last (dfa))
            self.move(self.walls, self.size, self.SHIELD_ON, y=0, x=0)
	
    def get_potential_position(self, choice):
        if choice == 0:
            # up
            return (self.y - 1, self.x + 0)
        elif choice == 1:
			# down
            return (self.y + 1, self.x + 0)
        elif choice == 2:
            # left
            return (self.y + 0, self.x - 1)
        elif choice == 3:
            # right
            return (self.y + 0, self.x + 1)
        elif choice == 4:
            # down left
            return (self.y + 1, self.x - 1)
        elif choice == 5:
            # down right
            return (self.y + 1, self.x + 1)
        elif choice == 6:
            # up left
            return (self.y - 1, self.x - 1)
        elif choice == 7:
            # up right
            return (self.y - 1, self.x + 1)
        elif choice == 8:
            # standing still (has to be last, (dfa))
            return (self.y, self.x)

    def move(self, walls, size, SHIELD_ON, x=False, y=False):
        # handle walls (y, x)
        check = (self.y + y, self.x + x)
        if check in walls:
            if SHIELD_ON:
                raise Exception("Shield not working, agent should not make mistakes")
            self.x = self.x
            self.y = self.y
        # handle boundaries env
        elif self.x + x < 0:
            self.x = 0
        elif self.x + x > size-1:
            self.x = size-1
        elif self.y + y < 0:
            self.y = 0
        elif self.y + y > size-1:
            self.y = size-1
        else:
            self.x += x
            self.y += y
