import random

class Agent:
    def __init__(self, places_no_walls, walls, SHIELD_ON, N_ACTIONS,  SIZE, x=None, y=None, random_init=True):
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
        self.N_ACTIONS = N_ACTIONS

    def __sub__(self, other):
        return (self.y - other.y, self.x - other.x)
	
    def state(self):
        return (self.y, self.x)

    def action(self, action):
        # change standing still action from 4 to 8 if N_ACTIONS is 5 (standing still action is encoded last in DFA)
        if action == 4 and self.N_ACTIONS == 5:
            action = 8

        if action == 0:
            # up
            self.move(y=-1, x=0)
        elif action == 1:
			# down
            self.move(y=1, x=0)
        elif action == 2:
            # left
            self.move(y=0, x=-1)
        elif action == 3:
            # right
            self.move(y=0, x=1)
        elif action == 4:
			# down left
            self.move(y=1, x=-1)
        elif action == 5:
			# down right
            self.move(y=1, x=1)
        elif action == 6:
			# up left
            self.move(y=-1, x=-1)
        elif action ==7:
			# up right
            self.move(y=-1, x=1)
        elif action == 8:
			# standing still (has to be last (dfa))
            self.move(y=0, x=0)
	
    def get_potential_position(self, action):
        # change standing still action from 4 to 8 if N_ACTIONS is 5 (standing still action is encoded last in DFA)
        if action == 4 and self.N_ACTIONS == 5:
            action = 8 

        if action == 0:
            # up
            return (self.y - 1, self.x + 0)
        elif action == 1:
			# down
            return (self.y + 1, self.x + 0)
        elif action == 2:
            # left
            return (self.y + 0, self.x - 1)
        elif action == 3:
            # right
            return (self.y + 0, self.x + 1)
        elif action == 4:
            # down left
            return (self.y + 1, self.x - 1)
        elif action == 5:
            # down right
            return (self.y + 1, self.x + 1)
        elif action == 6:
            # up left
            return (self.y - 1, self.x - 1)
        elif action == 7:
            # up right
            return (self.y - 1, self.x + 1)
        elif action == 8:
            # standing still (has to be last, (dfa))
            return (self.y, self.x)

    def move(self, x=False, y=False):
        # handle walls (y, x)
        check = (self.y + y, self.x + x)
        if check in self.walls:
            if self.SHIELD_ON:
                raise Exception("Shield not working, agent should not make mistakes")
            self.x = self.x
            self.y = self.y
        # handle boundaries env
        elif self.x + x < 0:
            self.x = 0
        elif self.x + x > self.size-1:
            self.x = self.size-1
        elif self.y + y < 0:
            self.y = 0
        elif self.y + y > self.size-1:
            self.y = self.size-1
        else:
            self.x += x
            self.y += y

class ShadowAgent:
    def __init__(self, y, x, N_ACTIONS):
        self.y = y
        self.x = x  
        self.N_ACTIONS = N_ACTIONS

    def __sub__(self, other):
        return (self.y - other.y, self.x - other.x)
    
    def state(self):
        return (self.y, self.x)

    def get_potential_position(self, action):
        # change standing still action from 4 to 8 if N_ACTIONS is 5 (standing still action is encoded last in DFA)
        if action == 4 and self.N_ACTIONS == 5:
            action = 8 

        if action == 0:
            # up
            return (self.y - 1, self.x + 0)
        elif action == 1:
			# down
            return (self.y + 1, self.x + 0)
        elif action == 2:
            # left
            return (self.y + 0, self.x - 1)
        elif action == 3:
            # right
            return (self.y + 0, self.x + 1)
        elif action == 4:
            # down left
            return (self.y + 1, self.x - 1)
        elif action == 5:
            # down right
            return (self.y + 1, self.x + 1)
        elif action == 6:
            # up left
            return (self.y - 1, self.x - 1)
        elif action == 7:
            # up right
            return (self.y - 1, self.x + 1)
        elif action == 8:
            # standing still (has to be last, (dfa))
            return (self.y, self.x)