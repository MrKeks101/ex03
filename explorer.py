# EXPLORER AGENT
# @Author: Tacla, UTFPR
#
### It walks randomly in the environment looking for victims. When half of the
### exploration has gone, the explorer goes back to the base.

import sys
import os
import random
import math
from abc import ABC, abstractmethod
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0

class Explorer(AbstAgent):
    """ class attribute """
    MAX_DIFFICULTY = 1             # the maximum degree of difficulty to enter into a cell
    
    def __init__(self, env, config_file, resc,exp):
        """ Construtor do agente random on-line
        @param env: a reference to the environment 
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """

        super().__init__(env, config_file)
        self.walk_stack = Stack()  # a stack to store the movements
        self.walk_time = 0         # time consumed to walk when exploring (to decide when to come back)
        self.set_state(VS.ACTIVE)  # explorer is active since the begin
        self.resc = resc           # reference to the rescuer agent
        self.x = 0                 # current x position relative to the origin 0
        self.y = 0                 # current y position relative to the origin 0
        self.map = Map()           # create a map for representing the environment
        self.victims = {}          # a dictionary of found victims: (seq): ((x,y), [<vs>])
                                   # the key is the seq number of the victim,(x,y) the position, <vs> the list of vital signals
        self.explored_cells = set()
        self.obstacle_cache = {}

        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())
        self.tried_directions = {}
        self.explorer_number = exp

    def get_next_position(self):
        """ Uses ONLINE-DFS to explore the map
        """
        """
        position = (self.x,self.y)
        if position not in self.tried_directions:
            self.tried_directions[position] = set()
        # Check the neighborhood walls and grid limits
        obstacles = self.check_walls_and_lim()

        number = (self.explorer_number - 1) * 2

        for i in range(0 + number,8 + number):
            j = i % 8
            if i not in self.tried_directions[position]:
                if obstacles[j] == VS.CLEAR:
                    self.tried_directions[position].add(j)
                    return Explorer.AC_INCR[j]
                else:
                    self.tried_directions[position].add(j)
        return None
        """
        position = (self.x, self.y)

        # Ignora se já está totalmente explorada
        if position in self.explored_cells:
            return None

        # Inicializa estruturas se necessário
        if position not in self.tried_directions:
            self.tried_directions[position] = set()

        if position not in self.obstacle_cache:
            self.obstacle_cache[position] = self.check_walls_and_lim()

        obstacles = self.obstacle_cache[position]

        # Heurística simples: direções "cardinais" em ordem preferencial (ex: cima, direita, baixo, esquerda)
        preferred_dirs = [0, 2, 4, 6, 1, 3, 5, 7]  # ajuste se quiser outra prioridade
        number = (self.explorer_number - 1) * 2   # para diversificar entre exploradores
        rotated_dirs = [(i + number) % 8 for i in preferred_dirs]

        for j in rotated_dirs:
            if j not in self.tried_directions[position]:
                self.tried_directions[position].add(j)
                if obstacles[j] == VS.CLEAR:
                    return Explorer.AC_INCR[j]

        # Todas direções testadas → célula marcada como explorada
        self.explored_cells.add(position)
        return None
        
    def explore(self):
        # get an random increment for x and y       
        dx, dy = self.get_next_position()

        # Moves the body to another position  
        rtime_bef = self.get_rtime()
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()


        # Test the result of the walk action
        # Should never bump, but for safe functionning let's test
        if result == VS.BUMPED:
            # update the map with the wall
            self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())
            #print(f"{self.NAME}: Wall or grid limit reached at ({self.x + dx}, {self.y + dy})")

        if result == VS.EXECUTED:
            # check for victim returns -1 if there is no victim or the sequential
            # the sequential number of a found victim
            self.walk_stack.push((dx, dy))

            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy

            # update the walk time
            self.walk_time = self.walk_time + (rtime_bef - rtime_aft)
            #print(f"{self.NAME} walk time: {self.walk_time}")

            # Check for victims
            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                self.victims[vs[0]] = ((self.x, self.y), vs)
                #print(f"{self.NAME} Victim found at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
                #print(f"{self.NAME} Seq: {seq} Vital signals: {vs}")
            
            # Calculates the difficulty of the visited cell
            difficulty = (rtime_bef - rtime_aft)
            if dx == 0 or dy == 0:
                difficulty = difficulty / self.COST_LINE
            else:
                difficulty = difficulty / self.COST_DIAG

            # Update the map with the new cell
            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())
            #print(f"{self.NAME}:at ({self.x}, {self.y}), diffic: {difficulty:.2f} vict: {seq} rtime: {self.get_rtime()}")

        return

    def come_back(self):
        dx, dy = self.walk_stack.pop()
        dx = dx * -1
        dy = dy * -1

        result = self.walk(dx, dy)
        if result == VS.BUMPED:
            print(f"{self.NAME}: when coming back bumped at ({self.x+dx}, {self.y+dy}) , rtime: {self.get_rtime()}")
            return
        
        if result == VS.EXECUTED:
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            #print(f"{self.NAME}: coming back at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
        
    def deliberate(self) -> bool:
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""

        # forth and back: go, read the vital signals and come back to the position

        time_tolerance = 2* self.COST_DIAG * Explorer.MAX_DIFFICULTY + self.COST_READ

        # keeps exploring while there is enough time
        if  self.walk_time < (self.get_rtime() - time_tolerance):
            self.explore()
            return True

        # no more come back walk actions to execute or already at base
        if self.walk_stack.is_empty() or (self.x == 0 and self.y == 0):
            # time to pass the map and found victims to the master rescuer
            self.resc.sync_explorers(self.map, self.victims)
            # finishes the execution of this agent
            return False
        
        # proceed to the base
        self.come_back()
        return True

