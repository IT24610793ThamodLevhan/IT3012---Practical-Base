# agent.py
import heapq
import math
import random


class SimpleReflexAgent:
    """
    A simple reflex agent that acts strictly based on immediate percepts
    using condition-action (IF-THEN) rules without any internal memory or state.
    """

    def sense_and_act(self, percept: dict) -> str:
        # Condition-Action Rules:
        if percept.get('food_here'):
            return 'Up'
        elif percept.get('wall_ahead'):
            return 'Right'
        else:
            return 'Up'


class ModelBasedAgent:
    """
    A model-based reflex agent that maintains an internal state (memory)
    to track position, visited cells, and previous actions to make informed decisions and break loops.
    """

    def __init__(self):
        self.current_pos = (0, 0)
        self.visited_cells = {(0, 0)}
        self.last_action = None
        self.last_percept = None
        self.action_history = []

    def update_state(self, percept: dict):
        """Transition & Sensor Model: updates the internal state based on percept and last action."""
        if self.last_action == 'Up':
            self.current_pos = (self.current_pos[0], self.current_pos[1] + 1)
        elif self.last_action == 'Down':
            self.current_pos = (self.current_pos[0], self.current_pos[1] - 1)
        elif self.last_action == 'Left':
            self.current_pos = (self.current_pos[0] - 1, self.current_pos[1])
        elif self.last_action == 'Right':
            self.current_pos = (self.current_pos[0] + 1, self.current_pos[1])

        self.visited_cells.add(self.current_pos)
        self.last_percept = percept

    def sense_and_act(self, percept: dict) -> str:
        # Step 1: Update internal state (Transition & Sensor model)
        self.update_state(percept)

        # Step 2: Query memory and condition-action rules
        left_pos = (self.current_pos[0] - 1, self.current_pos[1])
        left_is_visited = left_pos in self.visited_cells

        if percept.get('food_here'):
            action = 'Up'
        elif percept.get('wall_ahead'):
            # Example rule: IF wall_ahead AND left_is_visited THEN turn_right
            if self.last_action == 'Right':
                action = 'Down'
            elif self.last_action == 'Down':
                action = 'Left'
            elif self.last_action == 'Left' or left_is_visited:
                action = 'Right'
            else:
                action = 'Left'
        else:
            action = 'Up'

        # Step 3: Record chosen action
        self.last_action = action
        self.action_history.append(action)
        return action


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept.get('agent_pos')
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)


class SearchAgent:
    """
    A search agent that uses search algorithms and heuristic functions to navigate grid environments.
    """

    def manhattan_distance(self, pos, goal):
        """
        Calculates the Manhattan distance using the formula h(n) = |x_1 - x_2| + |y_1 - y_2|
        and returns the integer value.
        """
        return int(abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]))

    def euclidean_distance(self, pos, goal):
        """
        Calculates the straight-line distance using the formula h(n) = sqrt((x_1 - x_2)^2 + (y_1 - y_2)^2).
        """
        return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        """
        Finds the shortest path from start_pos to goal_pos using Breadth-First Search (BFS).
        """
        from collections import deque

        queue = deque([(start_pos, [])])
        visited = {start_pos}
        walls_set = set(walls)
        width, height = grid_size

        while queue:
            current_pos, path = queue.popleft()

            if current_pos == goal_pos:
                return path

            directions = [
                ('Up', (0, 1)),
                ('Down', (0, -1)),
                ('Left', (-1, 0)),
                ('Right', (1, 0))
            ]

            for action, (dx, dy) in directions:
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)

                if (0 <= next_pos[0] < width and
                        0 <= next_pos[1] < height and
                        next_pos not in walls_set and
                        next_pos not in visited):
                    visited.add(next_pos)
                    queue.append((next_pos, path + [action]))

        return None

    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan'):
        """
        Finds the optimal path from start_pos to goal_pos using A* Search algorithm.
        """
        # Choose heuristic function
        if heuristic_type == 'euclidean':
            heuristic_func = self.euclidean_distance
        else:
            heuristic_func = self.manhattan_distance

        start = tuple(start_pos)
        goal = tuple(goal_pos)
        walls_set = {tuple(w) for w in walls}
        width, height = grid_size

        # Initialize empty priority queue and reached_states set
        pq = []
        reached_states = set()

        # Initial node: (f_cost, g_cost, current_pos, path_taken)
        g_0 = 0
        h_0 = heuristic_func(start, goal)
        f_0 = g_0 + h_0
        heapq.heappush(pq, (f_0, g_0, start, []))

        # While loop to process the queue
        while pq:
            _, g_cost, current_pos, path_taken = heapq.heappop(pq)

            if current_pos in reached_states:
                continue

            # Check if goal is reached
            if current_pos == goal:
                return path_taken

            reached_states.add(current_pos)

            # Node expansion: check adjacent cells (Up, Down, Left, Right)
            directions = [
                ('Up', (0, 1)),
                ('Down', (0, -1)),
                ('Left', (-1, 0)),
                ('Right', (1, 0))
            ]

            for action, (dx, dy) in directions:
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)

                # Valid neighbor check (within bounds, not a wall, and not reached)
                if (0 <= next_pos[0] < width and
                        0 <= next_pos[1] < height and
                        next_pos not in walls_set and
                        next_pos not in reached_states):
                    g_new = g_cost + 1
                    h_new = heuristic_func(next_pos, goal)
                    f_new = g_new + h_new
                    heapq.heappush(pq, (f_new, g_new, next_pos, path_taken + [action]))

        return None