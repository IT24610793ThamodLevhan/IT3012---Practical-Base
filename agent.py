# agent.py
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