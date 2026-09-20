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


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept.get('agent_pos')
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)