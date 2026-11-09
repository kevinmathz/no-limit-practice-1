import random

from poker_agent_sdk import Action, Agent as BaseAgent


class Agent(BaseAgent):
    """Random legal actions. Extra baseline (not in the real kit)."""

    def act(self, state, memory):
        a = random.choice(state.legal_actions)
        if a.kind == "choose":
            keep, reveal = random.sample(range(3), 2)
            return Action.choose(keep, reveal), memory
        if a.kind == "raise":
            return Action.raise_to(random.randint(a.min_amount, a.max_amount)), memory
        return Action(a.kind), memory
