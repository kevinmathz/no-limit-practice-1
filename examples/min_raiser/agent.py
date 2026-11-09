from poker_agent_sdk import Action, Agent as BaseAgent


class Agent(BaseAgent):
    """Always min-raises when it can, otherwise calls/checks."""

    def act(self, state, memory):
        r = state.can("raise")
        if r:
            return Action.raise_to(r.min_amount), memory
        if state.can("call"):
            return Action.call(), memory
        return Action.check(), memory
