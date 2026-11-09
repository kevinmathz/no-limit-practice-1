from poker_agent_sdk import Action, Agent as BaseAgent


class Agent(BaseAgent):
    """Never folds, never raises. Extra baseline (not in the real kit)."""

    def act(self, state, memory):
        if state.can("call"):
            return Action.call(), memory
        return Action.check(), memory
