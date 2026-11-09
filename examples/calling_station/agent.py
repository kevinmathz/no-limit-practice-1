from poker_agent_sdk import Action, Agent as BaseAgent


class Agent(BaseAgent):
    """Never folds, never raises. Extra baseline (not in the real kit)."""

    def act(self, state, memory):
        if state.phase != "betting":
            first = state.legal_actions[0]
            return Action(first.kind, first.min_amount), memory
        if state.can("call"):
            return Action.call(), memory
        return Action.check(), memory
