from poker_agent_sdk import Action, Agent as BaseAgent


class Agent(BaseAgent):
    """Checks when free, otherwise folds (the upload test-match opponent)."""

    def act(self, state, memory):
        if state.can("check"):
            return Action.check(), memory
        if state.can("fold"):
            return Action.fold(), memory
        first = state.legal_actions[0]
        return Action(first.kind, first.min_amount), memory
