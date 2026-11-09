"""MOCK of the Sequoia kit's poker_agent_sdk, rebuilt from the public docs.

The interface (Agent.act(state, memory) -> (Action, bytes), Action.fold/check/call/raise_to,
state.can(kind), state.legal_actions) follows the published "Build and submit" page.
Field names are best guesses; swap in the real SDK when the kit is released.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class Action:
    kind: str
    amount: Optional[int] = None
    cards: Optional[tuple] = None  # variant "keep-reveal-discard": (keep, reveal)

    @staticmethod
    def choose(keep, reveal) -> "Action":
        """Keep-reveal-discard variant. keep/reveal are card strings ("As") or indices 0-2
        into state.cards; the third card is discarded. Action("choose") = automatic choice."""
        return Action("choose", None, (keep, reveal))

    @staticmethod
    def fold() -> "Action":
        return Action("fold")

    @staticmethod
    def check() -> "Action":
        return Action("check")

    @staticmethod
    def call() -> "Action":
        return Action("call")

    @staticmethod
    def raise_to(total: int) -> "Action":
        """total = your whole bet for the current betting round."""
        return Action("raise", total)


@dataclass(frozen=True)
class LegalAction:
    kind: str
    min_amount: Optional[int] = None
    max_amount: Optional[int] = None


@dataclass(frozen=True)
class Event:
    kind: str  # hand, ante, blind, fold, check, call, raise, board, show, win, stack
    seat: Optional[int] = None
    amount: Optional[int] = None
    cards: tuple = ()


@dataclass(frozen=True)
class SeatInfo:
    seat: int
    stack: int
    bet: int
    status: str  # active, folded, all_in, busted
    shown_cards: tuple = ()


@dataclass(frozen=True)
class Config:
    table_id: str
    seat: int
    variant: str
    players: int
    starting_stack: int
    blinds: tuple  # schedule: ((sb, bb), ...)
    hands_per_level: int
    hand_cap: int
    payouts: tuple
    decision_ms: int
    time_bank_ms: int
    memory_bytes: int


@dataclass
class State:
    # the hand
    seat: int
    cards: tuple
    board: tuple
    pot: int
    to_call: int
    legal_actions: list
    history: list
    phase: str
    # the table
    seats: list
    button: int
    # the tournament
    blind_level: int
    blinds: tuple  # current (sb, bb)
    hands_until_next_level: int
    hand_number: int
    players_remaining: int
    bust_order: list
    # misc
    events: list
    config: Config
    time_bank_ms: float
    variant: dict = field(default_factory=dict)

    def can(self, kind: str) -> Optional[LegalAction]:
        for a in self.legal_actions:
            if a.kind == kind:
                return a
        return None


class Agent:
    def act(self, state: State, memory: bytes) -> tuple[Action, bytes]:
        raise NotImplementedError
