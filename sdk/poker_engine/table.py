"""Local sit-and-go simulator (MOCK of the real kit's engine).

Variants: "keep-reveal-discard" (default: 3 cards, keep 1 private, reveal 1, discard 1)
and "reference-holdem".

Simplifications vs. a full rules engine (fine for practice):
- a short all-in raise does not block later re-raises,
- no antes by default, dead-button rules simplified (button moves to next live seat).
"""
from __future__ import annotations

import math
import random
import time

from poker_agent_sdk import Action, Config, Event, LegalAction, SeatInfo, State
from .cards import DECK, best_hand

DEFAULTS = dict(
    variant="keep-reveal-discard",  # or "reference-holdem"
    starting_stack=1500,
    blinds=((10, 20), (15, 30), (25, 50), (50, 100), (75, 150), (100, 200),
            (150, 300), (200, 400), (300, 600), (400, 800), (600, 1200), (1000, 2000)),
    hands_per_level=10,
    hand_cap=150,
    payouts=(50, 30, 20),
    decision_ms=1,
    time_bank_ms=100,
    memory_bytes=64 * 1024,
)
DELIVERY_ALLOWANCE_MS = 1.0  # guess at the "small delivery allowance"


def icm(stacks, payouts):
    """Malmuth-Harville ICM. Returns expected payout per index."""
    ev = [0.0] * len(stacks)

    def rec(remaining, prob, place):
        if place >= len(payouts) or not remaining:
            return
        tot = sum(stacks[i] for i in remaining)
        for i in remaining:
            p = prob * stacks[i] / tot
            ev[i] += p * payouts[place]
            rec([j for j in remaining if j != i], p, place + 1)

    rec([i for i, s in enumerate(stacks) if s > 0], 1.0, 0)
    return ev


class SeatPlayer:
    """One seat at one table: a pool of agent copies, like production."""

    def __init__(self, copies, stats):
        self.copies = copies
        self.stats = stats


class Table:
    def __init__(self, players, table_id, rng, settings=None):
        s = dict(DEFAULTS, **(settings or {}))
        self.s = s
        self.n = len(players)
        self.players = players
        self.rng = rng
        self.table_id = table_id
        self.payouts = tuple(s["payouts"][: self.n])
        self.stacks = [s["starting_stack"]] * self.n
        self.memory = [b""] * self.n
        self.bank = [float(s["time_bank_ms"])] * self.n
        self.seen = [0] * self.n  # index into self.events already delivered
        self.events: list[Event] = []
        self.bust_order: list[int] = []
        self.shown = [()] * self.n
        self.button = rng.randrange(self.n)
        self.hand_number = 0
        self.krd = s["variant"] == "keep-reveal-discard"
        self.revealed = {}
        self.discarded = {}

    # ---------- helpers ----------
    def alive(self):
        return [i for i in range(self.n) if self.stacks[i] > 0]

    def next_alive(self, i, alive):
        for k in range(1, self.n + 1):
            j = (i + k) % self.n
            if j in alive:
                return j
        return i

    def emit(self, kind, seat=None, amount=None, cards=()):
        e = Event(kind, seat, amount, tuple(cards))
        self.events.append(e)
        self.hand_events.append(e)

    def level(self):
        lv = min((self.hand_number - 1) // self.s["hands_per_level"], len(self.s["blinds"]) - 1)
        return lv

    # ---------- main loop ----------
    def run(self):
        while len(self.alive()) > 1 and self.hand_number < self.s["hand_cap"]:
            self.hand_number += 1
            start_stacks = list(self.stacks)
            self.play_hand()
            newly = [i for i in range(self.n) if start_stacks[i] > 0 and self.stacks[i] == 0]
            newly.sort(key=lambda i: start_stacks[i])  # smaller starting stack busts first
            self.bust_order.extend(newly)
        return self.results()

    def results(self):
        points = [0.0] * self.n
        places = [0] * self.n
        for k, i in enumerate(self.bust_order):
            place = self.n - k  # 1-based
            places[i] = place
            points[i] = float(self.payouts[place - 1]) if place - 1 < len(self.payouts) else 0.0
        alive = self.alive()
        if len(alive) == 1:
            places[alive[0]] = 1
            points[alive[0]] = float(self.payouts[0])
        else:  # hand cap: ICM on remaining stacks
            stacks = [self.stacks[i] for i in alive]
            ev = icm(stacks, self.payouts[: len(alive)])
            for rank, i in enumerate(sorted(alive, key=lambda i: -self.stacks[i])):
                places[i] = rank + 1
            for k, i in enumerate(alive):
                points[i] = ev[k]
        return points, places

    # ---------- one hand ----------
    def play_hand(self):
        alive = self.alive()
        lv = self.level()
        sb_amt, bb_amt = self.s["blinds"][lv]
        self.button = self.next_alive(self.button, alive) if self.hand_number > 1 else self.button
        if self.button not in alive:
            self.button = self.next_alive(self.button, alive)
        deck = list(DECK)
        self.rng.shuffle(deck)
        ncards = 3 if self.krd else 2
        self.hole = {i: tuple(deck.pop() for _ in range(ncards)) for i in alive}
        self.revealed = {}
        self.discarded = {}
        self.board = []
        self.in_hand = set(alive)
        self.folded = set()
        self.contrib = [0] * self.n
        self.bet = [0] * self.n
        self.shown = [()] * self.n
        self.hand_events = []
        self.emit("hand", amount=self.hand_number)

        if len(alive) == 2:
            sb, bb = self.button, self.next_alive(self.button, alive)
        else:
            sb = self.next_alive(self.button, alive)
            bb = self.next_alive(sb, alive)
        self.post(sb, sb_amt)
        self.post(bb, bb_amt)
        if self.krd:
            self.choose_phase(alive, start=self.next_alive(bb, alive))

        first = self.next_alive(bb, alive)
        for street in range(4):
            if street == 1:
                self.deal(deck, 3)
            elif street > 1:
                self.deal(deck, 1)
            if street > 0:
                first = self.next_alive(self.button, alive)
                self.bet = [0] * self.n
            if len(self.live()) > 1 and len(self.can_act()) > 0:
                self.betting_round(first, bb_amt, preflop=(street == 0))
            if len(self.live()) <= 1:
                break
        # finish board if all-in
        if len(self.live()) > 1:
            while len(self.board) < 5:
                self.deal(deck, 3 if not self.board else 1)
        self.showdown()
        for i in alive:
            self.emit("stack", seat=i, amount=self.stacks[i])

    def live(self):
        return [i for i in self.in_hand if i not in self.folded]

    def can_act(self):
        return [i for i in self.live() if self.stacks[i] > 0]

    def post(self, i, amt):
        a = min(amt, self.stacks[i])
        self.stacks[i] -= a
        self.bet[i] += a
        self.contrib[i] += a
        self.emit("blind", seat=i, amount=a)

    def deal(self, deck, k):
        cards = [deck.pop() for _ in range(k)]
        self.board.extend(cards)
        self.emit("board", cards=cards)

    def betting_round(self, first, bb_amt, preflop):
        current = max(self.bet)
        min_inc = bb_amt
        acted = set()
        i = first
        alive = list(self.in_hand)
        guard = 0
        while guard < 1000:
            guard += 1
            if len(self.live()) <= 1:
                return
            actors = self.can_act()
            if not actors:
                return
            if all(p in acted and self.bet[p] == current for p in actors):
                return
            if len(actors) == 1 and self.bet[actors[0]] >= current:
                return
            if i in actors and not (i in acted and self.bet[i] == current):
                kind, amount = self.decide(i, current, min_inc)
                if kind == "fold":
                    self.folded.add(i)
                    self.emit("fold", seat=i)
                elif kind == "check":
                    self.emit("check", seat=i)
                elif kind == "call":
                    add = min(current - self.bet[i], self.stacks[i])
                    self.move(i, add)
                    self.emit("call", seat=i, amount=add)
                elif kind == "raise":
                    add = amount - self.bet[i]
                    self.move(i, add)
                    if amount - current >= min_inc:
                        min_inc = amount - current
                    current = max(current, amount)
                    acted = set()
                    self.emit("raise", seat=i, amount=amount)
                acted.add(i)
            i = self.next_alive(i, alive)

    def move(self, i, add):
        self.stacks[i] -= add
        self.bet[i] += add
        self.contrib[i] += add

    def legal(self, i, current, min_inc):
        to_call = current - self.bet[i]
        la = []
        if to_call > 0:
            la.append(LegalAction("fold"))
            c = min(to_call, self.stacks[i])
            la.append(LegalAction("call", c, c))
        else:
            la.append(LegalAction("check"))
        others = [p for p in self.can_act() if p != i]
        if self.stacks[i] > to_call and others:
            mx = self.bet[i] + self.stacks[i]
            mn = min(current + min_inc, mx)
            la.append(LegalAction("raise", mn, mx))
        return la

    # ---------- keep-reveal-discard ----------
    def choose_phase(self, alive, start):
        """Every live player picks keep/reveal/discard simultaneously (no one sees others'
        choices until all have chosen); then reveals are public as `reveal` events."""
        order = [start] + [self.next_alive(start, alive)]
        while len(order) < len(alive):
            order.append(self.next_alive(order[-1], alive))
        current = max(self.bet)
        picks = {}
        for i in order:
            la = [LegalAction("choose")]
            state = self.build_state(i, la, current, phase="choose")
            action = self.ask(i, state)
            picks[i] = self.parse_choice(i, action)
        for i in order:
            keep, reveal, discard = picks[i]
            self.hole[i] = (keep, reveal)
            self.revealed[i] = reveal
            self.discarded[i] = discard
            self.shown[i] = (reveal,)
        for i in order:
            self.emit("reveal", seat=i, cards=(self.revealed[i],))

    def parse_choice(self, i, action):
        h = self.hole[i]
        auto = (h[0], h[1], h[2])  # automatic choice: keep 1st, reveal 2nd, discard 3rd
        if action is None:
            return auto
        st = self.players[i].stats
        if getattr(action, "kind", None) != "choose":
            st.fail("illegal", f"kind {getattr(action, 'kind', None)!r} not offered (['choose'])")
            return auto
        if action.cards is None:
            return auto
        try:
            keep, reveal = action.cards
            keep = h[keep] if isinstance(keep, int) else keep
            reveal = h[reveal] if isinstance(reveal, int) else reveal
            if keep not in h or reveal not in h or keep == reveal:
                raise ValueError(f"{action.cards!r} not two distinct cards of {h}")
        except Exception as e:  # noqa: BLE001
            st.fail("illegal", f"choose: {e}")
            return auto
        discard = next(c for c in h if c not in (keep, reveal))
        return (keep, reveal, discard)

    def ask(self, i, state):
        """Call the bot with timing/memory rules. Returns its Action, or None on failure."""
        player = self.players[i]
        st = player.stats
        agent = self.rng.choice(player.copies)
        t0 = time.perf_counter_ns()
        try:
            out = agent.act(state, self.memory[i])
            err = None
        except Exception as e:  # noqa: BLE001
            out, err = None, f"{type(e).__name__}: {e}"
        ms = (time.perf_counter_ns() - t0) / 1e6
        st.times.append(ms)
        st.decisions += 1
        over = ms - self.s["decision_ms"] - DELIVERY_ALLOWANCE_MS
        if over > 0:
            self.bank[i] -= over
            if self.bank[i] < 0:
                self.bank[i] = 0.0
                st.fail("timeout", f"act took {ms:.2f} ms, bank exhausted")
                return None
        if err:
            st.fail("exception", err)
            return None
        try:
            action, mem = out
            if not isinstance(mem, (bytes, bytearray)):
                raise TypeError("memory must be bytes")
        except Exception as e:  # noqa: BLE001
            st.fail("bad_return", f"{type(e).__name__}: {e}")
            return None
        self.memory[i] = bytes(mem[: self.s["memory_bytes"]])
        self.seen[i] = len(self.events)
        if len(mem) > self.s["memory_bytes"]:
            st.notes.add("memory truncated")
        return action

    def decide(self, i, current, min_inc):
        la = self.legal(i, current, min_inc)
        fallback = ("check", None) if any(a.kind == "check" for a in la) else ("fold", None)
        state = self.build_state(i, la, current)
        action = self.ask(i, state)
        if action is None:
            return fallback
        return self.validate(i, la, action, fallback)

    def validate(self, i, la, action, fallback):
        st = self.players[i].stats
        kind = getattr(action, "kind", None)
        legal = next((a for a in la if a.kind == kind), None)
        if legal is None:
            st.fail("illegal", f"kind {kind!r} not offered ({[a.kind for a in la]})")
            return fallback
        if kind == "raise":
            amt = action.amount
            if isinstance(amt, float) or type(amt).__name__.startswith("float"):
                amt = math.floor(amt)
            try:
                amt = int(amt)
            except Exception:  # noqa: BLE001
                st.fail("illegal", f"raise amount {action.amount!r}")
                return fallback
            if not (legal.min_amount <= amt <= legal.max_amount):
                st.fail("illegal", f"raise_to {amt} outside [{legal.min_amount}, {legal.max_amount}]")
                return fallback
            return ("raise", amt)
        return (kind, None)

    def build_state(self, i, la, current, phase="betting"):
        s = self.s
        alive = self.alive()
        lv = self.level()
        seats = []
        for j in range(self.n):
            if self.stacks[j] == 0 and j not in self.in_hand:
                status = "busted"
            elif j in self.folded:
                status = "folded"
            elif j in self.in_hand and self.stacks[j] == 0:
                status = "all_in"
            elif self.stacks[j] == 0:
                status = "busted"
            else:
                status = "active"
            seats.append(SeatInfo(j, self.stacks[j], self.bet[j], status, self.shown[j]))
        cfg = Config(
            table_id=self.table_id, seat=i, variant=s["variant"], players=self.n,
            starting_stack=s["starting_stack"], blinds=tuple(s["blinds"]),
            hands_per_level=s["hands_per_level"], hand_cap=s["hand_cap"], payouts=self.payouts,
            decision_ms=s["decision_ms"], time_bank_ms=s["time_bank_ms"], memory_bytes=s["memory_bytes"],
        )
        return State(
            seat=i, cards=self.hole[i], board=tuple(self.board), pot=sum(self.contrib),
            to_call=min(current - self.bet[i], self.stacks[i]), legal_actions=la,
            history=list(self.hand_events), phase=phase, seats=seats, button=self.button,
            blind_level=lv, blinds=tuple(s["blinds"][lv]),
            hands_until_next_level=s["hands_per_level"] - (self.hand_number - 1) % s["hands_per_level"],
            hand_number=self.hand_number, players_remaining=len(alive),
            bust_order=list(self.bust_order), events=self.events[self.seen[i]:], config=cfg,
            time_bank_ms=self.bank[i],
            variant=({"revealed": dict(self.revealed), "my_reveal": self.revealed.get(i),
                      "my_discard": self.discarded.get(i)} if self.krd else {}),
        )

    def showdown(self):
        live = self.live()
        if len(live) > 1:
            for i in sorted(live):
                self.shown[i] = self.hole[i]
                self.emit("show", seat=i, cards=self.hole[i])
        strength = {i: best_hand(list(self.hole[i]) + self.board) for i in live} if len(live) > 1 else {}
        levels = sorted({c for c in self.contrib if c > 0})
        prev = 0
        winnings = [0] * self.n
        for lvl in levels:
            pot = sum(min(c, lvl) - min(c, prev) for c in self.contrib)
            elig = [i for i in live if self.contrib[i] >= lvl]
            if not elig:  # everyone at this level folded: give to best remaining
                elig = live
            if len(elig) == 1:
                winners = elig
            else:
                best = max(strength[i] for i in elig)
                winners = [i for i in elig if strength[i] == best]
            share, odd = divmod(pot, len(winners))
            order = sorted(winners, key=lambda w: (w - self.button - 1) % self.n)
            for k, w in enumerate(order):
                winnings[w] += share + (1 if k < odd else 0)
            prev = lvl
        for i, w in enumerate(winnings):
            if w:
                self.stacks[i] += w
                self.emit("win", seat=i, amount=w)
