"""Design parameters of Plott and Sunder (1982).

Replaces ps1982/markets.py of the simulation-engine repository so that this package
runs standalone.  Two parts, with different provenance:

  * The DATA BLOCK below (_MARKETS and the endowment and urn constants) was
    machine-emitted by printing the engine's own objects, so it is value-for-value
    identical to the engine and can be diffed against it.
  * The METHODS on the _Market shim at the foot of this file were re-implemented by
    hand.  They reproduce the engine's arithmetic but are not textually equivalent to
    it, and `card_for` deliberately drops the engine's `rnd` argument and its
    state-redraw fallback, which serve resampled machine-agent runs and are
    unreachable here -- this package only ever reads the published state sequence.
    Verify them against the arithmetic, not by diffing against the engine.

Every value here is a published parameter of the original experiment: the dividend
table and priors (their Table 1), the realized state and information condition of
each period (their Figures 2-6), and the endowments (their Instruction Set 2).

Market 1's clue is a ten-draw urn sample rather than a state letter, so its fully
revealing price is a function of the card drawn, not of the state; `paper_clue_cards`
holds the cards the original printed.  Markets 6-8 are not part of this paper; they
come from the engine's own market table and are carried unused so that the data block
stays a faithful copy of it.
"""

from fractions import Fraction

INITIAL_CERTS = 2      # certificates per investor
INITIAL_CASH = 10000   # francs of working capital per investor
FIXED_COST = 10000     # deducted at period end

# prior_ev[t] is type t's expected dividend under the prior; the highest of them
# is vbar, the price a market with no information should support.
_MARKETS = {
    1: dict(
        number=1,
        n_per_type=3,
        insiders_per_type=1,
        states=('X', 'Y'),
        prior={'X': 0.3333333333333333, 'Y': 0.6666666666666666},
        dividends={'I': {'X': 150, 'Y': 350}, 'II': {'X': 250, 'Y': 300}, 'III': {'X': 300, 'Y': 100}},
        prior_ev={'I': 283.3333333333333, 'II': 283.3333333333333, 'III': 166.66666666666666},
        sequence_states=('Y', 'Y', 'X', 'Y', 'Y', 'X', 'Y', 'Y', 'Y', 'X', 'Y'),
        sequence_info=('none', 'none', 'none', 'none', 'insider', 'insider', 'insider', 'insider', 'all', 'all', 'all'),
        imperfect=True,
        franc_to_usd=0.002,
        bingo_total=30,
        paper_clue_cards={5: '0100101010', 6: '0000000011', 7: '0100110100', 8: '0000010000', 9: '1110000011', 10: '1010000011', 11: '1111111001'},
        public_clue_periods=(11,),
        announce_no_info=True,
        dividends_constant_is_common_knowledge=False,
    ),
    2: dict(
        number=2,
        n_per_type=4,
        insiders_per_type=2,
        states=('X', 'Y'),
        prior={'X': 0.3333333333333333, 'Y': 0.6666666666666666},
        dividends={'I': {'X': 100, 'Y': 350}, 'II': {'X': 200, 'Y': 300}, 'III': {'X': 240, 'Y': 175}},
        prior_ev={'I': 266.66666666666663, 'II': 266.66666666666663, 'III': 196.66666666666666},
        sequence_states=('X', 'X', 'Y', 'Y', 'Y', 'Y', 'X', 'Y', 'X', 'Y', 'Y'),
        sequence_info=('none', 'none', 'none', 'none', 'all', 'all', 'insider', 'insider', 'insider', 'insider', 'insider'),
        imperfect=False,
        franc_to_usd=0.002,
        bingo_total=30,
        paper_clue_cards={},
        public_clue_periods=(),
        announce_no_info=True,
        dividends_constant_is_common_knowledge=True,
    ),
    3: dict(
        number=3,
        n_per_type=4,
        insiders_per_type=2,
        states=('X', 'Y'),
        prior={'X': 0.4, 'Y': 0.6},
        dividends={'I': {'X': 400, 'Y': 100}, 'II': {'X': 300, 'Y': 150}, 'III': {'X': 125, 'Y': 175}},
        prior_ev={'I': 220.0, 'II': 210.0, 'III': 155.0},
        sequence_states=('X', 'Y', 'Y', 'X', 'Y', 'Y', 'X', 'Y', 'X', 'Y', 'Y', 'X'),
        sequence_info=('none', 'none', 'insider', 'insider', 'insider', 'insider', 'insider', 'insider', 'insider', 'insider', 'all', 'all'),
        imperfect=False,
        franc_to_usd=0.003,
        bingo_total=40,
        paper_clue_cards={},
        public_clue_periods=(),
        announce_no_info=False,
        dividends_constant_is_common_knowledge=True,
    ),
    4: dict(
        number=4,
        n_per_type=4,
        insiders_per_type=2,
        states=('X', 'Y'),
        prior={'X': 0.4, 'Y': 0.6},
        dividends={'I': {'X': 375, 'Y': 100}, 'II': {'X': 275, 'Y': 150}, 'III': {'X': 100, 'Y': 175}},
        prior_ev={'I': 210.0, 'II': 200.0, 'III': 145.0},
        sequence_states=('X', 'Y', 'Y', 'X', 'Y', 'X', 'Y', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y'),
        sequence_info=('none', 'none', 'none', 'none', 'insider', 'insider', 'insider', 'insider', 'insider', 'insider', 'insider', 'insider', 'insider', 'none'),
        imperfect=False,
        franc_to_usd=0.003,
        bingo_total=40,
        paper_clue_cards={},
        public_clue_periods=(),
        announce_no_info=False,
        dividends_constant_is_common_knowledge=True,
    ),
    5: dict(
        number=5,
        n_per_type=4,
        insiders_per_type=2,
        states=('X', 'Y', 'Z'),
        prior={'X': 0.35, 'Y': 0.25, 'Z': 0.4},
        dividends={'I': {'X': 120, 'Y': 170, 'Z': 320}, 'II': {'X': 155, 'Y': 245, 'Z': 135}, 'III': {'X': 180, 'Y': 100, 'Z': 160}},
        prior_ev={'I': 212.5, 'II': 169.5, 'III': 152.0},
        sequence_states=('Z', 'X', 'Z', 'X', 'X', 'Y', 'Z', 'Z', 'Y', 'Y', 'X', 'Y', 'Z'),
        sequence_info=('none', 'none', 'none', 'insider', 'insider', 'insider', 'insider', 'insider', 'insider', 'insider', 'insider', 'insider', 'insider'),
        imperfect=False,
        franc_to_usd=0.003,
        bingo_total=20,
        paper_clue_cards={},
        public_clue_periods=(),
        announce_no_info=True,
        dividends_constant_is_common_knowledge=True,
    ),
    6: dict(
        number=6,
        n_per_type=4,
        insiders_per_type=2,
        states=('X', 'Y'),
        prior={'X': 0.6, 'Y': 0.4},
        dividends={'I': {'X': 300, 'Y': 100}, 'II': {'X': 230, 'Y': 130}, 'III': {'X': 225, 'Y': 140}},
        prior_ev={'I': 220.0, 'II': 190.0, 'III': 191.0},
        sequence_states=('X', 'X', 'X', 'Y', 'Y', 'X', 'Y', 'X', 'X', 'Y', 'X', 'Y'),
        sequence_info=('none', 'none', 'insider', 'insider', 'insider', 'insider', 'insider', 'insider', 'insider', 'insider', 'all', 'all'),
        imperfect=False,
        franc_to_usd=0.003,
        bingo_total=40,
        paper_clue_cards={},
        public_clue_periods=(),
        announce_no_info=False,
        dividends_constant_is_common_knowledge=True,
    ),
    7: dict(
        number=7,
        n_per_type=4,
        insiders_per_type=2,
        states=('X', 'Y'),
        prior={'X': 0.6, 'Y': 0.4},
        dividends={'I': {'X': 360, 'Y': 110}, 'II': {'X': 330, 'Y': 130}, 'III': {'X': 290, 'Y': 160}},
        prior_ev={'I': 260.0, 'II': 250.0, 'III': 238.0},
        sequence_states=('X', 'Y', 'Y', 'X', 'Y', 'X', 'Y', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y'),
        sequence_info=('none', 'none', 'none', 'none', 'insider', 'insider', 'insider', 'insider', 'insider', 'insider', 'insider', 'insider', 'insider', 'none'),
        imperfect=False,
        franc_to_usd=0.003,
        bingo_total=40,
        paper_clue_cards={},
        public_clue_periods=(),
        announce_no_info=False,
        dividends_constant_is_common_knowledge=True,
    ),
    8: dict(
        number=8,
        n_per_type=4,
        insiders_per_type=2,
        states=('X', 'Y'),
        prior={'X': 0.6, 'Y': 0.4},
        dividends={'I': {'X': 380, 'Y': 180}, 'II': {'X': 350, 'Y': 200}, 'III': {'X': 400, 'Y': 100}},
        prior_ev={'I': 300.0, 'II': 290.0, 'III': 280.0},
        sequence_states=('X', 'Y', 'Y', 'X', 'Y', 'X', 'Y', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y'),
        sequence_info=('none', 'none', 'none', 'none', 'insider', 'insider', 'insider', 'insider', 'insider', 'insider', 'insider', 'insider', 'insider', 'none'),
        imperfect=False,
        franc_to_usd=0.003,
        bingo_total=40,
        paper_clue_cards={},
        public_clue_periods=(),
        announce_no_info=False,
        dividends_constant_is_common_knowledge=True,
    ),
}


class _Market:
    """Attribute access over one market's parameter dict, so callers can write
    `MARKETS[3].dividends` exactly as they would against the engine repository."""

    def __init__(self, d):
        self.__dict__.update(d)

    @property
    def types(self):
        return ["I", "II", "III"]

    @property
    def n_periods(self):
        return len(self.sequence_states)

    @property
    def n_investors(self):
        return self.n_per_type * len(self.types)

    @property
    def supply(self):
        """Certificates outstanding: every investor is endowed with two."""
        return self.n_investors * INITIAL_CERTS

    @property
    def n_insiders(self):
        return self.insiders_per_type * len(self.types)

    def __repr__(self):
        return f"Market(number={self.number})"


MARKETS = {n: _Market(d) for n, d in _MARKETS.items()}
VBAR = {n: max(m.prior_ev.values()) for n, m in MARKETS.items()}


def re_price(market: int, state: str) -> float:
    """The fully revealing price in `state`: the highest dividend any type draws."""
    d = MARKETS[market].dividends
    return max(d[t][state] for t in d)


def pi_price(market: int, state: str) -> float:
    """The prior-information price: the highest valuation held by anyone who has not
    seen the clue, or the fully revealing price where that is higher."""
    return max(VBAR[market], re_price(market, state))


def informed_side(market: int, state: str) -> str:
    """Which side the informed profit on.  'seller' iff RE < vbar (Proposition 1)."""
    r, v = re_price(market, state), VBAR[market]
    return "seller" if r < v else ("buyer" if r > v else "")


# --------------------------------------------------------------- market 1's clue
# Market 1's clue is not a state letter but a ten-draw sample from one of two urns,
# so a clue leaves residual uncertainty and the fully revealing price depends on the
# card drawn.  URN[s] = (P(0 | s), P(1 | s)).  Both are verbatim from the engine.
URN = {"X": (Fraction(4, 5), Fraction(1, 5)),
       "Y": (Fraction(3, 5), Fraction(2, 5))}
CLUE_DRAWS = 10


def sample_posterior(sample: str, prior: dict) -> dict:
    """Posterior over states given a market-1 clue sample of '0'/'1' characters."""
    ones = sample.count("1")
    zeros = len(sample) - ones
    w = {s: Fraction(str(prior[s])) * URN[s][0] ** zeros * URN[s][1] ** ones
         for s in URN}
    tot = sum(w.values())
    return {s: float(v / tot) for s, v in w.items()}


def _posterior_from_card(self, card):
    """What a correctly-reasoning investor believes after seeing `card`."""
    if card is None:
        return dict(self.prior)
    if self.imperfect:
        return sample_posterior(card, self.prior)
    return {s: (1.0 if s == card else 0.0) for s in self.states}


_Market.posterior_from_card = _posterior_from_card


def _card_for(self, period, state):
    """What the clue card carries: the state, or market 1's ten-draw sample.

    'The clues of all insiders were identical' -- one card per period.
    """
    if not self.imperfect:
        return state
    card = self.paper_clue_cards.get(period)
    if card is None:
        raise ValueError(f"market {self.number} period {period}: no printed clue card")
    return card


def _theory_price(self, period, card=None):
    """The RE and PI price predictions for a period, in whole francs.

    RE  the market aggregates the information that is in it, so the price is the
        highest expected dividend any type holds conditional on the clue.
    PI  no aggregation: the price is the highest valuation anyone holds given their
        OWN information, which is RE where the informed value it most and the
        uninformed level otherwise.

    RE conditions on the CLUE, not on the state.  For markets 2-5 these coincide,
    the clue being a letter; market 1's clue is a ten-draw sample and there they
    differ, which is the point -- a market cannot reveal more than it knows.
    Rounded to whole francs, as the original's Table 3 prints them.
    """
    info = self.sequence_info[period - 1]
    state = self.sequence_states[period - 1]
    vbar = max(self.prior_ev.values())

    if info == "none":
        return {"RE": round(vbar), "PI": round(vbar)}

    post = self.posterior_from_card(
        card if card is not None else self.card_for(period, state))
    informed = max(sum(post[s] * self.dividends[t][s] for s in self.states)
                   for t in self.types)
    pi = informed if info == "all" else max(informed, vbar)
    return {"RE": round(informed), "PI": round(pi)}


_Market.card_for = _card_for
_Market.theory_price = _theory_price
