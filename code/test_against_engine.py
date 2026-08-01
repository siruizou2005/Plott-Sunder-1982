"""Check the hand-ported methods against the simulation engine, where it is available.

The data block of ps1982_params.py is a machine-emitted copy of the engine's own
objects and can be diffed against it.  The methods are NOT: they were re-implemented
by hand.  This script is how that re-implementation is verified -- it imports both
modules and compares them value-for-value on every published period.

    python3 code/test_against_engine.py [path/to/Plott-Sunder-1982-LLM]

The engine is not needed to RUN the package; it is needed only to run this check.
Skips with a clear message when the engine is not on disk.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ps1982_params as P

# The engine lives in a separate repository, github.com/siruizou2005/Plott-Sunder-1982-LLM.
# With no argument, look for a checkout sitting beside this one.
SIBLING_NAMES = ["Plott-Sunder-1982-LLM", "19-Plott-Sunder-1982-LLM"]


def default_engine() -> Path:
    root = HERE.parent
    for beside in [root.parent, root.parent.parent]:
        for name in SIBLING_NAMES:
            if (beside / name / "ps1982" / "markets.py").exists():
                return beside / name
    return root.parent / SIBLING_NAMES[0]


def main(engine_path: Path) -> int:
    if not (engine_path / "ps1982" / "markets.py").exists():
        print(f"SKIP: no engine at {engine_path}\n"
              "      Pass its path as an argument to run this check.")
        return 0
    sys.path.insert(0, str(engine_path))
    from ps1982.markets import (MARKETS as EM, INITIAL_CASH, INITIAL_CERTS,
                                FIXED_COST)

    fails = []

    def check(label, mine, theirs):
        if mine != theirs:
            fails.append(f"{label}: package {mine!r} vs engine {theirs!r}")

    check("INITIAL_CASH", P.INITIAL_CASH, INITIAL_CASH)
    check("INITIAL_CERTS", P.INITIAL_CERTS, INITIAL_CERTS)
    check("FIXED_COST", P.FIXED_COST, FIXED_COST)
    check("market numbers", sorted(P.MARKETS), sorted(EM))

    n = 0
    for num in sorted(EM):
        mine, theirs = P.MARKETS[num], EM[num]
        for field in ["states", "prior", "dividends", "prior_ev", "sequence_states",
                      "sequence_info", "imperfect", "paper_clue_cards", "n_per_type",
                      "insiders_per_type", "bingo_total", "franc_to_usd"]:
            check(f"m{num}.{field}", getattr(mine, field), getattr(theirs, field))
        for prop in ["types", "n_periods"]:
            check(f"m{num}.{prop}", getattr(mine, prop), getattr(theirs, prop))

        for period in range(1, theirs.n_periods + 1):
            state = theirs.sequence_states[period - 1]
            info = theirs.sequence_info[period - 1]

            # card_for is reached only where somebody holds a clue.  In market 1's
            # no-information periods there is no printed card and BOTH implementations
            # raise, which is itself the behaviour to check.
            if info == "none" and theirs.imperfect:
                for label, m in [("package", mine), ("engine", theirs)]:
                    try:
                        m.card_for(period, state)
                        fails.append(f"m{num} p{period} card_for: {label} returned "
                                     "a card where none is printed")
                    except (ValueError, KeyError):
                        pass
            else:
                check(f"m{num} p{period} card_for",
                      mine.card_for(period, state), theirs.card_for(period, state))
                check(f"m{num} p{period} posterior",
                      mine.posterior_from_card(mine.card_for(period, state)),
                      theirs.posterior_from_card(theirs.card_for(period, state)))
            check(f"m{num} p{period} theory_price",
                  mine.theory_price(period), theirs.theory_price(period))
            n += 1

            # re_price/pi_price are exact where theory_price rounds, so compare rounded
            if theirs.sequence_info[period - 1] == "insider" and not theirs.imperfect:
                tp = theirs.theory_price(period)
                check(f"m{num} p{period} re_price (rounded)",
                      round(P.re_price(num, state)), tp["RE"])
                check(f"m{num} p{period} pi_price (rounded)",
                      round(P.pi_price(num, state)), tp["PI"])
                side = "buyer" if tp["RE"] > round(P.VBAR[num]) else \
                       "seller" if tp["RE"] < round(P.VBAR[num]) else ""
                check(f"m{num} p{period} informed_side",
                      P.informed_side(num, state), side)

    if fails:
        print(f"FAIL: {len(fails)} mismatch(es) against the engine")
        for f in fails[:20]:
            print("  " + f)
        return 1
    print(f"PASS: package matches the engine on {n} periods across "
          f"{len(EM)} markets -- constants, every parameter field, and "
          f"card_for / posterior_from_card / theory_price / re_price / pi_price / "
          f"informed_side")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else default_engine()))
