import numpy as np
import pandas as pd


def build_transitions(vitals, periods, index_time, death_time=None, step_h=1,
                      window_days=28, c_death=0.0, c_reintub=0.0):
    """
    vitals : DataFrame, index = uur_start, kolommen = m_<concept_id> (gemiddelde per uur)
    periods: lijst (start, einde) van beademing, bij voorkeur al samengevoegd
    Actie a_t = beademd op t+1 (1) of niet (0); reward is de VFD-term (+ optionele shaping).
    """
    n = int(window_days * 24 / step_h)
    t = pd.date_range(index_time.floor("h"), periods=n + 1, freq=f"{step_h}h")
    end = t[0] + pd.Timedelta(days=window_days)

    # Toestand: sample-and-hold (zoals in het artikel), rest met mediaan
    X = vitals.reindex(vitals.index.union(t)).sort_index().ffill().reindex(t)
    X = X.fillna(vitals.median())

    on = np.zeros(n + 1, dtype=bool)
    for s, e in periods:
        on |= np.asarray((t >= s) & (t < e))
    alive = (np.ones(n + 1, dtype=bool) if death_time is None
             else np.asarray(t < death_time))

    # Extra toestandskenmerken: beademd ja/nee en aantal intubaties tot nu toe
    starts = np.r_[on[0], (~on[:-1]) & on[1:]]
    n_intub = np.cumsum(starts)
    S = np.column_stack([X.to_numpy(dtype=float), on, n_intub])

    cur_alive, nxt_alive, nxt_on = alive[:-1], alive[1:], on[1:]
    in_win = np.asarray(t[1:] < end)

    r = (step_h / 24 / window_days) * (nxt_alive & ~nxt_on & in_win)
    reint = ~on[:-1] & nxt_on & nxt_alive
    died = cur_alive & ~nxt_alive
    r = r + c_reintub * reint + c_death * died

    done = died | (np.arange(n) == n - 1)
    keep = cur_alive
    return {
        "s": S[:-1][keep], "a": nxt_on.astype(int)[keep], "r": r[keep],
        "s2": S[1:][keep], "done": done[keep],
    }


# ---- Synthetisch voorbeeld ----
d0 = pd.Timestamp("2020-01-01")
D = lambda x: d0 + pd.Timedelta(days=x)
idx = pd.date_range(d0, periods=28 * 24, freq="h")
rng = np.random.default_rng(0)
vitals = pd.DataFrame(rng.normal(size=(len(idx), 3)), index=idx,
                      columns=["m_a", "m_b", "m_c"])

periods = [(D(0), D(5)), (D(6), D(8))]
tr = build_transitions(vitals, periods, D(0))
print("transities:", len(tr["r"]), "| state-dim:", tr["s"].shape[1])
print("VFD-28 uit rewards:", round(tr["r"].sum() * 28, 2))      # verwacht 21