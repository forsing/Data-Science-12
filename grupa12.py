"""Grupa 12 — informacija / kompleksnost (Loto 7/39)."""

from __future__ import annotations

import csv
from collections import Counter
from math import log2
from pathlib import Path

import numpy as np

SEED = 39
FRONT_N = 39
FRONT_SELECT = 7
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "loto7_4648_k55.csv"

np.random.seed(SEED)


def load_draws(csv_path: Path = CSV_PATH) -> np.ndarray:
    draws = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < FRONT_SELECT:
                continue
            try:
                draw = sorted(int(x.strip()) for x in row[:FRONT_SELECT])
            except ValueError:
                continue
            if len(draw) == FRONT_SELECT and all(1 <= x <= FRONT_N for x in draw):
                if len(set(draw)) == FRONT_SELECT:
                    draws.append(draw)
    if not draws:
        raise ValueError(f"Nema validnih kola u {csv_path}")
    return np.array(draws, dtype=int)


def presence_matrix(draws: np.ndarray) -> np.ndarray:
    x = np.zeros((len(draws), FRONT_N), dtype=float)
    for i, draw in enumerate(draws):
        for n in draw.tolist():
            x[i, n - 1] = 1.0
    return x


def shannon_entropy_numbers(draws: np.ndarray) -> dict:
    """Shannon H na empirijskoj raspodeli brojeva 1..39 (ceo CSV)."""
    cnt = Counter(draws.reshape(-1).tolist())
    total = sum(cnt.values())
    p = np.array([cnt.get(n, 0) / total for n in range(1, FRONT_N + 1)])
    h = float(-np.sum(p[p > 0] * np.log2(p[p > 0])))
    h_max = log2(FRONT_N)
    return {"H": h, "H_max": h_max, "H_norm": h / h_max, "p": p}


def joint_conditional_entropy(draws: np.ndarray, top_k: int = 10) -> dict:
    """
    Joint H(X,Y) i conditional H(Y|X) za presence parove.
    Top parovi po I(X;Y)=H(X)+H(Y)-H(X,Y).
    """
    x = presence_matrix(draws)
    rows = []
    for i in range(FRONT_N):
        for j in range(i + 1, FRONT_N):
            a, b = x[:, i], x[:, j]
            # marginal
            def h_bin(v):
                p1 = float(v.mean())
                p0 = 1.0 - p1
                s = 0.0
                for p in (p0, p1):
                    if p > 0:
                        s -= p * log2(p)
                return s

            hx, hy = h_bin(a), h_bin(b)
            hxy = 0.0
            for aa in (0.0, 1.0):
                for bb in (0.0, 1.0):
                    p = float(np.mean((a == aa) & (b == bb)))
                    if p > 0:
                        hxy -= p * log2(p)
            mi = hx + hy - hxy
            h_ygx = hxy - hx
            rows.append((i + 1, j + 1, hxy, h_ygx, mi))
    rows.sort(key=lambda t: (-t[4], t[0], t[1]))
    return {"top_MI_pairs": rows[:top_k]}


def kl_js_vs_uniform(draws: np.ndarray) -> dict:
    """KL i JS divergence empirijske raspodele brojeva vs uniform."""
    ent = shannon_entropy_numbers(draws)
    p = ent["p"]
    q = np.ones(FRONT_N) / FRONT_N
    kl = float(np.sum(p[p > 0] * np.log2(p[p > 0] / q[p > 0])))
    m = 0.5 * (p + q)
    js = 0.0
    for dist in (p, q):
        mask = dist > 0
        js += 0.5 * float(np.sum(dist[mask] * np.log2(dist[mask] / m[mask])))
    # cross-entropy H(p,q)
    ce = float(-np.sum(p[p > 0] * np.log2(q[p > 0])))
    return {"KL_p_uniform": kl, "JS_p_uniform": js, "cross_entropy": ce, "H_p": ent["H"]}


def lz_complexity(seq: list[int]) -> int:
    """Lempel–Ziv complexity (LZ76) — broj fraza."""
    n = len(seq)
    if n == 0:
        return 0
    i, c = 0, 0
    while i < n:
        c += 1
        max_match = 0
        for j in range(0, i):
            k = 0
            while i + k < n and seq[j + k] == seq[i + k]:
                k += 1
                if j + k >= i:
                    break
            if k > max_match:
                max_match = k
        i += max_match + 1
    return c


def lz_on_draws(draws: np.ndarray) -> dict:
    """LZ na nizu kola (kao tokeni) i na binovanoj sumi (ceo CSV)."""
    # token po kolu = lex-ish hash of sorted tuple
    tokens = [hash(tuple(d.tolist())) & 0xFFFFFFFF for d in draws]
    s = draws.sum(axis=1).astype(float)
    bins = np.digitize(s, np.linspace(s.min(), s.max() + 1e-9, 12)).tolist()
    return {
        "LZ_draw_tokens": lz_complexity(tokens),
        "LZ_sum_bins": lz_complexity(bins),
        "n_draws": len(draws),
        "n_unique_draws": len(set(tokens)),
    }


def approx_entropy(series: np.ndarray, m: int = 2, r: float | None = None) -> float:
    """Approximate Entropy (Pincus)."""
    x = np.asarray(series, dtype=float)
    n = len(x)
    if r is None:
        r = 0.2 * float(x.std())
    if r <= 0 or n < m + 2:
        return 0.0

    def phi(mm: int) -> float:
        patterns = np.array([x[i : i + mm] for i in range(n - mm + 1)])
        c = []
        for i in range(len(patterns)):
            d = np.max(np.abs(patterns - patterns[i]), axis=1)
            c.append(np.mean(d <= r))
        return float(np.mean(np.log(np.array(c) + 1e-300)))

    return phi(m) - phi(m + 1)


def sample_entropy(series: np.ndarray, m: int = 2, r: float | None = None) -> float:
    """Sample Entropy (Richman & Moorman): -ln(A/B)."""
    x = np.asarray(series, dtype=float)
    n = len(x)
    if r is None:
        r = 0.2 * float(x.std())
    if r <= 0 or n < m + 2:
        return 0.0

    def count(mm: int) -> float:
        patterns = np.array([x[i : i + mm] for i in range(n - mm)])
        B = 0.0
        for i in range(len(patterns)):
            d = np.max(np.abs(patterns[i + 1 :] - patterns[i]), axis=1)
            B += float(np.sum(d <= r))
        return B

    B = count(m)
    A = count(m + 1)
    if A <= 0 or B <= 0:
        return 0.0
    return float(-np.log(A / B))


def permutation_entropy(series: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """Permutation entropy (Bandt–Pompe)."""
    x = np.asarray(series, dtype=float)
    n = len(x)
    perms = Counter()
    total = 0
    for i in range(n - (order - 1) * delay):
        window = x[i : i + order * delay : delay]
        # rank pattern
        ranks = tuple(np.argsort(np.argsort(window)))
        perms[ranks] += 1
        total += 1
    if total == 0:
        return 0.0
    h = 0.0
    for c in perms.values():
        p = c / total
        h -= p * log2(p)
    fac = 1
    for k in range(2, order + 1):
        fac *= k
    h_max = log2(fac)
    return float(h / h_max) if h_max > 0 else float(h)


def number_entropy_profile(draws: np.ndarray) -> list[tuple]:
    """
    Za svaki broj: entropija gap-procesa (složenost pojavljivanja).
    Niža H gapova → pravilniji ritam.
    """
    x = presence_matrix(draws)
    rows = []
    for n in range(FRONT_N):
        idx = np.where(x[:, n] == 1.0)[0]
        if len(idx) < 3:
            rows.append((n + 1, 0.0, 0))
            continue
        gaps = np.diff(idx).astype(float)
        # discrete entropy of gaps
        cnt = Counter(gaps.tolist())
        tot = sum(cnt.values())
        h = 0.0
        for c in cnt.values():
            p = c / tot
            h -= p * log2(p)
        rows.append((n + 1, float(h), int(len(idx))))
    rows.sort(key=lambda t: (t[1], -t[2], t[0]))
    return rows


def learn_next_rule(draws: np.ndarray) -> dict:
    """
    Pravilo next iz grupe 12:
    skor = niža gap-entropija (pravilniji ritam) + frekvencija
         + boost ako je broj „informativan“ vs uniform (p * log p/q).
    """
    ent = shannon_entropy_numbers(draws)
    p = ent["p"]
    q = 1.0 / FRONT_N
    # pointwise KL contribution
    pkl = p * np.log2((p + 1e-12) / q)

    gap_h = {n: h for n, h, _ in number_entropy_profile(draws)}
    max_gh = max(gap_h.values()) if gap_h else 1.0
    freq = Counter(draws.reshape(-1).tolist())
    max_f = max(freq.values()) if freq else 1

    # overdue from gaps
    x = presence_matrix(draws)
    t_end = len(draws) - 1
    overdue = {}
    for n in range(FRONT_N):
        idx = np.where(x[:, n] == 1.0)[0]
        last = int(idx[-1]) if len(idx) else -1
        mean_g = float(np.diff(idx).mean()) if len(idx) >= 2 else float(len(draws))
        cur = t_end - last if last >= 0 else len(draws)
        overdue[n + 1] = cur / mean_g if mean_g > 0 else 0.0
    max_ov = max(overdue.values()) if overdue else 1.0

    number_score = {}
    for y in range(1, FRONT_N + 1):
        # prefer lower gap entropy (more structured) + overdue + freq + |pkl|
        number_score[y] = (
            1.0 * (1.0 - gap_h[y] / max_gh)
            + 0.8 * (overdue[y] / max_ov)
            + 0.4 * (freq.get(y, 0) / max_f)
            + 0.3 * float(abs(pkl[y - 1]) / (np.abs(pkl).max() + 1e-12))
        )

    return {
        "number_score": number_score,
        "last_draw": [int(v) for v in draws[-1].tolist()],
        "target_sum": float(draws.sum(axis=1).mean()),
        "H_numbers": ent["H"],
        "KL_uniform": float(np.sum(pkl)),
    }


def _combo_fit(combo: list[int], rule: dict) -> float:
    score = sum(rule["number_score"][x] for x in combo)
    score -= 0.015 * abs(sum(combo) - rule["target_sum"])
    return score


def predict_next_from_rule(draws: np.ndarray, rule: dict | None = None) -> list[int]:
    if rule is None:
        rule = learn_next_rule(draws)
    ranked = sorted(rule["number_score"], key=lambda n: (-rule["number_score"][n], n))
    best = None
    best_fit = -1e18
    for start in range(0, min(20, FRONT_N - FRONT_SELECT + 1)):
        base = sorted(ranked[start : start + FRONT_SELECT])
        for repl in ranked[:28]:
            cand = sorted(set(base[1:] + [repl]))
            if len(cand) != FRONT_SELECT:
                continue
            fit = _combo_fit(cand, rule)
            if fit > best_fit:
                best_fit = fit
                best = cand
    return best if best is not None else sorted(ranked[:FRONT_SELECT])


def run_grupa12(csv_path: Path = CSV_PATH) -> None:
    draws = load_draws(csv_path)
    print(f"CSV: {csv_path.name}")
    print(f"Kola: {len(draws)} | seed={SEED} | 7/39 | grupa12")
    print()

    print("=== Shannon entropy brojeva ===")
    ent = shannon_entropy_numbers(draws)
    print({k: ent[k] for k in ("H", "H_max", "H_norm")})
    print()

    print("=== KL / JS / cross-entropy vs uniform ===")
    print(kl_js_vs_uniform(draws))
    print()

    print("=== joint/conditional — top MI pairs ===")
    print(joint_conditional_entropy(draws)["top_MI_pairs"])
    print()

    print("=== LZ complexity ===")
    print(lz_on_draws(draws))
    print()

    sums = draws.sum(axis=1).astype(float)
    print("=== ApEn / SampEn / PermEn (suma) ===")
    print(
        {
            "approx_entropy": approx_entropy(sums),
            "sample_entropy": sample_entropy(sums),
            "permutation_entropy_norm": permutation_entropy(sums, order=3),
        }
    )
    print()

    print("=== gap-entropy po broju (najpravilniji ritam, top10) ===")
    print(number_entropy_profile(draws)[:10])
    print()

    print("=== pravilo → next (grupa 12) ===")
    rule = learn_next_rule(draws)
    combo = predict_next_from_rule(draws, rule)
    print(
        "rule:",
        {
            "last_draw": rule["last_draw"],
            "target_sum": round(rule["target_sum"], 2),
            "H_numbers": round(rule["H_numbers"], 4),
            "KL_uniform": round(rule["KL_uniform"], 6),
        },
    )
    print("next:", combo)


if __name__ == "__main__":
    run_grupa12()


"""
12. Informacija / kompleksnost
Shannon entropy, conditional entropy, joint entropy, KL divergence, JS divergence,
cross-entropy, compression complexity (LZ), approximate entropy, sample entropy,
permutation entropy
"""



"""
CSV: loto7_4648_k55.csv
Kola: 4648 | seed=39 | 7/39 | grupa12

=== Shannon entropy brojeva ===
{'H': 5.284368713970772, 'H_max': 5.285402218862249, 'H_norm': 0.9998044605029701}

=== KL / JS / cross-entropy vs uniform ===
{'KL_p_uniform': 0.0010335048914770944, 'JS_p_uniform': 0.0002582611965607843, 'cross_entropy': 5.2854022188622505, 'H_p': 5.284368713970772}

=== joint/conditional — top MI pairs ===
[(12, 22, 1.354202496061096, 0.6848979698227494, 0.0028188289962081114), (21, 33, 1.3619998918171456, 0.6864810927806663, 0.0026236435452091644), (8, 39, 1.3993760707556777, 0.6837749073795901, 0.0025499428926030188), (18, 27, 1.329681838431818, 0.6555902191317114, 0.002527810143490239), (3, 39, 1.361740385083137, 0.6838520784966851, 0.002472771775508198), (4, 30, 1.3244817138679816, 0.6546964054303016, 0.0024371449489861696), (9, 39, 1.3684063643832503, 0.6839437038869125, 0.0023811463852807524), (2, 26, 1.368183711324708, 0.6936159081692612, 0.0023687286070095404), (32, 34, 1.3868712443045086, 0.6959221452117651, 0.002333799441238016), (24, 37, 1.3730553718406087, 0.689994061816368, 0.002333657920755572)]

=== LZ complexity ===
{'LZ_draw_tokens': 4647, 'LZ_sum_bins': 1011, 'n_draws': 4648, 'n_unique_draws': 4647}

=== ApEn / SampEn / PermEn (suma) ===
{'approx_entropy': 2.110737378868958, 'sample_entropy': 2.1963711529191556, 'permutation_entropy_norm': 0.999762935576865}

=== gap-entropy po broju (najpravilniji ritam, top10) ===
[(8, 3.5903796508605272, 915), (23, 3.6233064764949385, 908), (34, 3.675334478748287, 876), (26, 3.6788311437290613, 871), (11, 3.693674638485278, 862), (37, 3.69796806601526, 863), (32, 3.7070863810886774, 860), (29, 3.709176817067929, 853), (33, 3.71243423382204, 856), (22, 3.7149897699790877, 853)]

=== pravilo → next (grupa 12) ===
rule: {'last_draw': [3, 7, 12, 13, 18, 24, 29], 'target_sum': 140.43, 'H_numbers': 5.2844, 'KL_uniform': 0.001034}
next: [2, x, 21, y, 26, z, 39]
"""
