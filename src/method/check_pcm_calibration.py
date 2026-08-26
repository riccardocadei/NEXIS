"""Size and power of the modified PCM test against the linear and GCM tests.

Checks:
  (a) type-I calibration under H0,
  (b) power on a linear (GCM-friendly) alternative,
  (c) power on a U-shaped alternative with zero conditional covariance (GCM-blind).
"""
import sys, time
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from method.nexis import (
    conditional_interaction_pvalues,
    conditional_interaction_pvalues_gcm,
    conditional_interaction_pvalues_pcm,
)

RNG = np.random.default_rng(0)


def draw(n, m, kind, eta, seed):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, m))
    Z[:, 1] = 0.6 * Z[:, 0] + 0.8 * rng.normal(size=n)   # correlated companion
    T = rng.binomial(1, 0.5, size=n).astype(float)
    if kind == "null":
        tau = np.full(n, 0.5)
    elif kind == "linear":
        tau = 0.5 + eta * Z[:, 0]
    elif kind == "ushape":
        tau = 0.5 + eta * (Z[:, 0] ** 2 - 1.0)           # E[tau*Z0] = 0 exactly
    Y = 0.3 * Z[:, 0] - 0.2 * Z[:, 2] + tau * T + rng.normal(size=n)
    return Y, T, Z


def run(kind, eta, n=600, m=40, reps=300, S=(2, 3)):
    out = {k: [] for k in ["linear", "gcm-q", "gcm-lgbm", "pcm-q", "pcm-lgbm"]}
    times = {k: 0.0 for k in out}
    for r in range(reps):
        Y, T, Z = draw(n, m, kind, eta, 1000 + r)
        S_ = list(S)
        for name, fn in [
            ("linear",   lambda: conditional_interaction_pvalues(y=Y, t=T, z=Z, S=S_, candidates=[0])),
            ("gcm-q",    lambda: conditional_interaction_pvalues_gcm(y=Y, t=T, z=Z, S=S_, candidates=[0], nuisance="poly2", n_splits=3)),
            ("gcm-lgbm", lambda: conditional_interaction_pvalues_gcm(y=Y, t=T, z=Z, S=S_, candidates=[0], nuisance="lgbm", n_splits=3)),
            ("pcm-q",    lambda: conditional_interaction_pvalues_pcm(y=Y, t=T, z=Z, S=S_, candidates=[0], nuisance="poly2", n_splits=3, projection="poly")),
            # nuisance="poly2" matches the shipped test="PCM: lgbm" alias: the ML
            # projection is what buys power, but LightGBM *nuisances* on the half-split
            # are not calibrated at these n (size 0.080 vs 0.053).
            ("pcm-lgbm", lambda: conditional_interaction_pvalues_pcm(y=Y, t=T, z=Z, S=S_, candidates=[0], nuisance="poly2", n_splits=3, projection="lgbm", screen_top=4)),
        ]:
            t0 = time.perf_counter()
            out[name].append(float(fn()[0]))
            times[name] += time.perf_counter() - t0
    print(f"\n=== {kind}  eta={eta}  n={n}  reps={reps} ===")
    print(f"{'test':10s} {'rej@0.05':>9s} {'rej@0.10':>9s} {'med p':>8s} {'s/rep':>7s}")
    for k, v in out.items():
        a = np.array(v)
        print(f"{k:10s} {(a<=0.05).mean():9.3f} {(a<=0.10).mean():9.3f} "
              f"{np.median(a):8.3f} {times[k]/reps:7.3f}")


if __name__ == "__main__":
    run("null",   0.0, reps=1000)
    run("linear", 0.4, reps=200)
    run("ushape", 0.5, reps=200)
