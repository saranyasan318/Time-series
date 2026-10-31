"""Exponential smoothing family implemented with numpy/scipy (no statsmodels needed).

Supports SES, Holt (optionally damped) and Holt-Winters (additive / multiplicative seasonality).
Parameters are fitted by minimising one-step-ahead SSE with bounded L-BFGS-B from several starts.
Forecast variance uses the standard analytic ETS formulas (Hyndman et al., class 1 / additive errors).
"""
import numpy as np
from scipy.optimize import minimize


def _filter(y, par, trend, damped, seasonal, m, init):
    """Run the recursions; returns (one-step fitted values, final level, trend, seasonal ring)."""
    n = len(y)
    a = par[0]
    i = 1
    b_ = par[i] if trend else 0.0
    i += 1 if trend else 0
    g = par[i] if seasonal else 0.0
    i += 1 if seasonal else 0
    phi = par[i] if damped else 1.0
    l, b, s = init["l"], init["b"], list(init["s"]) if seasonal else []
    fitted = np.empty(n)
    for t in range(n):
        st = s[t % m] if seasonal else None
        base = l + phi * b
        if seasonal == "mul":
            f = base * st
        elif seasonal == "add":
            f = base + st
        else:
            f = base
        fitted[t] = f
        yt = y[t]
        if seasonal == "mul":
            l_new = a * (yt / st) + (1 - a) * base
        elif seasonal == "add":
            l_new = a * (yt - st) + (1 - a) * base
        else:
            l_new = a * yt + (1 - a) * base
        if trend:
            b = b_ * (l_new - l) + (1 - b_) * phi * b
        if seasonal == "mul":
            s[t % m] = g * (yt / base if base != 0 else st) + (1 - g) * st
        elif seasonal == "add":
            s[t % m] = g * (yt - base) + (1 - g) * st
        l = l_new
    return fitted, l, b, s


def _init_states(y, trend, seasonal, m):
    if seasonal:
        l0 = float(np.mean(y[:m]))
        b0 = float((np.mean(y[m:2 * m]) - np.mean(y[:m])) / m) if trend else 0.0
        s0 = (y[:m] / l0 if seasonal == "mul" and l0 != 0 else y[:m] - l0)
        return {"l": l0, "b": b0, "s": np.asarray(s0, float)}
    k = min(len(y), 3)
    l0 = float(y[0])
    b0 = float((y[min(len(y) - 1, 3)] - y[0]) / min(len(y) - 1, 3)) if trend else 0.0
    return {"l": l0, "b": b0, "s": []}


class ETSResult:
    def __init__(self, y, trend, damped, seasonal, m):
        self.y, self.trend, self.damped, self.seasonal, self.m = np.asarray(y, float), trend, damped, seasonal, m
        self.params = None

    def fit(self):
        y, trend, damped, seasonal, m = self.y, self.trend, self.damped, self.seasonal, self.m
        if seasonal == "mul" and np.any(y <= 0):
            raise ValueError("Multiplicative seasonality requires strictly positive data")
        init = _init_states(y, trend, seasonal, m)
        bounds = [(1e-4, 0.999)]
        starts = [[0.3], [0.7], [0.05], [0.15]]
        if trend:
            bounds.append((1e-4, 0.5))
            for st, v in zip(starts, (0.05, 0.15, 0.01, 0.05)): st.append(v)
        if seasonal:
            bounds.append((1e-4, 0.999))
            for st, v in zip(starts, (0.1, 0.4, 0.4, 0.2)): st.append(v)
        if damped:
            bounds.append((0.8, 0.98))
            for st, v in zip(starts, (0.95, 0.9, 0.95, 0.85)): st.append(v)
        warm = min(len(y), m) if seasonal else 1

        def sse(par):
            f, *_ = _filter(y, par, trend, damped, seasonal, m, {**init, "s": list(init["s"]) if seasonal else []})
            e = y[warm:] - f[warm:] if seasonal else y - f
            v = float(np.sum(e * e))
            return v if np.isfinite(v) else 1e30

        best = None
        for x0 in starts:
            r = minimize(sse, x0, bounds=bounds, method="L-BFGS-B", options={"maxiter": 80})
            if best is None or r.fun < best.fun:
                best = r
        self.params = best.x
        fitted, l, b, s = _filter(y, self.params, trend, damped, seasonal, m,
                                  {**init, "s": list(init["s"]) if seasonal else []})
        self.fitted, self.level, self.slope, self.season_state = fitted, l, b, s
        self.resid = (y - fitted)[warm:] if seasonal else y - fitted
        self.sigma = float(np.std(self.resid, ddof=min(1, max(len(self.resid) - 1, 0)))) if len(self.resid) > 1 else 0.0
        return self

    # --------------------------------------------------------------
    def _unpack(self):
        p, i = self.params, 1
        beta = p[i] if self.trend else 0.0; i += 1 if self.trend else 0
        gamma = p[i] if self.seasonal else 0.0; i += 1 if self.seasonal else 0
        phi = p[i] if self.damped else 1.0
        return p[0], beta, gamma, phi

    def forecast(self, h):
        a, beta, gamma, phi = self._unpack()
        steps = np.arange(1, h + 1)
        if self.trend:
            tr = np.cumsum(phi ** steps) * self.slope if self.damped else steps * self.slope
        else:
            tr = np.zeros(h)
        base = self.level + tr
        if self.seasonal:
            n = len(self.y)
            seas = np.array([self.season_state[(n + k) % self.m] for k in range(h)])
            return base * seas if self.seasonal == "mul" else base + seas
        return base

    def std_path(self, h):
        """Analytic forecast standard deviation for horizons 1..h (additive-error approximation)."""
        a, beta, gamma, phi = self._unpack()
        c = np.zeros(h)  # c_j for j=1..h-1 (c[0] unused)
        for j in range(1, h):
            if self.trend:
                cj = a * (1 + beta * (np.sum(phi ** np.arange(1, j + 1)) if self.damped else j))
            else:
                cj = a
            if self.seasonal and j % self.m == 0:
                cj += gamma * (1 - a)
            c[j] = cj
        var = self.sigma ** 2 * (1 + np.concatenate([[0], np.cumsum(c[1:] ** 2)])[:h] if h > 1 else np.array([1.0]))
        return np.sqrt(np.maximum(var, 0))[:h]
