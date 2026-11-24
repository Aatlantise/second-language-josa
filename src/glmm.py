"""Random-intercept logistic regression (GLMM) fitted by Laplace approximation.

A small, dependency-light stand-in for lme4::glmer(y ~ X + (1 | group), family = binomial)
because R/lme4 is not available in this environment (see DECISIONS.md).  Model
    logit P(y_ij = 1) = X_ij beta + u_j,   u_j ~ N(0, sigma^2)
Marginal log-likelihood per group j (Laplace, = glmer nAGQ = 1):
    log f_j(u*) - u*^2 / (2 s2) - 0.5 log(s2 * H_j)
with u* the conditional mode and H_j = sum p(1-p) + 1/s2.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2, norm


class Laplace:
    def __init__(self, X, y, g):
        self.X = np.asarray(X, float)
        self.y = np.asarray(y, float)
        _, self.g = np.unique(g, return_inverse=True)
        self.J = self.g.max() + 1
        self.u = np.zeros(self.J)

    def nll(self, theta):
        beta, ls = theta[:-1], theta[-1]
        s2 = np.exp(2 * ls)
        eta0 = self.X @ beta
        u = self.u
        g, y, J = self.g, self.y, self.J
        for _ in range(50):
            p = expit(eta0 + u[g])
            grad = np.bincount(g, y - p, J) - u / s2
            hess = np.bincount(g, p * (1 - p), J) + 1 / s2
            step = grad / hess
            u = u + step
            if np.max(np.abs(step)) < 1e-9:
                break
        self.u = u
        eta = eta0 + u[g]
        p = expit(eta)
        hess = np.bincount(g, p * (1 - p), J) + 1 / s2
        logf = np.bincount(g, y * eta - np.logaddexp(0, eta), J)
        ll = logf - u ** 2 / (2 * s2) - 0.5 * ls * 2 - 0.5 * np.log(hess)
        return -ll.sum()


def _logit_start(X, y):
    b = np.zeros(X.shape[1])
    for _ in range(50):
        p = expit(X @ b)
        W = p * (1 - p) + 1e-9
        H = (X * W[:, None]).T @ X + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return b


def num_hessian(f, x, eps=1e-4):
    n = len(x)
    H = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = eps
        H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / eps ** 2
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = eps
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * eps ** 2)
    return H


def fit(X, y, g, names, se=True):
    X = np.asarray(X, float)
    model = Laplace(X, y, g)
    b0 = _logit_start(X, np.asarray(y, float))
    theta0 = np.append(b0, np.log(0.5))
    bounds = [(None, None)] * X.shape[1] + [(-6, 3)]
    res = minimize(model.nll, theta0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=500, maxfun=100000))
    # polish (L-BFGS with numeric gradients can stop early)
    res = minimize(model.nll, res.x, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=500, maxfun=100000, ftol=1e-12))
    out = dict(names=list(names), beta=res.x[:-1], sigma=float(np.exp(res.x[-1])), ll=-res.fun,
               success=bool(res.success), n=len(y), n_groups=model.J, k=X.shape[1] + 1)
    if se:
        free = res.x.copy()
        H = num_hessian(model.nll, free)
        # sigma may sit at its lower bound: drop that dimension from the covariance
        at_bound = free[-1] <= -5.99
        keep = np.arange(len(free) - 1) if at_bound else np.arange(len(free))
        try:
            cov = np.linalg.inv(H[np.ix_(keep, keep)])
            sd = np.sqrt(np.clip(np.diag(cov), 0, None))
        except np.linalg.LinAlgError:
            cov = np.full((len(keep), len(keep)), np.nan)
            sd = np.full(len(keep), np.nan)
        out["se"] = sd[: X.shape[1]]
        out["cov"] = cov[: X.shape[1], : X.shape[1]]
    return out


def lrt(full, reduced):
    stat = max(2 * (full["ll"] - reduced["ll"]), 0.0)
    df = full["k"] - reduced["k"]
    return stat, df, chi2.sf(stat, df)


def coef_table(fit_, level=0.95):
    z = norm.ppf(0.5 + level / 2)
    b, s = fit_["beta"], fit_["se"]
    return [dict(term=n, coef=bi, se=si, OR=np.exp(bi), OR_lo=np.exp(bi - z * si), OR_hi=np.exp(bi + z * si),
                 p=2 * norm.sf(abs(bi / si)) if si > 0 else np.nan) for n, bi, si in zip(fit_["names"], b, s)]
