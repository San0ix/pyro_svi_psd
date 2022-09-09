import numpy as np
import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
import torch
import torch.nn.functional as F
from torch.distributions import constraints


def stick_breaking_weights(stick_breaking_fractions: torch.Tensor):
    """Compute the stick-breaking weights from the given fractions."""
    stick_breaking_fractions1m_cumprod = (1 - stick_breaking_fractions).cumprod(-1)
    return F.pad(stick_breaking_fractions, (0, 1), value=1) * F.pad(
        stick_breaking_fractions1m_cumprod, (1, 0), value=1
    )


def compute_f(
    omegas: torch.Tensor,
    tau: torch.Tensor,
    V: torch.Tensor,
    Z: torch.Tensor,
    K: torch.Tensor,
    **kwargs
):
    """Compute the approximated spectral density through the Bernstein polynomial prior."""
    ### reshape tensors
    # W.shape == (L+1, 1)
    # use dimension -1 when multiplying the values of
    # delta_{((j - 1) / K, j / K]}(Z) with W, for each value of j
    W = torch.unsqueeze(stick_breaking_weights(V), dim=1)

    # Z.shape == (L+1, 1)
    # use dimension -1 when computing
    # delta_{((j - 1) / K, j / K]}(Z), for each value of j
    Z = torch.unsqueeze(Z, dim=1)

    # omegas.shape == (nu, 1)
    # use dimension -1 when computing the individual values for the bernstein betas
    # for each j
    omegas = torch.unsqueeze(omegas, dim=1)

    ### compute the modeled spd
    # compute the beta densities in the Bernstein polynomial
    # js.shape == (K)
    js = torch.arange(1, K.item() + 1)
    # bernstein_beta.shape == (nu, K)
    bernstein_betas = torch.exp(
        dist.Beta(js, torch.flip(js, dims=(-1,))).log_prob(omegas)
    )

    # compute the weights in the Bernstein polynomial
    # torch.logical_and(Z >= (js - 1) / K, Z < js / K).shape == (L+1, K)
    # bernstein_weights.shape == (K)
    bernstein_weights = torch.sum(
        W * torch.logical_and(Z >= (js - 1) / K, Z < js / K), dim=-2
    )

    # calculate bernstein polynomial approximation for the spd
    # f.shape == (nu)
    return tau * torch.sum(bernstein_weights * bernstein_betas, dim=-1)


def model(
    periodogram: torch.Tensor, omegas: torch.Tensor, data: torch.Tensor, **kwargs
):
    """The model, contaiting priors and the model likelihood."""
    ### obtain constants or set them to default values
    L = kwargs.get("L", 10)
    M = kwargs.get("M", 1)
    MIN_K = kwargs.get("MIN_K", 10)
    MAX_K = kwargs.get("MAX_K", 200)
    STEP_K = kwargs.get("STEP_K", 1)

    LEN_K = int(np.ceil((MAX_K - MIN_K) / STEP_K)) + 1

    nu = len(periodogram)

    ### priors
    # tau ~ Exp(1 / S_n^2)
    tau = pyro.sample("tau", dist.Gamma(torch.var(data, unbiased=True), torch.ones(1)))

    # p_K is proportional to Exp
    K = pyro.sample("K", dist.Exponential(1 / LEN_K)).long() * STEP_K + MIN_K

    # V iid Beta(1, M)
    with pyro.plate("V_plate", L):
        V = pyro.sample("V", dist.Beta(1, M))

    # Z iid Uniform(0, 1)
    with pyro.plate("Z_plate", L + 1):
        Z = pyro.sample("Z", dist.Uniform(0, 1))

    ### whittle likelihood
    f = compute_f(omegas=omegas, tau=tau, K=K, Z=Z, V=V)

    with pyro.plate("obs_plate", nu):
        pyro.sample(
            "obs",
            dist.Exponential(1 / f),
            obs=periodogram,
        )

    return f


def guide(
    periodogram: torch.Tensor, omegas: torch.Tensor, data: torch.Tensor, **kwargs
):
    """The variational family used to approximate the posterior."""
    ### obtain constants or set them to default values
    L: int = kwargs.get("L", 10)
    M: int = kwargs.get("M", 1)
    MIN_K: int = kwargs.get("MIN_K", 10)
    MAX_K: int = kwargs.get("MAX_K", 200)
    STEP_K: int = kwargs.get("STEP_K", 1)

    LEN_K = int(np.ceil((MAX_K - MIN_K) / STEP_K))

    ### define the variational parameters
    alpha_V = kwargs.get(
        "alpha_V",
        pyro.param(
            "alpha_V",
            lambda: dist.Uniform(0, 2).sample([L]),
            constraint=constraints.positive,
        ),
    )
    beta_V = kwargs.get(
        "beta_V",
        pyro.param(
            "beta_V",
            lambda: dist.Uniform(0, 2).sample([L]),
            constraint=constraints.positive,
        ),
    )

    alpha_Z = kwargs.get(
        "alpha_Z",
        pyro.param(
            "alpha_Z",
            lambda: dist.Uniform(0, 2).sample([L + 1]),
            constraint=constraints.positive,
        ),
    )
    beta_Z = kwargs.get(
        "beta_Z",
        pyro.param(
            "beta_Z",
            lambda: dist.Uniform(0, 2).sample([L + 1]),
            constraint=constraints.positive,
        ),
    )

    alpha_tau = kwargs.get(
        "alpha_tau",
        pyro.param(
            "alpha_tau",
            lambda: dist.Uniform(0, 2).sample([1]),
            constraint=constraints.greater_than(0),
        ),
    )
    beta_tau = kwargs.get(
        "beta_tau",
        pyro.param(
            "beta_tau",
            lambda: dist.Uniform(0, 2).sample([1]),
            constraint=constraints.positive,
        ),
    )

    ps_K = kwargs.get(
        "ps_K",
        pyro.param(
            "ps_K",
            # torch.ones(LEN_K) / (1.0 * LEN_K),
            lambda: dist.Dirichlet(torch.ones(LEN_K)).sample([1]),
            constraint=constraints.simplex,
        ),
    )

    ### define the variational distributions
    # K ~ Categorical
    K = pyro.sample("K", dist.Categorical(ps_K)) * STEP_K + MIN_K

    # V_i ind. ~ Beta(1, beta_{V_i})
    with pyro.plate("V_plate", L):
        V = pyro.sample("V", dist.Beta(alpha_V, beta_V))

    # Z_i ind. ~ Beta(1, beta_{Z_i})
    with pyro.plate("Z_plate", L + 1):
        Z = pyro.sample("Z", dist.Beta(alpha_Z, beta_Z))

    # tau ~ Gamma(alpha_tau, beta_tau)
    tau = pyro.sample("tau", dist.Gamma(alpha_tau, beta_tau))

    return {"tau": tau, "K": K, "V": V, "Z": Z}
