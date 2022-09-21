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
    omegas: torch.Tensor, tau: torch.Tensor, W: torch.Tensor, K: torch.Tensor, **kwargs
):
    """Compute the approximated spectral density through the Bernstein polynomial prior."""
    ### reshape tensors
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

    # calculate bernstein polynomial approximation for the spd
    # f.shape == (nu)
    return tau * torch.sum(W * bernstein_betas, dim=-1)


def model(
    periodogram: torch.Tensor, omegas: torch.Tensor, data: torch.Tensor, **kwargs
):
    """The model, contaiting priors and the model likelihood."""
    ### obtain constants or set them to default values
    M = kwargs.get("M", 1)

    nu = len(periodogram)

    ### priors
    # tau ~ Exp(1 / S_n^2)
    # tau = 1 / pyro.sample("tau", dist.InverseGamma(2, torch.var(data, unbiased=True)))
    tau = pyro.sample(
        "tau", dist.Gamma(torch.var(data, unbiased=True) / (2 * np.pi), torch.ones(1))
    )

    K = torch.tensor([150])

    # W ~ Dirichlet(1, 1, ...)
    W = pyro.sample("W", dist.Dirichlet(M * torch.ones([150])))

    ### whittle likelihood
    f = compute_f(omegas=omegas, tau=tau, K=K, W=W)

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
    L = kwargs.get("L", 10)
    M = kwargs.get("M", 1)

    ### define the variational parameters
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

    ### define the variational distributions
    K = torch.tensor([150])

    ps_W = kwargs.get(
        "ps_W",
        pyro.param(
            "ps_W",
            lambda: dist.Uniform(0, 1).sample([150]),
            constraint=constraints.positive,
        ),
    )

    W = pyro.sample("W", dist.Dirichlet(ps_W))
    # tau ~ Gamma(alpha_tau, beta_tau)
    # tau = 1 / pyro.sample("tau", dist.InverseGamma(alpha_tau, beta_tau))
    tau = pyro.sample("tau", dist.Gamma(alpha_tau, beta_tau))

    return {"tau": tau, "K": K, "W": W}
