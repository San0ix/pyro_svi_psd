import time
from collections import defaultdict
from functools import partial

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
import torch

from pyro_specification import compute_f, guide, model


def my_loss(
    model,
    guide,
    *args,
    loss_fun,
    max_loss=1e30,
    max_repeats=10,
    num_particles=3,
    **kwargs,
):
    """Custom estimator for the ELBO that uses the median, rather than mean."""

    def get_loss():
        loss = loss_fun(model, guide, *args, **kwargs)
        repeats = 0
        while torch.abs(loss) > max_loss or torch.isnan(loss):
            loss = loss_fun(model, guide, *args, **kwargs)
            repeats += 1
            if repeats > max_repeats:
                raise ValueError("Too many repeats")
        return loss

    losses = torch.stack([get_loss() for i in range(num_particles)])
    return torch.median(losses, dim=0)[0]


def run_svi(
    model,
    guide,
    optim,
    loss,
    num_iterations,
    input,
    interval=100,
    to_print=None,
    seed=49380,
):
    """Run Pyro's SVI."""
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    svi = pyro.infer.SVI(
        model=model,
        guide=guide,
        optim=optim,
        loss=loss,
    )

    losses = []
    times = []
    time_elapsed = 0
    try:
        for step in range(1, num_iterations + 1):  # Consider running for more steps.

            start_time = time.time()
            loss = svi.step(**input)
            end_time = time.time()

            losses.append(loss)
            time_elapsed = time_elapsed + end_time - start_time
            times.append(time_elapsed)

            if step % interval == 0:
                if to_print:
                    print(to_print)
                print(f"Iteration: {step},\t Total time: {time_elapsed:.2f}s")
                print(
                    f"Median ELBO loss within last {interval} steps: {np.median(losses[step - interval:])}"
                )

                print()

    except Exception as e:
        print(e)
        print("Exception occured!")

    losses = np.array(losses)
    times = np.array(times)
    return (losses, times)


# some conveniance functions
def slice_dict(dictionary, sub_key):
    return {key: value[sub_key] for key, value in dictionary.items()}


def infinite_range():
    num = 1
    while True:
        yield num
        num += 1


ordinal = (
    lambda n: "%d%s "
    % (n, "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10 :: 4])
    if n > 1
    else ""
)


def main():
    ### import the data
    print("importing the data...")
    dataset = "sunspots"
    dataset_names = {
        "ma_simulation": "MA Simulation",
        "sunspots": "Sunspots Dataset",
    }
    dataset_name = dataset_names[dataset]

    df = pd.read_csv(f"../data/{dataset}.csv")
    df.columns = ["X"]
    data = torch.from_numpy(np.array(df["X"]))

    periodogram = torch.from_numpy(
        np.array(pd.read_csv(f"../data/{dataset}_periodogram.csv")["x"])[0:-1]
        / (2 * np.pi)
    )

    omegas = torch.from_numpy(
        np.array(pd.read_csv(f"../data/{dataset}_frequency.csv")["x"])[0:-1] / np.pi
    )
    omegas_np = omegas.detach().numpy() * np.pi

    input = {
        "periodogram": periodogram,
        "omegas": omegas,
        "data": data,
        "L": 20,
        "MIN_K": 20,
        "MAX_K": 200,
        "STEP_K": 5,
    }

    ### plot the data
    fig, ax = plt.subplots()

    ax.plot(
        np.arange(1700, 1700 + len(data))
        if dataset == "sunspots"
        else np.arange(len(data)),
        df,
        label=f"{dataset_name}",
    )

    ax.grid(which="major")
    ax.grid(which="minor", linestyle="--", alpha=0.5)

    ax.set_xlabel("Index" if dataset == "ma_simulation" else "Year")
    ax.set_ylabel(
        "value"
        if dataset == "ma_simulation"
        else "De-Meaned Sqrt. of the Number of Sunspots"
    )

    fig.set_size_inches(6, 4)
    filename = f"../images/{dataset}_data.png"
    fig.savefig(filename)
    plt.close(fig)

    ### sample from the prior
    print("sampling from the prior...")
    prior_samples = np.array([model(**input).numpy() for _ in range(3000)])

    ### run the inference process
    print("running the inference process...")
    interval = 100
    num_particles = 5
    num_iterations = 1000
    seeds = np.arange(10)
    optim = pyro.optim.AdagradRMSProp({"eta": 1, "t": 1 / 2})
    loss = partial(
        my_loss,
        loss_fun=pyro.infer.Trace_ELBO().differentiable_loss,
        num_particles=num_particles,
    )

    results_by_seed = {}
    for seed in seeds:
        losses, times = run_svi(
            model,
            guide,
            input=input,
            optim=optim,
            loss=loss,
            num_iterations=num_iterations,
            interval=interval,
            seed=seed,
        )

        results_by_seed[seed] = (
            losses,
            times,
            {
                param: pyro.param(param)
                for param in [
                    "alpha_V",
                    "beta_V",
                    "alpha_Z",
                    "beta_Z",
                    "alpha_tau",
                    "beta_tau",
                    "ps_K",
                ]
            },
        )

    # Transform the results into a data frame
    results_by_seed_df = pd.DataFrame.from_dict(
        results_by_seed, orient="index", columns=["losses", "times", "parameters"]
    )
    # filter out runs that did not complete
    results_by_seed_df = results_by_seed_df[
        list(map(lambda x: len(x) == num_iterations, results_by_seed_df.losses))
    ]
    # sort the results by their final ELBO
    results_by_seed_df["final loss"] = list(
        map(lambda x: x[-1], results_by_seed_df.losses)
    )
    results_by_seed_df.sort_values("final loss", inplace=True)

    ### Sample from the approximated posterior
    print("sampling from the posterior...")
    num_samples = 3000
    pyro.set_rng_seed(1654)

    samples_by_seed = {}

    for seed, row in results_by_seed_df.iterrows():
        samples_by_seed[seed] = [
            guide(**input, **row.parameters) for _ in range(num_samples)
        ]

    def samples_into_tensor_dict(samples):
        samples_dict = defaultdict(lambda: [])
        samples_list = defaultdict(lambda: [])
        for sample in samples:
            for key, value in sample.items():
                samples_dict[key].append(value)
                samples_list[key].append(value.detach().numpy())
            samples_dict["f"].append(compute_f(**input, **sample))
            samples_list["f"].append(compute_f(**input, **sample).detach().numpy())

        return {
            key: torch.stack(value) for key, value in samples_dict.items()
        }, samples_list

    samples_list_by_seed = {}
    for seed, samples in samples_by_seed.items():
        samples_by_seed[seed], samples_list_by_seed[seed] = samples_into_tensor_dict(
            samples
        )

    # create a pandas dataframe to hold the samples
    samples_df_by_seed = {
        seed: pd.DataFrame.from_dict(samples_list)
        for seed, samples_list in samples_list_by_seed.items()
    }

    # store the approximated spectral densities separately
    spds_by_seed = {}
    for seed, samples_df in samples_df_by_seed.items():
        spds_by_seed[seed] = torch.tensor(samples_df.f.to_list())

    def mean_k(ps_K, MIN_K, MAX_K, STEP_K, **kwargs):
        LEN_K = int(np.ceil((MAX_K - MIN_K) / STEP_K))
        return torch.sum(ps_K * (torch.arange(LEN_K) * STEP_K + MIN_K))

    def expected_value_parameters(
        alpha_V, beta_V, alpha_Z, beta_Z, alpha_tau, beta_tau, ps_K
    ):
        """Compute the expected value approximation."""
        V = dist.Beta(alpha_V, beta_V).mean
        Z = dist.Beta(alpha_Z, beta_Z).mean
        tau = dist.Gamma(alpha_tau, beta_tau).mean
        K = mean_k(ps_K, **input).to(torch.long)
        return {"V": V, "Z": Z, "tau": tau, "K": K}

    ### load the MCMC estimate and true posterior, if available
    mcmc_spd = np.genfromtxt(
        f"../data/{dataset}_mcmc.csv", delimiter=",", skip_header=1
    )
    N = len(mcmc_spd)
    omegas_mcmc = np.linspace(0, np.pi, num=N)
    true_psd = None
    if dataset == "ma_simulation":
        true_psd = np.genfromtxt(
            f"../data/{dataset}_true_psd.csv", delimiter=",", skip_header=1
        )

    print("creating plots...")

    ### plot the prior
    fig, ax = plt.subplots()
    colors = infinite_range()

    ax.plot(
        omegas_np,
        periodogram.detach().numpy(),
        color=f"C{next(colors)}",
        label="Periodogram",
    )

    ax.plot(
        omegas_mcmc,
        mcmc_spd,
        color=f"C{next(colors)}",
        label="MCMC Estimate",
    )

    # plot the mean / median of the samples of the prior
    ax.plot(
        omegas_np,
        np.mean(prior_samples, axis=0),
        color=f"C{next(colors)}",
        label="Point-Wise Mean of Prior Samples",
    )
    ax.plot(
        omegas_np,
        np.median(prior_samples, axis=0),
        color=f"C{next(colors)}",
        label="Point-Wise Median of Prior Samples",
    )

    if true_psd is not None:
        ax.plot(
            omegas_mcmc,
            true_psd,
            color=f"C{next(colors)}",
            label="True Spectral Density",
        )

    ax.set_yscale("log")
    ax.grid(which="major")

    fig.legend()

    ax.set_xlabel("Frequency")
    ax.set_ylabel("Value")

    fig.set_size_inches(6, 4)
    filename = f"../images/{dataset}_prior.png"
    fig.savefig(filename)
    plt.close(fig)

    ### plot the expected value approximation
    num_seeds = 5

    fig, ax = plt.subplots()
    colors = infinite_range()

    ax.plot(
        omegas_np,
        periodogram.detach().numpy(),
        color=f"C{next(colors)}",
        label="Periodogram",
    )

    ax.plot(
        omegas_mcmc,
        mcmc_spd,
        color=f"C{next(colors)}",
        label="MCMC Estimate",
    )

    # plot the expected value approximation
    for index, parameters in enumerate(results_by_seed_df.parameters):
        if index > num_seeds - 1:
            break

        approximation_parameters = expected_value_parameters(**parameters)
        if approximation_parameters is None:
            continue
        ax.plot(
            omegas_np,
            compute_f(**approximation_parameters, **input).detach().numpy(),
            color=f"C{next(colors)}",
            label=f"{ordinal(index+1)}Largest Final ELBO",
        )

    if true_psd is not None:
        ax.plot(
            omegas_mcmc,
            true_psd,
            color=f"C{next(colors)}",
            label="True Spectral Density",
        )

    ax.set_yscale("log")
    ax.grid(which="major")

    lines, labels = ax.get_legend_handles_labels()
    fig.legend(lines, labels, loc="center")

    ax.set_xlabel("Frequency")
    ax.set_ylabel("Value")

    fig.set_size_inches(6, 4)
    filename = f"../images/{dataset}_expected_value_spds_best_seeds.png"
    fig.savefig(filename)
    plt.close(fig)

    ### plot some samples from the posterior with the best final ELBO
    fig, ax = plt.subplots()
    colors = infinite_range()

    spds = spds_by_seed[list(spds_by_seed.keys())[0]]

    ax.plot(
        omegas_np,
        periodogram.detach().numpy(),
        color=f"C{next(colors)}",
        label="Periodogram",
    )

    ax.plot(
        omegas_mcmc,
        mcmc_spd,
        color=f"C{next(colors)}",
        label="MCMC Estimate",
    )

    for index, psd in enumerate(spds[:5]):
        ax.plot(
            omegas_np,
            psd,
            color=f"C{next(colors)}",
            label=f"Sample {index+1}",
        )

    if true_psd is not None:
        ax.plot(
            omegas_mcmc,
            true_psd,
            color=f"C{next(colors)}",
            label="True Spectral Density",
        )

    ax.set_yscale("log")
    ax.grid(which="major")

    lines, labels = ax.get_legend_handles_labels()
    fig.legend(lines, labels, loc="lower right")

    ax.set_xlabel("Frequency")
    ax.set_ylabel("Value")

    fig.set_size_inches(6, 4)
    filename = f"../images/{dataset}_spds_one_seed.png"
    fig.savefig(filename)
    plt.close(fig)

    ### plot the mean of all sample approximation, by seed
    num_seeds = 5

    fig, ax = plt.subplots()
    colors = infinite_range()

    ax.plot(
        omegas_np,
        periodogram.detach().numpy(),
        color=f"C{next(colors)}",
        label="Periodogram",
    )
    ax.plot(
        omegas_mcmc,
        mcmc_spd,
        color=f"C{next(colors)}",
        label="MCMC Estimate",
    )

    for index, spds in enumerate(spds_by_seed.values()):
        if index >= num_seeds:
            break
        ax.plot(
            omegas_np,
            torch.mean(spds, dim=-2).detach().numpy(),
            color=f"C{next(colors)}",
            label=f"{ordinal(index+1)}Largest Final ELBO",
        )

    if true_psd is not None:
        ax.plot(
            omegas_mcmc,
            true_psd,
            color=f"C{next(colors)}",
            label="True Spectral Density",
        )

    ax.set_yscale("log")
    ax.grid(which="major")

    lines, labels = ax.get_legend_handles_labels()
    fig.legend(lines, labels, loc="lower right")

    ax.set_xlabel("Frequency")
    ax.set_ylabel("Value")

    fig.set_size_inches(6, 4)
    filename = f"../images/{dataset}_mean_spd_best_seeds.png"
    fig.savefig(filename)
    plt.close(fig)

    ### plot the median of all sample approximations, by seed
    num_seeds = 5

    fig, ax = plt.subplots()
    colors = infinite_range()

    ax.plot(
        omegas_np,
        periodogram.detach().numpy(),
        color=f"C{next(colors)}",
        label="Periodogram",
    )
    ax.plot(
        omegas_mcmc,
        mcmc_spd,
        color=f"C{next(colors)}",
        label="MCMC Estimate",
    )

    for index, spds in enumerate(spds_by_seed.values()):
        if index >= num_seeds:
            break
        ax.plot(
            omegas_np,
            torch.median(spds, dim=-2)[0].detach().numpy(),
            color=f"C{next(colors)}",
            label=f"{ordinal(index+1)}Largest Final ELBO",
        )

    if true_psd is not None:
        ax.plot(
            omegas_mcmc,
            true_psd,
            color=f"C{next(colors)}",
            label="True Spectral Density",
        )

    ax.set_yscale("log")
    ax.grid(which="major")

    fig.legend(loc="lower right")

    ax.set_xlabel("Frequency")
    ax.set_ylabel("Value")

    fig.set_size_inches(6, 4)
    filename = f"../images/{dataset}_median_spd_best_seeds.png"
    fig.savefig(filename)
    plt.close(fig)

    ### plot the convergence of the inference process
    smoothing_number = "100"
    smoothing_unit = "ms"

    colors = infinite_range()

    fig, ax = plt.subplots()
    for index, result in results_by_seed_df.reset_index().iterrows():
        diagnostics = pd.DataFrame({"elbo": result.losses, "time": result.times})
        diagnostics["time"] = pd.to_datetime(
            (diagnostics["time"] * 10**9).astype(int)
        )
        diagnostics.set_index("time", inplace=True)
        elbo_by_time = (
            -diagnostics["elbo"].rolling(f"{smoothing_number}{smoothing_unit}").median()
        )

        ax.plot(
            elbo_by_time,
            color=f"C{next(colors)}",
            label=f"{ordinal(index+1)}Largest ELBO",
        )

    ax.set_yscale("symlog")

    ax.grid(which="major")
    ax.grid(which="minor", linestyle="--", alpha=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%M:%S"))

    ax.legend()

    ax.set_xlabel("Processing Time (min:s)")
    ax.set_ylabel(f"Median ELBO within ${smoothing_number}\,${smoothing_unit}")

    fig.set_size_inches(6, 4)
    filename = f"../images/{dataset}_convergence.png"
    fig.savefig(filename)
    plt.close(fig)


if __name__ == "__main__":
    main()
