import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


if __name__ == "__main__":
    scipy = pd.read_csv("examples/latent_tgas/scipy/scipy_no_heating.csv", index_col=0)
    jax = pd.read_csv("examples/latent_tgas/jax/jax_no_heating.csv", index_col=0)
    carbox = pd.read_csv("carbox/carbox.csv", index_col=0)

    fig, ax = plt.subplots(1, 1, figsize=(19, 7), sharex=True, sharey=True)
    molecules = carbox.columns
    ax.set_prop_cycle(plt.cycler("color", plt.cm.tab20.colors))
    carbox.loc[:, molecules].plot(ax=ax, loglog=True)
    ax.set_prop_cycle(plt.cycler("color", plt.cm.tab20.colors))
    jax.loc[:, molecules].plot(
        ax=ax, loglog=True, linestyle="--", label=[None] * len(molecules), legend=None
    )
    ax.set_prop_cycle(plt.cycler("color", plt.cm.tab20.colors))
    scipy.loc[:, molecules].plot(ax=ax, loglog=True, linestyle="dotted")

    handles, labels = ax.get_legend_handles_labels()
    handles = handles[: len(molecules)]
    labels = labels[: len(molecules)]
    handles.extend(
        [
            Line2D([0], [0], color="black", lw=2, label="carbox"),
            Line2D(
                [0],
                [0],
                color="black",
                lw=2,
                linestyle="--",
                label="jax implementation",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                lw=2,
                linestyle="dotted",
                label="scipy implementation",
            ),
        ]
    )
    ax.legend(handles=handles)
    ax.axvline(1e10)
    ax.set_ylim(1e-15, 1e5)
    ax.set(xlabel="Time (s)", ylabel="Abundance (cm$^{-3}$)")
    fig.show()
    plt.show()
