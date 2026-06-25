"""
Plot configuration and helper functions for the audio data mining notebooks.

These utilities keep the notebooks focused on concepts instead of repeating
plot styling code in every notebook.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------
# Plot configuration
# ---------------------------------------

DPI = 150

FIGSIZE_WIDE = (12, 4)
FIGSIZE_TALL = (12, 5)

GRAYSCALE = {
    "main": "0.2",
    "secondary": "0.3",
    "light": "0.7",
    "band": "0.8",
    "black": "0.0",
    "dark": "#333333",
}

ACCENT = {
    "blue": "#1f4e79",
    "terracotta": "#b85c38",
    "gray": "#7a7a7a",
}


def apply_plot_style() -> None:
    """
    Apply a clean, publication-inspired plotting style.
    """
    plt.rcParams.update({
        # Typography
        "font.family": "serif",
        "font.size": 11,

        # Axes
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "axes.linewidth": 1.1,

        # Ticks
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4,
        "ytick.major.size": 4,

        # Lines
        "lines.linewidth": 1.4,

        # Grid
        "grid.alpha": 0.20,

        # Background
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",

        # Legend
        "legend.fontsize": 10,
        "legend.frameon": False,
    })


def clean_axes(ax, grid_axis: str | None = "y") -> None:
    """
    Apply a clean style to a matplotlib axes object.

    Parameters
    ----------
    ax:
        Matplotlib axes object.
    grid_axis:
        Axis for grid lines: "x", "y", "both", or None.
        If None, no grid is shown.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color(GRAYSCALE["dark"])
    ax.spines["bottom"].set_color(GRAYSCALE["dark"])

    ax.tick_params(axis="both", colors=GRAYSCALE["dark"])

    if grid_axis is None:
        ax.grid(False)
    else:
        ax.grid(True, axis=grid_axis, alpha=0.20)


def align_axes_at_origin(ax) -> None:
    """
    Align the visible x and y axes at the data origin.

    This is useful for spectra and conceptual plots where the lower-left
    corner should correspond exactly to (0, 0).
    """
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["left"].set_position(("data", 0))


def plot_magnitude_spectrum(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    title: str,
    xlim: Tuple[float, float],
    xlabel: str = "Frequency [Hz]",
    ylabel: str = "Magnitude",
    figsize: Tuple[float, float] = FIGSIZE_WIDE,
    dpi: int = DPI,
    color: str = ACCENT["blue"],
) -> None:
    """
    Plot a magnitude spectrum using the project style.

    The x- and y-axes are forced to start at zero so that the origin
    of the abscissa and ordinate coincide visually.

    Parameters
    ----------
    frequencies:
        Frequency axis in Hz.
    magnitude:
        Magnitude spectrum values.
    title:
        Plot title.
    xlim:
        Frequency range to show in Hz.
    xlabel:
        x-axis label.
    ylabel:
        y-axis label.
    figsize:
        Figure size in inches.
    dpi:
        Figure resolution.
    color:
        Line color.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.plot(
        frequencies,
        magnitude,
        color=color,
        linewidth=1.4,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xlim(*xlim)

    # Compute y-limit using the visible frequency range.
    visible = (frequencies >= xlim[0]) & (frequencies <= xlim[1])
    if np.any(visible):
        y_max = float(np.nanmax(magnitude[visible]))
    else:
        y_max = float(np.nanmax(magnitude))

    if not np.isfinite(y_max) or y_max <= 0:
        y_max = 1.0

    ax.set_ylim(0, y_max * 1.08)

    clean_axes(ax, grid_axis="y")
    align_axes_at_origin(ax)

    plt.tight_layout()
    plt.show()


def plot_mel_filter_bank(
    fft_frequencies: np.ndarray,
    mel_filter_bank: np.ndarray,
    n_mels: Optional[int] = None,
    xlim: Tuple[float, float] = (0, 8000),
    figsize: Tuple[float, float] = FIGSIZE_TALL,
    dpi: int = DPI,
) -> None:
    """
    Plot all triangular mel filters using a clean grayscale style.
    """
    if n_mels is None:
        n_mels = mel_filter_bank.shape[0]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for i in range(n_mels):
        ax.plot(
            fft_frequencies,
            mel_filter_bank[i],
            color=GRAYSCALE["secondary"],
            alpha=0.55,
            linewidth=1.0,
        )

    ax.set_title(f"Mel filter bank with {n_mels} triangular filters")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Filter weight")

    ax.set_xlim(*xlim)
    ax.set_ylim(0, float(mel_filter_bank.max()) * 1.08)

    clean_axes(ax, grid_axis="y")
    align_axes_at_origin(ax)

    plt.tight_layout()
    plt.show()


def plot_selected_mel_filters(
    fft_frequencies: np.ndarray,
    mel_filter_bank: np.ndarray,
    selected_filters: Optional[Iterable[int]] = None,
    xlim: Tuple[float, float] = (0, 8000),
    figsize: Tuple[float, float] = FIGSIZE_TALL,
    dpi: int = DPI,
) -> None:
    """
    Plot all mel filters in light gray and highlight selected filters.
    """
    n_mels = mel_filter_bank.shape[0]

    if selected_filters is None:
        selected_filters = [0, 5, 10, 15, 20, 25, 30, 35, n_mels - 1]

    selected_filters = [i for i in selected_filters if 0 <= i < n_mels]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # All filters in light gray
    for i in range(n_mels):
        ax.plot(
            fft_frequencies,
            mel_filter_bank[i],
            color=GRAYSCALE["light"],
            linewidth=0.9,
            alpha=0.75,
        )

    # Highlight selected filters
    for i in selected_filters:
        ax.plot(
            fft_frequencies,
            mel_filter_bank[i],
            linewidth=1.8,
            label=f"Filter {i + 1}",
        )

    ax.set_title(f"Selected mel filters from a {n_mels}-band filter bank")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Filter weight")

    ax.set_xlim(*xlim)
    ax.set_ylim(0, float(mel_filter_bank.max()) * 1.08)

    clean_axes(ax, grid_axis="y")
    align_axes_at_origin(ax)

    ax.legend(ncol=3, loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_mel_filter_bank_matrix(
    mel_filter_bank: np.ndarray,
    sr: int,
    title: Optional[str] = None,
    xlim: Tuple[float, float] = (0, 8000),
    figsize: Tuple[float, float] = FIGSIZE_TALL,
    dpi: int = DPI,
) -> None:
    """
    Plot the mel filter bank as a matrix.

    Rows correspond to mel filters, columns correspond to FFT frequency bins,
    and values correspond to filter weights.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    n_mels, _ = mel_filter_bank.shape
    x_min, x_max = 0, sr / 2

    image = ax.imshow(
        mel_filter_bank,
        aspect="auto",
        origin="lower",
        extent=[x_min, x_max, 0, n_mels - 1],
        interpolation="nearest",
    )

    fig.colorbar(image, ax=ax, label="Filter weight")

    if title is None:
        title = f"Mel filter bank matrix — {n_mels} mel bands"

    ax.set_title(title)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Mel filter index")
    ax.set_xlim(*xlim)

    # No grid for matrix plots.
    clean_axes(ax, grid_axis=None)

    plt.tight_layout()
    plt.show()
