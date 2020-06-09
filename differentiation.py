#!/usr/bin/env python
# coding: utf-8
# spectral_differentiation.py
"""Functions for computing spectral differentiation."""

import math

import numpy as np
import scipy.spatial
from numpy.core.multiarray import normalize_axis_index as _normalize_axis_index
# from toolz import curry


def join_axes(a, b, array):
    """Join adjacent axes ``a`` and ``b``."""
    a = _normalize_axis_index(a, array.ndim)
    b = _normalize_axis_index(b, array.ndim)
    # Check that the axes to be joined are adjacent
    if abs(a - b) != 1:
        raise np.AxisError(f"Axes to be joined must be adjacent; got {a} and {b}.")
    # Ensure that `a` is the earlier axis and `b` is the later
    a, b = min(a, b), max(a, b) # NOTE: Must be on one line
    # Compute the new shape
    return array.reshape(
        array.shape[:a] + (array.shape[a] * array.shape[b],) + array.shape[(b + 1) :]
    )


def split_axis(axis, a, b, array):
    """Split an axis into two adjacent axes of size ``a`` and ``b``."""
    axis = _normalize_axis_index(axis, array.ndim)
    a = int(a)
    b = int(b)
    if array.shape[axis] != a * b and a != -1 and b != -1:
        raise ValueError(
            f"Axis {axis} of size {array.shape[axis]} cannot be "
            f"split into axes of sizes {a} and {b}."
        )
    new_shape = array.shape[:axis] + (a, b) + array.shape[(axis + 1) :]
    return array.reshape(*map(int, new_shape))


def window(sample_rate, window_length, data):
    """Split data into windows indexed by the third-to-last dimension.

    Assumes the data has shape (..., neuron, sample).

    Returns:
    An array with shape (..., window, neuron, sample).
    """
    # Check that data can be windowed with the given parameters
    samples_per_window = sample_rate * window_length
    num_windows = data.shape[-1] / samples_per_window
#     print('sample rate', sample_rate)
#     print('window length', window_length)
#     print(data.shape)
#     print('samples per window', samples_per_window)
#     print('num windows', num_windows)
    if not num_windows.is_integer():
        print(sample_rate, window_length, data.shape)
        raise ValueError("Data cannot be windowed evenly.")
    # Split last axis into windows and samples
    data = split_axis(-1, num_windows, samples_per_window, data)
    # Move window axis before neuron axis
    return np.moveaxis(data, -2, -3)


# TODO add arguments to docstring
def spectral_states(
    sample_rate, window_length, data, average_spectra=False, axes=(-1,)
):
    """Compute the power spectrum for each window, then concatenate spectra for
    each neuron within each window.

    Assumes the data has shape (..., trial, neuron, sample).

    Returns:
    An array containing spectra for each window, with shape
    (..., trial, window, frequency). If ``average_spectra`` is ``True``,
    then the shape is (..., window, frequency).
    """
    data = window(sample_rate, window_length, data)
    # Compute the FFT along the last axis by default
    spectra = np.fft.rfftn(data, axes=axes)
    # Get the power spectrum
    spectra = np.square(np.abs(spectra))
    if average_spectra:
        # Assumes shape is now (..., trial, window, neuron, sample)
        # -4 is the axis before the window
        # TODO handle this with params?
        spectra = spectra.mean(axis=(-4), keepdims=True)
    # Concatenate power spectra of each neuron
    return join_axes(-2, -1, spectra)


# TODO test
def time_domain_states(sample_rate, window_length, data, axes=(-1,)):
    """Window the data, then concatenate data for each neuron within each window.

    Assumes the data has shape (..., trial, neuron, sample).

    Returns:
    An array containing spectra for each window, with shape
    (..., trial, window, sample). If ``average`` is ``True``,
    then the shape is (..., window, sample).
    """
    raise Exception("time_domain_states is not tested; write tests before using")
    data = window(sample_rate, window_length, data)
    # Concatenate timeseries of each neuron
    return join_axes(-2, -1, data)


def differentiation(states):
    """Return the differentiation among a set of states.

    Returns:
    A condensed pairwise distance matrix containing the Euclidean distances
    between the states.

    Note:
    From the NumPy docs for `pdist`::
    Returns a condensed distance matrix Y. For each i and j (where
    i<j<m), where m is the number of original observations, the metric
    dist(u=X[i], v=X[j]) is computed and stored in entry ij.
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html)
    """
    # Compute the pairwise distances
    return scipy.spatial.distance.pdist(states, metric="euclidean")


def squareform_side_length(condensed_length):
    """Return the side length of the un-condensed, square form of a consdensed
    distance matrix."""
    return int(math.ceil(math.sqrt(condensed_length * 2)))


def spectral_differentiation(
    data, spectral=True, window_length=1, average_spectra=False,
    axes=(-1,), sample_rate=50, sqrt=False
):
    """Return the spectral differentiation.

    Returns:
    A condensed pairwise distance matrix containing the Euclidean distances
    between the spectra, concatenated across all neurons, of 1 s
    non-overlapping windows of the traces.
    """
    # Get spectral states
    if spectral:
        states = spectral_states(
            sample_rate,
            window_length,
            data,
            average_spectra=average_spectra,
            axes=axes,
        )
    else:
        states = time_domain_states(
            sample_rate, window_length, data, axes=axes
        )
    if sqrt:
        states = np.sqrt(states)
    # Flatten all dimensions except the last two to get a list of state sets
    state_sets = states.reshape(-1, states.shape[-2], states.shape[-1])
    # Compute differentiation for each state set
    distances = np.array(list(map(differentiation, state_sets)))
    # Restore the flattened dimensions
    return distances.reshape(*states.shape[:-2], -1)