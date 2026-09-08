import numpy as np
import pandas as pd

from maldi_openset.features import peaks_to_sparse_matrix


def test_peak_binning_and_tic_normalization():
    peaks = pd.DataFrame(
        {
            "spectrum_id": ["a", "a", "a", "b"],
            "mass": [2000.2, 2000.8, 2001.2, 2002.4],
            "intensity": [1.0, 2.0, 3.0, 4.0],
        }
    )
    matrix = peaks_to_sparse_matrix(peaks, ["a", "b"], 2000, 2004, 1.0).toarray()
    assert matrix.shape == (2, 4)
    np.testing.assert_allclose(matrix.sum(axis=1), [1.0, 1.0])
    np.testing.assert_allclose(matrix[0, :2], [0.5, 0.5])

