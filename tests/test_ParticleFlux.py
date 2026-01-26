import numpy as np
import pytest

from classes.ParticleFlux import ParticleFlux

# === Creating dummy objects for the following tests ==========================
@pytest.fixture
def dammy_flux():
    return ParticleFlux(verbose=False)

# === Testing ParticleFlux ====================================================
def test_init_defaults(dammy_flux):
    """
    Checks the default inputs.
    """
    assert dammy_flux.fluxDirections is None
    assert dammy_flux.fluxStrength == 0.0
    assert dammy_flux.verbose is False

def test_set_external_flux_rejects_negative_strength(dammy_flux):
    """
    Tests if the class rejects a negative value as the input strength.
    """
    with pytest.raises(ValueError, match="can't be negative"):
        dammy_flux.set_external_flux([[1, 0, 0]], strength=-0.1)

@pytest.mark.parametrize(
    "dirs",
    [
        [1, 0, 0],     
        [[1, 0]],      
        [[1, 0, 0, 0]],
        np.zeros((2, 2)),         
    ], )
def test_set_external_flux_wrong_shape_disables_flux_and_warns(dammy_flux, capsys, dirs):
    """
    Tests if a wrong input shape disables the external flux.
    """
    dammy_flux.set_external_flux(dirs, strength=1.0)
    out, err = capsys.readouterr()

    assert dammy_flux.fluxDirections is None
    assert dammy_flux.fluxStrength == 0.0
    assert "WARNING" in out
    assert "no flux selected" in out

def test_set_external_flux_strength_zero_disables_flux_and_warns(dammy_flux, capsys):
    """
    Tests if a zero strength in input disables the external flux.
    """
    dammy_flux.set_external_flux([[1, 0, 0]], strength=0.0)
    out, err = capsys.readouterr()

    assert dammy_flux.fluxDirections is None
    assert dammy_flux.fluxStrength == 0.0
    assert "WARNING" in out
    assert "no flux selected" in out

def test_set_external_flux_all_zero_vectors_disables_flux_and_warns(dammy_flux, capsys):
    """
    Checks if the flux is disabled if all input vectors are zero vectors.
    """
    dammy_flux.set_external_flux([[0, 0, 0], [0, 0, 0]], strength=1.0)
    out, err = capsys.readouterr()

    assert dammy_flux.fluxDirections is None
    assert dammy_flux.fluxStrength == 0.0
    assert "WARNING" in out
    assert "no valid directions" in out

def test_set_external_flux_filters_zero_vectors_and_normalizes(dammy_flux):
    """
    Checks if the class ignores all the zero vectors in input.
    Also checks if the inputs get normalized.
    """
    dammy_flux.set_external_flux([[0, 0, 0], [2, 0, 0], [0, 3, 0]], strength=1.5)

    assert dammy_flux.fluxStrength == 1.5
    assert dammy_flux.fluxDirections is not None
    assert dammy_flux.fluxDirections.shape == (2, 3)

    norms = np.linalg.norm(dammy_flux.fluxDirections, axis=1)
    assert np.allclose(norms, 1.0)

    got = {tuple(np.round(v, 8)) for v in dammy_flux.fluxDirections}
    expected = {tuple(np.round(v, 8)) for v in np.array([[1, 0, 0], [0, 1, 0]], dtype=float)}
    assert got == expected

def test_clear_external_flux_resets(dammy_flux):
    """
    Test the clear function.
    """
    dammy_flux.set_external_flux([[1, 0, 0]], strength=2.0)
    dammy_flux.clear_external_flux()

    assert dammy_flux.fluxDirections is None
    assert dammy_flux.fluxStrength == 0.0
    assert str(dammy_flux) == "None"

def test_compute_external_flux_weights_zero_vector_returns_1(dammy_flux):
    """
    Tests if the weigth are all 1 if the vectors are all zero.
    """
    dammy_flux.set_external_flux([[1, 0, 0]], strength=2.0)
    assert dammy_flux.compute_external_flux_weights([0, 0, 0]) == 1.0

def test_compute_external_flux_weights_returns_1_when_disabled(dammy_flux):
    """
    Checks if the weigths are 1 if the flux is disabled.
    """
    assert dammy_flux.compute_external_flux_weights([1, 0, 0]) == 1.0

    dammy_flux.set_external_flux([[1, 0, 0]], strength=0.0)
    assert dammy_flux.compute_external_flux_weights([1, 0, 0]) == 1.0

def test_compute_external_flux_weights_invariant_to_scaling_of_input_direction(dammy_flux):
    """
    Tests if the weigths are different for different input direction lenghts.
    """
    dammy_flux.set_external_flux([[1, 0, 0], [0, 1, 0]], strength=1.0)

    w1 = dammy_flux.compute_external_flux_weights([10, 0, 0])
    w2 = dammy_flux.compute_external_flux_weights([1, 0, 0])
    assert np.isclose(w1, w2)

def test_compute_external_flux_weights_multiple_directions_sums_exponentials(dammy_flux):
    """
    Checks if the weights are correct when multiple directions are selected.
    """
    dammy_flux.set_external_flux([[1, 0, 0], [0, 1, 0]], strength=1.0)
    expected = np.exp(1.0 * 1.0) + np.exp(1.0 * 0.0)
    got = dammy_flux.compute_external_flux_weights([1, 0, 0])
    assert np.isclose(got, expected)

