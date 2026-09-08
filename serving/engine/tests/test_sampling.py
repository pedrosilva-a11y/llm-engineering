"""Tests for token sampling utilities."""

import pytest
import torch

from serving.engine.sampling import Sampler, SamplingParameters


@pytest.fixture
def sampler() -> Sampler:
    """Create a sampler for testing."""
    return Sampler(generator=torch.Generator())


# Default behavior


def test_sampling_parameters_defaults() -> None:
    """Create sampling parameters with deterministic default behavior."""
    parameters = SamplingParameters()

    assert parameters.temperature == 1.0
    assert parameters.top_k is None
    assert parameters.top_p is None
    assert parameters.greedy is True


# Temperature


@pytest.mark.parametrize(
    "temperature",
    (
        0.0,
        -0.5,
        -1.0,
    ),
)
def test_sampling_parameters_reject_non_positive_temperature(
    temperature: float,
) -> None:
    """Reject zero and negative sampling temperatures."""
    with pytest.raises(ValueError, match="temperature must be greater than zero"):
        SamplingParameters(temperature=temperature)


@pytest.mark.parametrize(
    "temperature",
    (
        0.1,
        0.5,
        1.0,
        2.0,
    ),
)
def test_sampling_parameters_accept_positive_temperature(
    temperature: float,
) -> None:
    """Accept every strictly positive sampling temperature."""
    parameters = SamplingParameters(temperature=temperature)

    assert parameters.temperature == temperature


def test_apply_temperature_one_leaves_logits_unchanged(
    sampler: Sampler,
) -> None:
    """Leave logits unchanged at neutral temperature."""
    logits = torch.tensor([2.0, 1.0, 0.0])

    scaled_logits = sampler._apply_temperature(
        logits=logits,
        temperature=1.0,
    )

    torch.testing.assert_close(logits, scaled_logits)


def test_apply_temperature_scales_logits(sampler: Sampler) -> None:
    """Divide every logit by the configured temperature."""
    logits = torch.tensor([4.0, 2.0, 1.0])

    scaled_logits = sampler._apply_temperature(
        logits=logits,
        temperature=0.5,
    )

    expected = torch.tensor([8.0, 4.0, 2.0])

    torch.testing.assert_close(scaled_logits, expected)


def test_apply_temperature_preserves_logit_ranking(
    sampler: Sampler,
) -> None:
    """Preserve token ranking after positive temperature scaling."""
    logits = torch.tensor([1.0, 4.0, 2.0, 3.0])

    scaled_logits = sampler._apply_temperature(
        logits=logits,
        temperature=0.5,
    )

    original_order = torch.argsort(logits, descending=True)
    scaled_order = torch.argsort(scaled_logits, descending=True)

    assert torch.equal(original_order, scaled_order)


def test_lower_temperature_sharpen_distribution(
    sampler: Sampler,
) -> None:
    """Increase the highest token probability at lower temperature."""
    logits = torch.tensor([2.0, 1.0, 0.0])

    base_probabilities = torch.softmax(logits, dim=-1)

    scaled_logits = sampler._apply_temperature(
        logits=logits,
        temperature=0.5,
    )
    cold_probabilities = torch.softmax(scaled_logits, dim=-1)

    assert cold_probabilities.max() > base_probabilities.max()


def test_higher_temperature_flattens_distribution(
    sampler: Sampler,
) -> None:
    """Decrease the highest token probability at higher temperature."""
    logits = torch.tensor([2.0, 1.0, 0.0])

    base_probabilities = torch.softmax(logits, dim=-1)

    scaled_logits = sampler._apply_temperature(
        logits=logits,
        temperature=2.0,
    )
    hot_probabilities = torch.softmax(scaled_logits, dim=-1)

    assert base_probabilities.max() > hot_probabilities.max()


def test_apply_temperature_does_not_mutate_input_logits(
    sampler: Sampler,
) -> None:
    """Preserve the original logits tensor when scaling."""
    logits = torch.tensor([2.0, 1.0, 0.0])
    original_logits = logits.clone()

    sampler._apply_temperature(
        logits=logits,
        temperature=0.5,
    )

    torch.testing.assert_close(logits, original_logits)


# Top-k


@pytest.mark.parametrize(
    "top_k",
    [
        -2,
        -1,
        0,
    ],
)
def test_sampling_parameters_reject_non_positive_top_k(top_k: int) -> None:
    """Reject zero and negative top-k values."""
    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        SamplingParameters(top_k=top_k)


@pytest.mark.parametrize(
    "top_k",
    [
        3,
        5,
        10,
    ],
)
def test_sampling_parameters_accept_positive_top_k(top_k: int) -> None:
    """Accept strictly positive top-k values."""
    parameters = SamplingParameters(top_k=top_k)

    assert parameters.top_k == top_k


def test_apply_top_k_none_leaves_logits_unchanged(
    sampler: Sampler,
) -> None:
    """Leave logits unchanged when top-k filtering is disabled."""
    logits = torch.tensor([4.0, 1.0, 3.0, 2.0])

    filtered_logits = sampler._apply_top_k(
        logits=logits,
        top_k=None,
    )

    torch.testing.assert_close(logits, filtered_logits)


def test_apply_top_k_none_returns_clone(sampler: Sampler) -> None:
    """Return independent storage when top-k filtering is disabled."""
    logits = torch.tensor([4.0, 1.0, 3.0, 2.0])

    filtered_logits = sampler._apply_top_k(
        logits=logits,
        top_k=None,
    )
    filtered_logits[0] = 999.0

    assert logits[0].item() == 4.0


def test_apply_top_k_keeps_highest_scoring_candidates(
    sampler: Sampler,
) -> None:
    """Keep only the highest-scoring top-k token candidates."""
    logits = torch.tensor([4.0, 1.0, 3.0, 2.0])

    filtered_logits = sampler._apply_top_k(
        logits=logits,
        top_k=2,
    )

    expected = torch.tensor([4.0, float("-inf"), 3.0, float("-inf")])

    torch.testing.assert_close(filtered_logits, expected)


def test_apply_top_k_one_keeps_only_best_candidate(
    sampler: Sampler,
) -> None:
    """Keep only the highest-scoring token when top-k is one."""
    logits = torch.tensor([1.0, 5.0, 3.0])

    filtered_logits = sampler._apply_top_k(
        logits=logits,
        top_k=1,
    )

    expected = torch.tensor([float("-inf"), 5.0, float("-inf")])

    torch.testing.assert_close(filtered_logits, expected)


def test_apply_top_k_larger_than_vocabulary_keeps_all_logits(
    sampler: Sampler,
) -> None:
    """Clamp top-k to vocabulary size when it is too large."""
    logits = torch.tensor([4.0, 1.0, 3.0, 2.0])

    filtered_logits = sampler._apply_top_k(
        logits=logits,
        top_k=100,
    )

    torch.testing.assert_close(logits, filtered_logits)


def test_apply_top_k_does_not_mutate_input_logits(
    sampler: Sampler,
) -> None:
    """Preserve the original logits tensor when applying top-k filtering."""
    logits = torch.tensor([4.0, 1.0, 3.0, 2.0])
    original_logits = logits.clone()

    sampler._apply_top_k(
        logits=logits,
        top_k=2,
    )

    torch.testing.assert_close(logits, original_logits)


def test_apply_top_k_masked_logits_have_zero_probability(
    sampler: Sampler,
) -> None:
    """Give masked tokens zero probability after softmax."""
    logits = torch.tensor([4.0, 1.0, 3.0, 2.0])

    filtered_logits = sampler._apply_top_k(
        logits=logits,
        top_k=2,
    )
    probabilities = torch.softmax(filtered_logits, dim=-1)

    assert probabilities[1].item() == 0.0
    assert probabilities[3].item() == 0.0


# Top-p


@pytest.mark.parametrize(
    "top_p",
    [
        -1.0,
        -0.5,
        0.0,
        1.1,
        2.0,
    ],
)
def test_sampling_parameters_reject_invalid_top_p(top_p: float) -> None:
    """Reject top-p values outside the interval (0, 1]."""
    with pytest.raises(
        ValueError,
        match="top_p must be greater than zero and at most one",
    ):
        SamplingParameters(top_p=top_p)


@pytest.mark.parametrize(
    "top_p",
    [
        0.1,
        0.5,
        0.9,
        1.0,
    ],
)
def test_sampling_parameters_accepts_valid_top_p(top_p: float) -> None:
    """Accept top-p values in the interval (0, 1]."""
    parameters = SamplingParameters(top_p=top_p)

    assert parameters.top_p == top_p


def test_apply_top_p_none_leaves_logits_unchanged(
    sampler: Sampler,
) -> None:
    """Leave logits unchanged when top-p filtering is disabled."""
    logits = torch.tensor([4.0, 1.0, 3.0, 2.0])

    filtered_logits = sampler._apply_top_p(
        logits=logits,
        top_p=None,
    )

    torch.testing.assert_close(logits, filtered_logits)


def test_apply_top_p_none_returns_clone(sampler: Sampler) -> None:
    """Return independent storage when top-p filtering is disabled."""
    logits = torch.tensor([4.0, 1.0, 3.0, 2.0])

    filtered_logits = sampler._apply_top_p(
        logits=logits,
        top_p=None,
    )
    filtered_logits[0] = 999.0

    assert logits[0].item() == 4.0


def test_apply_top_p_one_keeps_all_logits(sampler: Sampler) -> None:
    """Keep the full vocabulary when top-p is one."""
    logits = torch.tensor([4.0, 1.0, 3.0, 2.0])

    filtered_logits = sampler._apply_top_p(
        logits=logits,
        top_p=1.0,
    )

    torch.testing.assert_close(logits, filtered_logits)


def test_apply_top_p_keeps_smallest_nucleus_reaching_threshold(
    sampler: Sampler,
) -> None:
    """Keep the smallest probability prefix that reaches top-p."""
    logits = torch.log(torch.tensor([0.15, 0.50, 0.05, 0.30]))

    filtered_logits = sampler._apply_top_p(
        logits=logits,
        top_p=0.75,
    )

    expected = torch.tensor(
        [
            float("-inf"),
            logits[1].item(),
            float("-inf"),
            logits[3].item(),
        ]
    )

    torch.testing.assert_close(filtered_logits, expected)


def test_apply_top_p_keeps_cutoff_token(sampler: Sampler) -> None:
    """Keep the token that causes cumulative probability to cross top-p."""
    logits = torch.log(torch.tensor([0.50, 0.30, 0.15, 0.05]))

    filtered_logits = sampler._apply_top_p(
        logits=logits,
        top_p=0.75,
    )

    assert torch.isfinite(filtered_logits[0])
    assert torch.isfinite(filtered_logits[1])
    assert torch.isneginf(filtered_logits[2])
    assert torch.isneginf(filtered_logits[3])


def test_apply_top_p_small_threshold_keeps_best_token(
    sampler: Sampler,
) -> None:
    """Keep only the highest-probability token when it already reaches top-p."""
    logits = torch.log(torch.tensor([0.60, 0.20, 0.15, 0.05]))

    filtered_logits = sampler._apply_top_p(
        logits=logits,
        top_p=0.5,
    )

    assert torch.isfinite(filtered_logits[0])
    assert torch.isneginf(filtered_logits[1])
    assert torch.isneginf(filtered_logits[2])
    assert torch.isneginf(filtered_logits[3])


def test_apply_top_p_adapts_candidate_count_to_distribution(sampler: Sampler) -> None:
    """Keep fewer candidates for peaked distribution than flat ones."""
    top_p = 0.75

    peaked_logits = torch.log(torch.tensor([0.80, 0.10, 0.05, 0.05]))
    flat_logits = torch.log(torch.tensor([0.30, 0.25, 0.25, 0.20]))

    peaked_filtered = sampler._apply_top_p(
        logits=peaked_logits,
        top_p=top_p,
    )
    flat_filtered = sampler._apply_top_p(
        logits=flat_logits,
        top_p=top_p,
    )

    peaked_candidates = torch.isfinite(peaked_filtered).sum().item()
    flat_candidates = torch.isfinite(flat_filtered).sum().item()

    assert peaked_candidates == 1
    assert flat_candidates == 3


def test_apply_top_p_does_not_mutate_input_logits(
    sampler: Sampler,
) -> None:
    """Preserve the original logits tensor when applying top-p filtering."""
    logits = torch.tensor([4.0, 1.0, 3.0, 2.0])
    original_logits = logits.clone()

    sampler._apply_top_p(
        logits=logits,
        top_p=0.75,
    )

    torch.testing.assert_close(logits, original_logits)


def test_apply_top_p_masked_logits_have_zero_probability(
    sampler: Sampler,
) -> None:
    """Give tokens outside the nucleus zero probability after softmax."""
    logits = torch.log(torch.tensor([0.15, 0.50, 0.05, 0.30]))

    filtered_logits = sampler._apply_top_p(
        logits=logits,
        top_p=0.75,
    )
    probabilities = torch.softmax(filtered_logits, dim=-1)

    assert probabilities[0].item() == 0.0
    assert probabilities[2].item() == 0.0


# Filter pipeline


def test_filter_logits_applies_sampling_filters_in_order(
    sampler: Sampler,
) -> None:
    """Apply temperature, top-k, and top-p filtering in order."""
    temperature = 0.5
    logits = torch.log(torch.tensor([0.40, 0.30, 0.20, 0.10]))
    parameters = SamplingParameters(
        temperature=temperature,
        top_k=3,
        top_p=0.75,
        greedy=False,
    )

    filtered_logits = sampler._filter_logits(
        logits=logits,
        parameters=parameters,
    )

    expected = torch.tensor(
        [
            (logits[0] / temperature).item(),
            (logits[1] / temperature).item(),
            float("-inf"),
            float("-inf"),
        ]
    )

    torch.testing.assert_close(filtered_logits, expected)


def test_filter_logits_does_not_mutate_input_logits(sampler: Sampler) -> None:
    """Preserve the original logits when applying the filter pipeline."""
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0])
    original_logits = logits.clone()

    parameters = SamplingParameters(
        temperature=0.5,
        top_k=3,
        top_p=0.8,
        greedy=False,
    )

    sampler._filter_logits(logits=logits, parameters=parameters)

    torch.testing.assert_close(logits, original_logits)


# Token selection


def test_sample_token_greedy_selects_highest_logit(sampler: Sampler) -> None:
    """Select the highest-scoring token during greedy decoding."""
    logits = torch.tensor([1.0, 5.0, 3.0])
    parameters = SamplingParameters(greedy=True)

    token_id = sampler.sample_token(logits=logits, parameters=parameters)

    assert token_id == 1
    assert isinstance(token_id, int)


def test_sample_token_top_k_one_selects_only_candidate(sampler: Sampler) -> None:
    """Select the only surviving candidate when top-k is one."""
    logits = torch.tensor([1.0, 5.0, 3.0])
    parameters = SamplingParameters(top_k=1, greedy=False)

    token_id = sampler.sample_token(logits=logits, parameters=parameters)

    assert token_id == 1
    assert isinstance(token_id, int)


def test_sample_token_applies_full_sampling_pipeine(sampler: Sampler) -> None:
    """Apply configured filters before sampling through the public interface."""
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0])

    parameters = SamplingParameters(
        temperature=0.5,
        top_k=3,
        top_p=0.6,
        greedy=False,
    )

    token_id = sampler.sample_token(logits=logits, parameters=parameters)

    assert token_id == 0


def test_sample_token_is_reproducible_with_same_seed() -> None:
    """Produce the same sampled tokens from generators with the same seed."""
    seed = 42

    first_sampler = Sampler(
        generator=torch.Generator().manual_seed(seed),
    )
    second_sampler = Sampler(
        generator=torch.Generator().manual_seed(seed),
    )

    logits = torch.log(torch.tensor([0.40, 0.30, 0.20, 0.10]))
    parameters = SamplingParameters(greedy=False)

    first_tokens = [first_sampler.sample_token(logits, parameters) for _ in range(10)]

    second_tokens = [second_sampler.sample_token(logits, parameters) for _ in range(10)]

    assert first_tokens == second_tokens


def test_sample_token_greedy_does_not_consume_generator_state() -> None:
    """Avoid consuming random state during greedy decoding."""
    seed = 42

    sampler_generator = torch.Generator().manual_seed(seed)
    reference_generator = torch.Generator().manual_seed(seed)

    sampler = Sampler(generator=sampler_generator)

    sampler.sample_token(
        logits=torch.tensor([1.0, 5.0, 3.0]),
        parameters=SamplingParameters(greedy=True),
    )

    next_sampler_random = torch.rand((), generator=sampler_generator)
    next_reference_random = torch.rand((), generator=reference_generator)

    torch.testing.assert_close(next_sampler_random, next_reference_random)
