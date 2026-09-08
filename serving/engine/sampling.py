"""Token sampling utilities for the LLM inference engine.

Sampling applies operations in the following order:

    temperature scaling
    -> top-k masking
    -> top-p masking over the remaining logits
    -> final softmax
    -> inverse-CDF sampling

This ordering is an engine-level sampling contract. Other implementations may choose
different filtering orders.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingParameters:
    """Configure token selection from model logits.

    Attributes:
        temperature: Positive temperature used to scale logits.
        top_k: Maximum number of highest-scoring candidates to retain.
        top_p: Maximum cumulative probability mass for nucleus sampling.
        greedy: Whether to select the highest-scoring token deterministically.
    """

    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    greedy: bool = True

    def __post_init__(self) -> None:
        """Validate sampling parameters.

        Raises:
            ValueError: If temperature is not greater than zero, top_k is not
                positive when provided, or top_p is outside of the interval (0, 1].
        """
        if self.temperature <= 0:
            raise ValueError("temperature must be greater than zero.")

        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be greater than zero and at most one.")


class Sampler:
    """Select tokens from model logits using configurable sampling strategies."""

    def __init__(self, generator: torch.Generator) -> None:
        """Initialize the sampler.

        Args:
            generator: Random-number generator used for reproducible sampling.
        """
        self._generator = generator

    def sample_token(
        self,
        logits: torch.Tensor,
        parameters: SamplingParameters,
    ) -> int:
        """Select one token from vocabulary logits.

        Args:
            logits: Vocabulary logits for one sequence.
            parameters: Sampling configuration for the sequence.

        Returns:
            Selected vocabulary token ID.
        """
        if parameters.greedy:
            return int(torch.argmax(logits).item())

        filtered_logits = self._filter_logits(logits=logits, parameters=parameters)

        probabilities = torch.softmax(filtered_logits, dim=-1)
        cumulative_probabilities = torch.cumsum(probabilities, dim=-1)

        cumulative_probabilities = (
            cumulative_probabilities / cumulative_probabilities[-1]
        )

        random_value = torch.rand(
            (),
            generator=self._generator,
            device=logits.device,
        )

        token_index = torch.searchsorted(
            cumulative_probabilities,
            random_value,
            right=True,
        )

        return int(token_index.item())

    def _filter_logits(
        self,
        logits: torch.Tensor,
        parameters: SamplingParameters,
    ) -> torch.Tensor:
        """Apply temperature, top-k, and top-p filters in the declared order."""
        filtered_logits = self._apply_temperature(
            logits=logits,
            temperature=parameters.temperature,
        )

        filtered_logits = self._apply_top_k(
            logits=filtered_logits,
            top_k=parameters.top_k,
        )

        filtered_logits = self._apply_top_p(
            logits=filtered_logits,
            top_p=parameters.top_p,
        )

        return filtered_logits

    @staticmethod
    def _apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Scale logits by a positive sampling temperature."""
        return logits / temperature

    @staticmethod
    def _apply_top_k(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
        """Keep only the highest-scoring top-k logits.

        Notes:
            When multiple logits are tied at the top-k boundary, which tied
            tokens are retained is unspecified.
        """
        if top_k is None:
            return logits.clone()

        k = min(top_k, logits.numel())

        _, top_indices = torch.topk(logits, k=k)

        filtered_logits = torch.full_like(logits, fill_value=float("-inf"))
        filtered_logits[top_indices] = logits[top_indices]

        return filtered_logits

    @staticmethod
    def _apply_top_p(logits: torch.Tensor, top_p: float | None) -> torch.Tensor:
        """Keep the smallest token set reaching the configured probability mass."""
        if top_p is None or top_p == 1.0:
            return logits.clone()

        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
        )

        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

        cutoff_indices = torch.nonzero(
            cumulative_probabilities >= top_p,
            as_tuple=False,
        )
        cutoff_index = (
            int(cutoff_indices[0].item())
            if cutoff_indices.numel() > 0
            else logits.numel() - 1
        )

        indices_to_remove = sorted_indices[cutoff_index + 1 :]

        filtered_logits = logits.clone()
        filtered_logits[indices_to_remove] = float("-inf")

        return filtered_logits
