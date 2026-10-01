"""Distinguish global edge budgets from per-record lattice limits for future runs.

The original wrapper/kernel stay immutable. Their lowered per-record edge cap
can signal the cumulative budget using the same native error message. This
adapter promotes such exhaustion to an explicit work-budget exception.
"""

from voynich.native_censored_bridge import CensoredNativeBridge


class CensoredWorkBudgetExceeded(RuntimeError):
    """Cumulative work exhausted, never a per-record graph-cap outcome."""


class StrictCensoredNativeBridge(CensoredNativeBridge):
    """For new registrations only; does not alter previously closed experiments."""

    def _record(self,*args,**kwargs):
        try:
            return super()._record(*args,**kwargs)
        except RuntimeError as exc:
            if self.edges>=self.max_edges and 'edge cap' in str(exc):
                raise CensoredWorkBudgetExceeded(
                    "Native bridge cumulative edge budget exhausted; no graph-cap substitution"
                ) from exc
            raise
