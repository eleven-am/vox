from __future__ import annotations

import pytest

from scripts.benchmark_interruptions import REQUIRED_CATEGORIES, benchmark_cases, run_benchmark


@pytest.mark.asyncio
async def test_adversarial_interruption_benchmark_meets_acceptance_thresholds() -> None:
    report = await run_benchmark()

    assert report["passed"], report


def test_adversarial_corpus_covers_every_required_category() -> None:
    covered = {category for case in benchmark_cases() for category in case.categories}

    assert covered >= REQUIRED_CATEGORIES
