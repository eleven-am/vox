# Acoustic gate clause ablation (R2b)

Protocol: for each clause in the acoustic gate, run `scripts/benchmark_interruptions.py`
with only that clause disabled (threshold set to its pass-everything value) and compare
per-category false-positive and recall counts against the full gate. A clause is deleted
unless removing it worsens some category's false-positive rate; a surviving clause names
the category that witnesses it. Corpus precondition: a sparse-periodic-impulse category
(`phone_vibration`, 170 Hz bursts of 60 ms every 300 ms ending on a burst) was added
before ablating, because `voiced_frame_ratio` and `spectral_flatness` are computed over
active frames only, leaving `active_frame_ratio` as the only sparsity witness.

## Per-clause results

Corpus: 26 cases, 10 genuine (positives), 16 adversarial (negatives), 22 categories.
Re-verified after the confirmation-latency metric was corrected to use each
mechanism's real timer-arming computation (warmup barge-in and correlated-echo
late-transcript cases added); every clause verdict and witness below is
unchanged, with the full gate still at FPR 0.000 / recall 1.000.
Cells are false positives / negatives in category. Only categories with any delta are
shown per variant; every other category was identical to the full gate.

| variant | overall FPR | recall | FP reduction vs legacy | flipped cases | category deltas |
|---|---|---|---|---|---|
| full gate | 0.000 | 1.000 | 100% | — | — |
| min_rms removed | 0.000 | 1.000 | 100% | none | none |
| min_active_frame_ratio removed | 0.062 | 1.000 | 91.7% | phone_vibration_bursts | phone_vibration 0/1 -> 1/1, tts_playback 0/16 -> 1/16 |
| max_crest_factor removed | 0.000 | 1.000 | 100% | none | none |

## Verdict

| clause | verdict | named category (KEEP ledger) |
|---|---|---|
| min_rms | DELETED | no category witnesses it; every negative it could reject is already rejected by tail_rms, active_frame_ratio, voiced_frame_ratio, or spectral_flatness |
| min_active_frame_ratio | KEPT | phone_vibration — sparse periodic impulses in silence look voiced over active frames; frame sparsity is the only feature that rejects them |
| max_crest_factor | DELETED | no category witnesses it; impulse-heavy negatives (tap, keyboard, cough) are rejected by active-frame sparsity, voicing, or flatness before crest matters |

## Post-deletion gate

The surviving gate is `duration >= min_duration`, `tail_rms >= min_tail_rms`
(waived under EOU >= 0.5), `active_frame_ratio >= 0.55`, `voiced_frame_ratio >= 0.20`,
`spectral_flatness <= 0.70`. After the deletions the benchmark holds its full bar:
FPR 0.000, recall 1.000, FP reduction vs legacy 100%, no latency regression.
