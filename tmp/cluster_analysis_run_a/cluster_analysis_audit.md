# Image-cluster-aware corrected still-image analysis

Generated: 2026-07-23T17:32:06Z

## Input audit

- Corrected manifest: `TRB corrected reference semantics` (SHA-256 recorded in JSON).
- 346 image clusters and 1,100 questions verified.
- Original only: 1,065/1,100; Full multiview: 1,083/1,100.
- Paired outcomes: 1,056 both correct, 9 Original only, 27 Full only, and 8 both incorrect.
- All stored Original-only and Full-multiview predictions match the corrected rows.
- No image identifier or composite question identifier is missing or duplicated.
  The source has no explicit global question-ID field, so conflicting assignments
  for such an identifier cannot be tested.

Questions per image range from 1 to 19 (median
2.000, mean 3.179191).

## Cluster bootstrap

Using 100,000 image-cluster replicates
with seed 20260723, the observed micro-accuracy difference
was 0.01636364. The bootstrap mean was
0.01642379, SE
0.00798314, 95% percentile CI
[0.00180501,
0.03292568], and 99% percentile CI
[-0.00209212,
0.03899723]. The proportion of
replicates at or below zero was
0.01602000.

## Macro image-level analysis

- Original macro image accuracy: 0.97402858
- Full macro image accuracy: 0.98350996
- Mean/median image difference: 0.00948137 /
  0.00000000
- Full wins / Original wins / ties: 11 /
  7 / 328
- Macro-difference 95% percentile CI:
  [-0.00646295,
  0.02568260]

## Paired sensitivity tests

The exact image-cluster label-swap test enumerated
262,144 assignments and produced a two-sided p-value of
0.0523071289. The exact sign test over the
18 non-tied images produced p =
0.4806823730; this coarser test ignores effect
magnitudes and question counts.

## Interpretation and limitation

The improvement has qualified rather than uniform support after preserving all
questions from an image within each resampled or label-swapped cluster. The 95%
micro cluster-bootstrap interval excludes zero, but the 99% interval includes
zero and the exact cluster randomization test is just above the conventional
0.05 threshold. The macro-image interval includes zero, and the image win/loss
sign test is not significant. These results weaken any claim that Full
multiview wins broadly across images; the aggregate benefit is concentrated in
a small number of images with larger gains.

No verified physical-sign mapping exists. Image clustering therefore does not
establish independence across repeated photographs of the same physical sign and
does not support a physical-sign-level generalization claim.

## Smallest accurate manuscript revision

Retain the existing question-level McNemar result but explicitly qualify it.
Add the image-cluster bootstrap difference with both its 95% and 99% percentile
intervals, the exact cluster-randomization p-value, and one concise sentence
noting that the macro interval and image-level sign test were not significant.
Also state that no physical-sign mapping was available. Do not alter the
manuscript without author approval.
