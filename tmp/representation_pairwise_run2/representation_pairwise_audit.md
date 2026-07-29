# Seven-condition corrected still-image paired audit

Generated: 2026-07-23T15:00:00-04:00

## Verification

The corrected v2 artifacts reproduce 346 images and 1,100 unique
question assignments. All seven corrected totals match the expected values,
all corrected predictions match the corresponding frozen stored predictions,
and all correctness values were independently reproduced under the scalar,
alternatives, and joint-required scoring rules.

## Condition ranking

| Condition | Correct | Micro | Supported exact macro | Supported family macro | Image macro |
|---|---:|---:|---:|---:|---:|
| Full multiview | 1083/1,100 | 98.45% | 97.06% | 97.38% | 98.35% |
| Original only | 1065/1,100 | 96.82% | 93.77% | 95.92% | 97.40% |
| Original + raw crops | 1058/1,100 | 96.18% | 94.66% | 96.57% | 97.47% |
| Original + rectified crops | 1052/1,100 | 95.64% | 95.49% | 96.29% | 97.70% |
| Annotated original + raw crops | 1046/1,100 | 95.09% | 93.66% | 94.64% | 97.08% |
| Raw crops only | 1030/1,100 | 93.64% | 89.76% | 90.38% | 92.65% |
| Rectified crops only | 1023/1,100 | 93.00% | 88.71% | 90.20% | 93.22% |

## Interpretable contrasts

| ID | Contrast (second minus first) | Improved | Worsened | Net |
|---|---|---:|---:|---:|
| A | Raw crops only -> Rectified crops only | 18 | 25 | -7 |
| B | Original + raw crops -> Original + rectified crops | 14 | 20 | -6 |
| C | Annotated original + raw crops -> Full multiview | 42 | 5 | +37 |
| D | Original only -> Original + raw crops | 14 | 21 | -7 |
| E | Original only -> Original + rectified crops | 26 | 39 | -13 |
| F | Original + raw crops -> Annotated original + raw crops | 25 | 37 | -12 |
| G | Original + rectified crops -> Full multiview | 35 | 4 | +31 |
| H1 | Original only -> Full multiview | 27 | 9 | +18 |
| H2 | Raw crops only -> Full multiview | 59 | 6 | +53 |
| H3 | Rectified crops only -> Full multiview | 68 | 8 | +60 |
| H4 | Original + raw crops -> Full multiview | 31 | 6 | +25 |
| H5 | Original + rectified crops -> Full multiview | 35 | 4 | +31 |
| H6 | Annotated original + raw crops -> Full multiview | 42 | 5 | +37 |

## Multiplicity and clustering

Question-level Holm correction across 21 exploratory comparisons retained 10 comparisons. Cluster-randomization Holm correction retained 2 comparisons. Complete p-values and bootstrap intervals are in `all_pairwise_comparisons.csv`.

Full multiview had the highest observed accuracy. Its comparisons were:

| Alternative -> Full | Difference | Cluster 95% CI | Cluster randomization p | Holm p |
|---|---:|---:|---:|---:|
| Original only -> Full multiview | +1.64 pp | [0.18, 3.29] pp | 0.0523071 | 0.836914 |
| Raw crops only -> Full multiview | +4.82 pp | [2.61, 7.46] pp | 6.99999e-06 | 0.000147 |
| Rectified crops only -> Full multiview | +5.45 pp | [2.94, 8.35] pp | 1.1e-05 | 0.00022 |
| Original + raw crops -> Full multiview | +2.27 pp | [0.00, 5.58] pp | 0.0998535 | 1 |
| Original + rectified crops -> Full multiview | +2.82 pp | [-0.28, 7.12] pp | 0.304688 | 1 |
| Annotated original + raw crops -> Full multiview | +3.36 pp | [0.74, 7.27] pp | 0.00982666 | 0.17688 |

## Error overlap

- Correct under all seven: 930.
- Incorrect under all seven: 2.
- Correctness varied: 168.
- Full multiview alone correct: 0.
- Full multiview alone incorrect: 1.

## Conservative interpretation

The seven-condition design establishes comparative performance among fixed
representation bundles. It does not independently identify correspondence,
annotation, rectification, localization, or image-count effects because the
conditions are not a complete factorial decomposition. Contrast C estimates
the observed rectification-associated difference with the annotated scene held
constant. Contrast G estimates the observed difference from replacing the
unannotated scene with its annotation/correspondence-bearing version while
rectified crops are supplied; it does not separate annotation from correspondence.

## Smallest recommended manuscript changes (not applied)

1. Replace broad representation claims in Quantitative Results with a compact
   selected-contrasts summary and direct readers to the complete matrix artifact.
2. In Discussion, state which paired contrasts retain cluster-aware evidence
   after Holm correction and explicitly reject a factorial causal interpretation.
3. In Conclusion, retain Full multiview as the best observed bundle while
   clarifying that component effects were not independently isolated.
4. Revise the abstract only if multiplicity-adjusted cluster results materially
   change the current qualified aggregate headline.
