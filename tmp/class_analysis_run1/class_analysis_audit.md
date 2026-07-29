# Corrected still-image direction-class audit

Generated: 2026-07-23T12:00:00-04:00

## Verification and definitions

The authoritative corrected artifacts reproduce 346 images, 1,100 questions,
1,065 Original-only correct answers, 1,083 Full-multiview correct answers, and
paired counts 1,056/9/27/8. Stored predictions match every corrected row.

Exact categories preserve semantic type and the complete answer set. Families
are mutually exclusive: horizontal, vertical/forward, diagonal, locational,
unknown/uncertain, mixed multi-direction, and a separate u-turn/other family.
Multi-label references spanning families are not forced into one constituent
family. The unknown/uncertain family has zero support and remains explicit with
undefined accuracy.

## Imbalance

- Left is present in 486 references.
- Right is present in 427 references.
- The left-or-right union contains 909
  questions; 4 contain both.
- Rare exact categories (<10 questions) contain
  21 questions
  (1.91%).
- The horizontal family contains 908 questions
  (82.55%).

## Performance summaries

Micro accuracy is 96.8182% for Original only and
98.4545% for Full multiview. Across all 13 observed
exact categories, unweighted macro accuracy is
92.8011% versus
98.4172%. Across nonempty
direction families it is
88.9445% versus
98.2550%.

Original-only errors decline from 35 to
17, an absolute reduction of
18 and relative reduction of
51.43%.

## Improvement concentration

The horizontal family contributes 10
of the net 18 corrected questions
(55.56%). Using the
nonexclusive union of references containing left or right yields
10 of 18
(55.56%).
The machine-readable JSON and discordant CSV report all correction and loss
categories and every affected question.

## Exploratory family uncertainty

Image-cluster bootstrap intervals were calculated only for families with at
least 30 questions
and 10
represented images. Each used 100,000 PCG64 replicates with seed 20260723 and
retained all within-family questions from each sampled image. These subgroup
intervals are exploratory and unadjusted for multiplicity.

## Conservative interpretation

The benchmark is strongly dominated by horizontal left/right references.
Full multiview's advantage survives unweighted macro averaging, but the amount
and direction of change vary across categories and families. Rare categories
contain too few questions for generalization. The 98.45% micro headline is
valid for this benchmark, but class imbalance means it primarily reflects
performance on the heavily represented horizontal directions.

## Smallest recommended manuscript revision (not applied)

1. **Evaluation Task and Metrics:** add one sentence that exact semantic
   categories and mutually exclusive direction families were audited, with
   multi-label references retained intact.
2. **Quantitative Results:** add a compact class-analysis table containing
   horizontal, vertical/forward, diagonal, locational, mixed, and u-turn/other
   support and both accuracies; put exact-category detail in supplemental
   artifacts.
3. **Discussion:** replace the current imbalance paragraph with the exact
   horizontal share, macro results, error reduction, and the limitation that
   rare classes do not establish generalization.
4. **Abstract:** revise only if space permits; the class audit does not invalidate
   98.45%, but a short “horizontal directions comprised most questions” caveat
   would improve context.
5. **Conclusion:** no additional numerical detail is necessary; retain the
   existing class-imbalance limitation.
6. Replace “Only 18 images differed, precluding broad image-level dominance.”
   with “Only 18 images differed, so broad image-level dominance was not
   established.”
