# Corrected reference semantics, version 2

Author adjudication date: 2026-07-22.

The frozen still-image manifest, video question file, and all inference outputs remain unchanged. This versioned overlay supplies explicit semantic types for previously list-valued references and one corrected multi-direction reference. Stored predictions are rescored; Gemini is not called.

| Identifier | Question | Old reference | New reference | Semantic type | Reason |
|---|---|---|---|---|---|
| IMG_3391.JPG | Where is 4th Floor? | `{straight-right, straight}` | `{straight, straight-right}` | alternatives | Coarse and fine discretizations of one direction are admissible. |
| IMG_3393.JPG | Where is 4th floor? | `{straight-right, straight}` | `{straight, straight-right}` | alternatives | Coarse and fine discretizations of one direction are admissible. |
| IMG_3397.JPG | Where is 4th floor? | `{straight-right, straight}` | `{straight, straight-right}` | alternatives | Coarse and fine discretizations of one direction are admissible. |
| IMG_3572.JPG | Where is Mens Lockers? | `{left, right}` | `{left, right}` | joint_required | The bidirectional sign requires both directions. |
| IMG_3573.JPG | Where is Mens Lockers? | `{left, right}` | `{left, right}` | joint_required | The bidirectional sign requires both directions. |
| IMG_3574.JPG | Where is Mens Lockers? | `{left, right}` | `{left, right}` | joint_required | The bidirectional sign requires both directions. |
| IMG_3610.JPG | Where is the restroom? | `{left, right}` | `{left, right}` | joint_required | Both visible restroom directions are required. |
| IMG_3412.JPG | Where is LALLY 02? | `down` | `{right, down}` | joint_required | The adjudicated route requires both directions. |

The author separately confirmed scalar semantics for IMG_3412 Men's Restroom; all Design Lab/JEC 3232 formulations in IMG_3483--IMG_3485; and IMG_3769 Exit Here. Their reference values did not change, so they are recorded in the manifest as confirmed scalar adjudications rather than reference modifications.
