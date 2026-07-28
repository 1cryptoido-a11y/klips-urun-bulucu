# Figure-first model ranking benchmark

## Scope

The benchmark contains 30 catalog products: five each from necklaces, earrings,
rings, bracelets, anklets, and hair clips. It covers hearts, stars, flowers,
leaves, evil eyes, butterflies, animals, crowns, moons, and sea motifs.

Each query is a deterministic shop-photo simulation rather than the unchanged
catalog image. It changes scale, rotation, light, JPEG quality, and adds a
display-card/background composition.

## Result

| Metric | Before | Figure-first |
| --- | ---: | ---: |
| Correct model/figure at rank 1 | 76.7% | 83.3% |
| Correct model/figure in top 3 | 86.7% | 86.7% |
| Exact barcode at rank 1 | 60.0% | 60.0% |
| Exact barcode in top 5 | 86.7% | 86.7% |

Exact-barcode accuracy is intentionally not the primary metric because multiple
barcodes can use the same model image or differ only by color. The production
requirement is model identity first.

## Ranking order

1. Multi-prompt visual figure fingerprint.
2. Object geometry and instance structure from grayscale/object and DINO views.
3. Stones, finish, and color as supporting evidence.

The figure fingerprint uses the response pattern across 30 motif prompts. It
does not depend on a single forced label, preventing one uncertain label from
overriding the complete visual structure.

## Regression set

Real shop photos retained the expected behavior for key pendant `126476`, eye
pendant model `073538`/`119100`, layered necklace `113070`, evil-eye bracelet
`060015`, and earrings `120346`.
