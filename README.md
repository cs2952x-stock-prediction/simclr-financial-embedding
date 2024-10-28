# simclr-financial-embedding

## Notes:

- Consider working in log-space for the following reasons:

  - Captures multiplicative scaling of values
  - Avoids lopsided-weighting of large values/spikes
  - Model only needs to learn additive relations while implicitly capturing multiplicative ones
  - Can handle negatives implicitly (we don't need to handle that as a special case)

- Remember to only apply transformations to KPI values.

  - Should symbol embeddings be left alone? (equivariant to the identity of the stock?)
  - Should temporal features have their own transformation functions? (masking would be a random _but valid_ cyclical embedding)

- Abblation studies on the contributions of cyclical temporal features (professor's suggestion)
