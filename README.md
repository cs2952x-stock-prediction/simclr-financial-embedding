# simclr-financial-embedding

## Notes:

- Consider working in log-space for the following reasons:

  - Captures multiplicative scaling of values
  - Avoids lopsided-weighting of large values/spikes
  - Model only needs to learn additive relations while implicitly capturing multiplicative ones
  - Can handle negatives implicitly (we don't need to handle that as a special case)

- Remember to only apply transformations to KPI values.

  - Should symbol embeddings be left alone? (invariant to the identity of the stock?)
  - Should temporal features have their own transformation functions? (masking would be a random _but valid_ cyclical embedding)

- Ablation studies on the contributions of cyclical temporal features (professor's suggestion)

- Possible baseline comparisons
  - compare to the results of Denoising Financial Data paper
  - compare to training on the downstream task directly (use probe on encoder and allow the gradients to back-propagate)
  - compare to classic autoregression methods (linear/polynomial/ARIMA)

## Possible Embedding Models:

### Simple LSTM

#### Pipeline

- Feed through a linear layer to get an embedding of the data.
- Feed the embedding into the lstm
- Apply a final linear layer to the lstm output

### CNN-only

#### Pipeline

1.

## To-Do
