from keras import ops

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export
from keras_rs.src.losses.pairwise_loss import PairwiseLoss


@keras_rs_export("keras_rs.losses.PairwiseHingeLoss")
class PairwiseHingeLoss(PairwiseLoss):
    """Computes pairwise hinge loss between true labels and predicted scores.

    This loss function is designed for ranking tasks, where the goal is to
    correctly order items within each list. It computes the loss by comparing
    pairs of items within each list, penalizing cases where an item with a
    higher true label has a lower predicted score than an item with a lower
    true label.

    For each list of predicted scores `s` in `y_pred` and the corresponding list
    of true labels `y` in `y_true`, the loss is computed as follows:

    ```
    loss = sum_{i} sum_{j} I(y_i > y_j) * max(0, 1 - (s_i - s_j))
    ```

    where:
      - `y_i` and `y_j` are the true labels of items `i` and `j`, respectively.
      - `s_i` and `s_j` are the predicted scores of items `i` and `j`,
        respectively.
      - `I(y_i > y_j)` is an indicator function that equals 1 if `y_i > y_j`,
        and 0 otherwise.
      - `max(0, 1 - (s_i - s_j))` is the hinge loss, which penalizes cases where
        the score difference `s_i - s_j` is not sufficiently large when
        `y_i > y_j`.

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.
    """

    def pairwise_loss(self, pairwise_logits: types.Tensor) -> types.Tensor:
        return ops.relu(ops.subtract(ops.array(1), pairwise_logits))
