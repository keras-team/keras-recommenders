from typing import Optional

from keras import ops

from keras_rs.src import types
from keras_rs.src.api_export import keras_rs_export
from keras_rs.src.losses.pairwise_loss import PairwiseLoss
from keras_rs.src.utils.pairwise_loss_utils import apply_pairwise_op


@keras_rs_export("keras_rs.losses.PairwiseMeanSquaredError")
class PairwiseMeanSquaredError(PairwiseLoss):
    """Computes pairwise hinge loss between true labels and predicted scores.

    This loss function is designed for ranking tasks, where the goal is to
    correctly order items within each list. It computes the loss by comparing
    pairs of items within each list, penalizing cases where an item with a
    higher true label has a lower predicted score than an item with a lower
    true label.

    For each list of predicted scores `s` in `y_pred` and the corresponding list
    of true labels `y` in `y_true`, the loss is computed as follows:

    ```
    loss = sum_{i} sum_{j} I(y_i > y_j) * (s_i - s_j)^2
    ```

    where:
      - `y_i` and `y_j` are the true labels of items `i` and `j`, respectively.
      - `s_i` and `s_j` are the predicted scores of items `i` and `j`,
        respectively.
      - `I(y_i > y_j)` is an indicator function that equals 1 if `y_i > y_j`,
        and 0 otherwise.
      - `(s_i - s_j)^2` is the squared difference between the predicted scores
        of items `i` and `j`, which penalizes discrepancies between the
        predicted order of items relative to their true order.

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
        # Since we override `compute_unreduced_loss`, we do not need to
        # implement this method.
        pass

    def compute_unreduced_loss(
        self,
        labels: types.Tensor,
        logits: types.Tensor,
        mask: Optional[types.Tensor] = None,
    ) -> tuple[types.Tensor, types.Tensor]:
        """Override `PairwiseLoss.compute_unreduced_loss` since pairwise weights
        for MSE are computed differently.
        """
        batch_size, list_size = ops.shape(labels)

        # Mask all values less than 0 (since less than 0 implies invalid
        # labels).
        valid_mask = ops.greater_equal(labels, ops.cast(0.0, labels.dtype))

        if mask is not None:
            valid_mask = ops.logical_and(valid_mask, mask)

        # Compute the difference for all pairs in a list. The output is a tensor
        # with shape `(batch_size, list_size, list_size)`, where `[:, i, j]`
        # stores information for pair `(i, j)`.
        pairwise_labels_diff = apply_pairwise_op(labels, ops.subtract)
        pairwise_logits_diff = apply_pairwise_op(logits, ops.subtract)
        valid_pair = apply_pairwise_op(valid_mask, ops.logical_and)
        pairwise_mse = ops.square(pairwise_labels_diff - pairwise_logits_diff)

        # Compute weights.
        pairwise_weights = ops.ones_like(pairwise_mse)
        # Exclude self pairs.
        pairwise_weights = ops.subtract(
            pairwise_weights,
            ops.tile(ops.eye(list_size, list_size), (batch_size, 1, 1)),
        )
        # Include only valid pairs.
        pairwise_weights = ops.multiply(
            pairwise_weights, ops.cast(valid_pair, dtype=pairwise_weights.dtype)
        )

        return pairwise_mse, pairwise_weights
