import abc
from typing import Optional

import keras
from keras import ops

from keras_rs.src import types
from keras_rs.src.utils.pairwise_loss_utils import pairwise_comparison
from keras_rs.src.utils.pairwise_loss_utils import process_loss_call_inputs


class PairwiseLoss(keras.losses.Loss):
    """Base class for pairwise ranking losses.

    Pairwise loss functions are designed for ranking tasks, where the goal is to
    correctly order items within each list. Any pairwise loss function computes
    the loss value by comparing pairs of items within each list, penalizing
    cases where an item with a higher true label has a lower predicted score
    than an item with a lower true label.

    In order to implement any kind of pairwise loss, override the
    `pairwise_loss` method.
    """

    # TODO: Add `temperature`, `lambda_weights`.

    @abc.abstractmethod
    def pairwise_loss(self, pairwise_logits: types.Tensor) -> types.Tensor:
        raise NotImplementedError(
            "All subclasses of `keras_rs.losses.pairwise_loss.PairwiseLoss`"
            "must implement the `pairwise_loss()` method."
        )

    def compute_unreduced_loss(
        self,
        labels: types.Tensor,
        logits: types.Tensor,
        mask: Optional[types.Tensor] = None,
    ) -> tuple[types.Tensor, types.Tensor]:
        # Mask all values less than 0 (since less than 0 implies invalid
        # labels).
        valid_mask = ops.greater_equal(labels, ops.cast(0.0, labels.dtype))

        if mask is not None:
            valid_mask = ops.logical_and(valid_mask, mask)

        pairwise_labels, pairwise_logits = pairwise_comparison(
            labels=labels,
            logits=logits,
            mask=valid_mask,
            logits_op=ops.subtract,
        )

        return self.pairwise_loss(pairwise_logits), pairwise_labels

    def call(
        self,
        y_true: types.Tensor,
        y_pred: types.Tensor,
    ) -> types.Tensor:
        """
        Args:
            y_true: tensor or dict. Ground truth values. If tensor, of shape
                `(list_size)` for unbatched inputs or `(batch_size, list_size)`
                for batched inputs. If an item has a label of -1, it is ignored
                in loss computation.If it is a dictionary, it should have two
                keys: `"labels"` and `"mask"`. `"mask"` can be used to ignore
                elements in loss computation, i.e., pairs will not be formed
                with those items. Note that the final mask is an and of the
                passed mask, and `labels == -1`.
            y_pred: tensor. The predicted values, of shape `(list_size)` for
                unbatched inputs or `(batch_size, list_size)` for batched
                inputs. Should be of the same shape as `y_true`.
        """
        mask = None
        if isinstance(y_true, dict):
            if "labels" not in y_true:
                raise ValueError(
                    '`"labels"` should be present in `y_true`. Received: '
                    f"`y_true` = {y_true}"
                )

            mask = y_true.get("mask", None)
            y_true = y_true["labels"]

        y_true, y_pred, mask = process_loss_call_inputs(y_true, y_pred, mask)

        losses, weights = self.compute_unreduced_loss(
            labels=y_true, logits=y_pred, mask=mask
        )
        losses = ops.multiply(losses, weights)
        losses = ops.sum(losses, axis=-1)
        return losses
