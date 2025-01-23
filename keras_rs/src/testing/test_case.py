import os
import unittest
from typing import Any, Dict

import keras
import numpy as np
import tensorflow as tf

from keras_rs.src import types


class TestCase(tf.test.TestCase, unittest.TestCase):
    """TestCase class for all Keras Recommenders tests."""

    def setUp(self) -> None:
        super().setUp()
        keras.config.disable_traceback_filtering()
        keras.utils.clear_session()

    def assertAllClose(
        self,
        actual: types.Tensor,
        desired: types.Tensor,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        msg: str = "",
    ) -> None:
        """Verify that two tensors are close in value element by element.

        Args:
          actual: Actual tensor, the first tensor to compare.
          desired: Expected tensor, the second tensor to compare.
          atol: Absolute tolerance.
          rtol: Relative tolerance.
          msg: Optional error message.
        """
        if not isinstance(actual, np.ndarray):
            actual = keras.ops.convert_to_numpy(actual)
        if not isinstance(desired, np.ndarray):
            desired = keras.ops.convert_to_numpy(desired)
        np.testing.assert_allclose(
            actual, desired, atol=atol, rtol=rtol, err_msg=msg
        )

    def assertAllEqual(
        self, actual: types.Tensor, desired: types.Tensor, msg: str = ""
    ) -> None:
        """Verify that two tensors are equal in value element by element.

        Args:
          actual: Actual tensor, the first tensor to compare.
          desired: Expected tensor, the second tensor to compare.
          msg: Optional error message.
        """
        if not isinstance(actual, np.ndarray):
            actual = keras.ops.convert_to_numpy(actual)
        if not isinstance(desired, np.ndarray):
            desired = keras.ops.convert_to_numpy(desired)
        np.testing.assert_array_equal(actual, desired, err_msg=msg)

    def run_model_saving_test(
        self,
        cls: Any,
        init_kwargs: Dict[Any, Any],
        input_data: Any,
        atol: float = 1e-6,
        rtol: float = 1e-6,
    ) -> None:
        """Save and load a model from disk and assert output is unchanged."""
        model = cls(**init_kwargs)
        model_output = model(input_data)
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path, save_format="keras_v3")
        restored_model = keras.models.load_model(path)

        # # Check that output matches.
        restored_output = restored_model(input_data)
        self.assertAllClose(model_output, restored_output, atol=atol, rtol=rtol)
