import json

import numpy as np
import pytest

from src.json_utils import numpy_json_default


def test_serializes_ndarray_as_list():
    result = json.dumps({"cm": np.array([[1, 2], [3, 4]])}, default=numpy_json_default)

    assert json.loads(result) == {"cm": [[1, 2], [3, 4]]}


def test_serializes_numpy_scalar_types():
    result = json.dumps(
        {"acc": np.float64(0.913), "n": np.int64(500)}, default=numpy_json_default
    )

    loaded = json.loads(result)
    assert loaded["acc"] == pytest.approx(0.913)
    assert loaded["n"] == 500


def test_raises_typeerror_for_unsupported_type():
    with pytest.raises(TypeError):
        numpy_json_default(object())
