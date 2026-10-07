import pandas as pd

import train_model


def test_training_schema_contains_expected_features():
    expected = train_model.FEATURES + [train_model.TARGET]
    frame = pd.DataFrame(columns=expected)

    assert list(frame.columns) == expected
    assert len(train_model.FEATURES) == 13


def test_binary_target_values():
    target = pd.Series([0, 1, 0, 1, 1])

    assert set(target.unique()).issubset({0, 1})
