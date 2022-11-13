def train_ts_generator(train_dataset, model, batch_size, window_size, valid_set=False):
    """

    Parameters
    ----------
    train_dataset
    model
    batch_size
    window_size
    valid_set

    Yields
    --------
    [X_continouous, X_categorical],
    C (categorical grouping variable),
    y

    """
    while 1:
        yield train_dataset.next_batch(model, batch_size, window_size, valid_set)


def test_ts_generator(test_dataset, model, batch_size, window_size, include_all_training=False):
    while 1:
        yield test_dataset.next_batch(model, batch_size, window_size, include_all_training)
