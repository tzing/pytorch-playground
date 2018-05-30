class DimensionError(Exception):
    ...


def assert_last_dim(tensor, dim, allow_empty=True):
    if allow_empty and len(tensor) == 0:
        return
    elif tensor.size(-1) == dim:
        return

    message = f'required={dim}, get {tuple(tensor.size())}'
    raise DimensionError(message)
