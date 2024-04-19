import math


def range_to_bits(low: float, high: float) -> int:
    """
    Convert a range to a number of bits required to represent it
    :param low: Lower bound
    :param high: Upper bound
    :return: Number of bits required
    """

    low, high = min(low, high), max(low, high)

    return int(math.ceil(math.log2(high - low + 1)))
