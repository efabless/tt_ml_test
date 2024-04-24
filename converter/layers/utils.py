import math


def range_to_bits(low: float, high: float, scale = 0.1, zero = 127) -> int:
    """
    Convert a range to a number of bits required to represent it
    :param low: Lower bound
    :param high: Upper bound
    :return: Number of bits required
    """

    low, high = min(low, high), max(low, high)

    adj_high = high / scale + zero
    adj_low = low / scale + zero

    return int(math.ceil(math.log2(adj_high - adj_low + 1)))
