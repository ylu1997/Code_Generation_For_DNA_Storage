import numpy as np
import numba as nb
from numba import jit, njit
from numba.experimental import jitclass

BASE_TYPE = nb.uint32
BASE_TYPE_LEN = BASE_TYPE(0).dtype.itemsize * 8


@jit
def getIndex(pos):
    return pos // BASE_TYPE_LEN


@jit
def getOffset(pos):
    return pos % BASE_TYPE_LEN


@jit
def bit_and(arr1, arr2, ans, arr_len):
    for i in range(arr_len):
        ans[i] = arr1[i] & arr2[i]


@jit
def bit_or(arr1, arr2, ans, arr_len):
    for i in range(arr_len):
        ans[i] = arr1[i] | arr2[i]


@jit
def bit_shiftH(arr1, result, arr_len, shift_amount, bit_num):
    index = getIndex(shift_amount)
    offset = getOffset(shift_amount)

    n = arr_len - index - 1
    bound_offset = (BASE_TYPE_LEN * arr_len - bit_num)

    if offset != 0:
        for i in range(n):
            result[arr_len - 1 - i] = arr1[arr_len - 1 - (index + i)] << offset
            result[arr_len - 1 - i] |= arr1[arr_len - 1 - (index + i + 1)] >> (BASE_TYPE_LEN - offset)
        result[arr_len - 1 - n] = arr1[arr_len - 1 - (index + n)] << offset
    else:
        for i in range(n):
            result[arr_len - 1 - i] = arr1[arr_len - 1 - (index + i)]
        result[arr_len - 1 - n] = arr1[arr_len - 1 - (index + n)]

    if bound_offset != 0:
        result[arr_len - 1] = (result[arr_len - 1] << bound_offset) >> bound_offset


@jit
def set_bit(arr, index, bit):
    offset = index % BASE_TYPE_LEN
    ind = index // BASE_TYPE_LEN
    val = 0x1 << offset
    if bit:
        arr[ind] = arr[ind] | val
    else:
        arr[ind] = arr[ind] & (~val)


@jit
def is_zero(arr, arr_len):
    for i in range(arr_len):
        if arr[i] != 0:
            return False
    return True


@jit
def bit_inverse(arr, arr_len):
    ans = np.bitwise_not(arr)
    for i in range(arr_len):
        arr[i] = ans[i]


@jit
def lowest_set_bit_index(bit):
    """
       Finds the index of the lowest set (1) bit in the binary representation of `bit`.

       Parameters:
       bit (int): The integer for which the lowest set bit index is to be found.

       Returns:
       int: The index of the lowest set bit. Returns 0 if `bit` is 0.

       Example:
        >>> lowest_set_bit_index(10)
        2
        >>> lowest_set_bit_index(8)
        4
        >>> lowest_set_bit_index(0)
        0
       """
    ans = 0
    if bit == 0:
        return ans
    else:
        ans += 1
    b = bit
    while b & 1 == 0:
        b = b >> 1
        ans += 1
    return ans


@jit
def arr_lowest_set_bit_index(arr, arr_len):
    """
        Finds the index of the lowest set bit in the array `arr` and returns its global bit position.

        Parameters:
        arr (list): The array of integers in which to find the lowest set bit.
        arr_len (int): The length of the array `arr`.

        Returns:
        int: The global bit position of the lowest set bit in the array.

        The function works as follows:
        1. Initializes `ans` to 0.
        2. Iterates through each element in the array `arr`.
        3. For each element, if it is not zero, it finds the lowest set bit index within that element using the function `lowest_set_bit_index`.
        4. Calculates the global bit position by adding the bit index within the element to the product of the current index `i` and `BASE_TYPE_LEN`.
        5. Returns the global bit position.
        6. If no set bit is found in the entire array, returns 0.

        Example:
        >>> arr_lowest_set_bit_index([0, 0, 4], 3)  # BASE_TYPE_LEN = 8
        19
        >>> arr_lowest_set_bit_index([0, 0, 0], 3)  # BASE_TYPE_LEN = 8
        0
        >>> arr_lowest_set_bit_index([0, 2, 3], 3)  # BASE_TYPE_LEN = 8
        10
        """
    for i in range(arr_len):
        if arr[i] != 0:
            return lowest_set_bit_index(arr[i]) + i * BASE_TYPE_LEN
    return 0


@jit
def to_string(arr, bit_len):
    s = ''
    for item in arr:
        num = item
        for i in range(BASE_TYPE_LEN):
            s += str(num % 2)
            num = num // 2
    s = s[: bit_len]
    return s


spec_bitarr = [('bit_len', nb.uint32),
               ('arr_len', nb.uint32),
               ('arr', BASE_TYPE[:]), ]


@jitclass(spec_bitarr)
class BitArray:
    def __init__(self, bit_len):
        self.bit_len = bit_len
        self.arr_len = (self.bit_len + BASE_TYPE_LEN - 1) // BASE_TYPE_LEN
        self.arr = np.zeros(self.arr_len, dtype=np.dtype(BASE_TYPE))

    def bit_and(self, arr2):
        ans = BitArray(self.bit_len)
        bit_and(self.arr, arr2.arr, ans.arr, self.arr_len)
        return ans

    def bit_shiftH(self, shift_amount):
        ans = BitArray(self.bit_len)
        bit_shiftH(self.arr, ans.arr, self.arr_len, shift_amount, self.bit_len)
        return ans

    def bit_or(self, arr2):
        ans = BitArray(self.bit_len)
        bit_or(self.arr, arr2.arr, ans.arr, self.arr_len)
        return ans

    def set_bit(self, index, val):
        if index < self.bit_len:
            set_bit(self.arr, index, val)

    def is_zero(self):
        is_zero(self.arr, self.arr_len)

    def bit_inverse(self):
        bit_inverse(self.arr, self.arr_len)

    def lowest_bit_index(self):
        return arr_lowest_set_bit_index(self.arr, self.arr_len)

    def bit_loss(self):
        l_b_i = self.lowest_bit_index()
        if l_b_i == 0:
            return 0
        else:
            return self.bit_len - l_b_i + 1

    def to_string(self):
        return to_string(self.arr, self.bit_len)


if __name__ == '__main__':
    ba = BitArray(27)
    ba.set_bit(4, 1)
    ba.set_bit(9, 1)
    print(ba.arr_len)
    print(ba.lowest_bit_index())
    print(ba.bit_loss())
    print(ba.to_string())
    pass