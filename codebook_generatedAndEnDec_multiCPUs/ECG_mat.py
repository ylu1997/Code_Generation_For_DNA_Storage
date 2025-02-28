import numpy as np
from numba import njit, prange, jit
from numba.experimental import jitclass
import numba as nb
from BaseUtils import (BASE_TYPE, BitArray, BASE_TYPE_LEN, lowest_set_bit_index, to_string,
                       bit_or, bit_and, bit_shiftH)


@jit
def index_to_rowcol(i, j, k, num_col, num_arr):
    return k + num_arr * (j + num_col * i)


@jit
def get_entry_index(i, j, col_num, arr_len):
    return arr_len * (j + col_num * i)
@jit
def get_index(i, j, k, col_num, arr_len):
    return index_to_rowcol(i, j, k, col_num, arr_len)


@jit
def mat_to_string(self):
    s = '['
    for i in range(self.row_num):
        for j in range(self.col_num):
            rc_index = get_entry_index(i, j, self.col_num, self.arr_len)
            arr = self.mat[rc_index: rc_index + self.arr_len]

            s += to_string(arr, self.bit_len) + ' '
        s += '\n ' if i < self.row_num - 1 else ']'
    return s

spec_ecg = [('bit_len', nb.uint32),
            ('arr_len', nb.uint32),
            ('dep_back', nb.uint32),
            ('wid_win', nb.uint32),
            ('mat', BASE_TYPE[:]),
            ('row_num', nb.uint32),
            ('col_num', nb.uint32),
            ('is_dist', nb.boolean)]


@jitclass(spec_ecg)
class ECG_Matrix:
    def __init__(self, depthBackTrack, widthWindow, bitLen):
        self.bit_len = bitLen
        self.arr_len = (bitLen + BASE_TYPE_LEN - 1) // BASE_TYPE_LEN
        self.dep_back = depthBackTrack
        self.wid_win = widthWindow
        self.row_num = self.dep_back + 1
        self.col_num = self.wid_win * 2 + 1
        self.mat = np.zeros((self.row_num) * (self.col_num) * (self.arr_len),
                            dtype=np.dtype(BASE_TYPE))
        self.is_dist = self.is_distinguished(0)

    def new_arr(self):
        return np.zeros((self.arr_len), dtype=np.dtype(BASE_TYPE))

    def clone(self):
        ans = ECG_Matrix(self.dep_back, self.wid_win, self.bit_len)
        ans.mat = np.copy(self.mat)
        ans.is_dist = self.is_dist
        return ans

    def arr_loss(self, i, j):
        k = 0
        for l in range(self.arr_len):
            r = get_index(i, j, self.arr_len - l - 1, self.col_num, self.arr_len)
            if self.mat[r] != 0:
                k = self.arr_len - 1 - l
                break
        ans = lowest_set_bit_index(self.mat[get_index(i, j, k, self.col_num, self.arr_len)]) + k * BASE_TYPE_LEN
        return ans

    def loss(self):
        ans = 0
        for i in range(self.row_num):
            for j in range(self.col_num):
                ans += self.arr_loss(i, j)
        return ans / (self.row_num * self.col_num)

    def to_string(self):
        return mat_to_string(self)

    def get_arr(self, i, j):
        i = i % self.row_num
        r = get_entry_index(i, j, self.col_num, self.arr_len)
        return self.mat[r: r + self.arr_len]

    def set_arr(self, i, j, arr):
        i = i % self.row_num
        r = get_entry_index(i, j, self.col_num, self.arr_len)
        self.mat[r: r + self.arr_len] = arr

    def transition(self, row_s: int, col_s: int, row_t: int,
                   col_t: int, mask, shiftAmount: int):

        ans1 = self.new_arr()
        ans2 = self.new_arr()
        arr_s = self.get_arr(row_s, col_s)
        arr_t = self.get_arr(row_t, col_t)
        bit_and(arr_s, mask, ans1, self.arr_len)
        bit_shiftH(ans1, ans2, self.arr_len, shiftAmount, self.bit_len)
        bit_or(arr_t, ans2, ans1, self.arr_len)
        self.set_arr(row_t, col_t, ans1)

    def initialize(self):
        r = get_index(0, self.wid_win, 0, self.col_num, self.arr_len)
        self.mat[r] = 1
        self.is_distinguished(0)

    def is_distinguished(self, start_row):
        for row in range(self.row_num + 1):
            for col in range(self.col_num):
                flag = self.get_arr(start_row - row, col)
                if flag!=0:
                    self.is_dist = False
                    return False
        self.is_dist = True
        return True

if __name__ == '__main__':
    m = ECG_Matrix(2, 2, 9)
    print(m.is_dist)
    print(m.to_string())

    m.mat[0] = 1
    m.mat[2] = 2
    ba = BitArray(9)
    for i in range(9):
        ba.set_bit(i, 1)
    print(ba.to_string())
    # print(m.get_index(0,1,0))
    # print(m.loss())
    m.initialize()
    print(m.loss())
    m.transition(0, 1, 1,2, ba.arr, 3)
    m.transition(1, 2, 0, 1, ba.arr, 3)
    print(m.to_string())
    # m.loss()
    print(m.is_distinguished(0))
    pass
