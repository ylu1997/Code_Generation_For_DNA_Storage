from ECG_alg import augment_ecg
from ECG_mat import ECG_Matrix
import numpy as np
from CFS import serialization
from numba import jit, njit, prange, set_num_threads

import time

@jit
def upper_triangular_mapping(i, j, N):
    return j - i - 1 + (i * (2 * N - i - 1)) // 2

@jit
def batch_initialize_mat(seqs_num, depth, width, arrLength):
    ans = []
    N = seqs_num * (seqs_num - 1) // 2
    for i in range(N):
        mat = ECG_Matrix(depth, width, arrLength)
        mat.initialize()
        ans.append(mat)
    return ans



@njit(parallel = True)
def batch_augment_ecg_4_codebook(seqs, augAmount, mats, startRow, p_s, i_l, m_l, msks, s_a):
    N = len(seqs)

    for i in prange(N):
        for j in prange(i + 1, N):
            s1 = seqs[i]
            s2 = seqs[j]
            mat_index = upper_triangular_mapping(i, j, N)
            mat = mats[mat_index]
            if not mat.is_dist:
                # print(s1, s2)
                _, next_start = augment_ecg(s1, s2, augAmount, mat,
                                            startRow, s_a, p_s, i_l, m_l, msks)
                _.is_distinguished(next_start)
    return mats, (startRow + augAmount) % (mats[0].row_num)

@jit
def is_stop(mats):
    for item in mats:
        if not item.is_dist:
            return False
    return True

@njit(parallel=True)
def add_tail(codebook, increment_option):
    for i in prange(len(codebook)):
        codebook[i] += increment_option[np.random.randint(0, len(increment_option))]

@jit
def cal_total_loss(mats):
    return sum([item.loss() for item in mats])

@jit
def codebook_copy(source, target):
    for i in range(len(source)):
        target[i] = source[i]

@jit
def mats_copy(source, target):
    for i in range(len(source)):
        target[i] = source[i].clone()

def cgg(booksize, increment_option, dep, wid, arrlen, p_s, i_l, m_l, msks, s_a):
    ans = ['' for i in range(booksize)]
    add_tail(ans, increment_option)
    mats = batch_initialize_mat(booksize, dep, wid, arrlen)
    startRow = 0
    print('start')

    test_step = 1
    test_num = 1

    trial_code = [item for item in ans]
    trial_mats = [item.clone() for item in mats]

    best_ans = [item for item in ans]
    best_mats = [item.clone() for item in mats]

    turns = 1
    while True:
        codebook_copy(ans, trial_code)
        mats_copy(mats, trial_mats)
        print(turns, '-th addition')
        # print(mats[0].to_string())
        min_loss = float('inf')

        for _ in range(test_num):
            print(' ', _, '-th test')
            # start trial
            trial_row = startRow
            for __ in range(test_step):
                trial_mats, trial_row = batch_augment_ecg_4_codebook(trial_code, len(increment_option[0]),
                                                                     trial_mats, trial_row, p_s, i_l, m_l, msks, s_a)
                add_tail(trial_code, increment_option)
            ls = cal_total_loss(trial_mats)
            # print(ls)
            if ls < min_loss:
                min_loss = ls
                best_ans = [item for item in trial_code]
                best_mats = [item.clone() for item in trial_mats]

            trial_code = [item for item in ans]
            trial_mats = [item.clone() for item in mats]
        ans = [item for item in best_ans]
        mats = [item.clone() for item in best_mats]
        # print(mats[0].to_string())
        startRow = trial_row
        print(ans[0])
        # mats, startRow = batch_augment_ecg_4_codebook(ans, len(increment_option[0]), mats, startRow, p_s, i_l, m_l, msks, s_a)
        # add_tail(ans, increment_option)
        print('xx',cal_total_loss(mats))
        turns += 1
        if is_stop(mats):
            break

    return ans



if __name__ == '__main__':

    charSeq = [(2, ('*', '*'), ('', '*'), ('*', '')), ]
    msks, s_a, p_s, i_l, m_l = serialization(charSeq)
    # seqs = ["G", "G", "C", "T", "A", "G", "C"]
    # mats = batch_initialize_mat(len(seqs), 1, 0, 3)


    set_num_threads(28)
    # p_s, i_l, m_l, msks, s_a
    # batch_augment_ecg_4_codebook(seqs, 1, mats, 0, p_s, i_l, m_l, msks, s_a)
    ans = cgg(2**10, ['A','C','T','G'], 1, 1, 3, p_s, i_l, m_l, msks, s_a)
    print(ans)
    print(len(ans[0]))