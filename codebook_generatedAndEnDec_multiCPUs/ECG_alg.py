from numba import jit, njit, prange
from ECG_mat import ECG_Matrix
from BaseUtils import BitArray
from CFS import extract_serialized_string, serialization, str_len

@jit
def index_to_row_col(index1, index2, rowStart, colStart, rowNum):
    bias = index2 - index1
    i = (index1 + index2 + abs(bias)) // 2
    row = i + rowStart
    col = bias + colStart
    return row % rowNum, col

@jit
def is_case(s1, s2, p1, p2):
    if str_len(s1) == str_len(p1) and str_len(s2) == str_len(p2):
        for i in range(str_len(s1)):
            c_p1 = p1[i]
            c_s1 = s1[i]
            if c_p1 != '*':
                if c_p1 == '=':
                    if i <= str_len(s2):
                        if c_s1 != s2[i]:
                            return False
                    else:
                        return False
                else:
                    if c_p1 != c_s1:
                        return False
        for j in range(str_len(s2)):
            c_p2 = p2[j]
            c_s2 = s2[j]
            if c_p2 != '*':
                if c_p2 == '=':
                    if j <= str_len(s1):
                        if c_s2 != s1[j]:
                            return False
                    else:
                        return False
                else:
                    if c_p2 != c_s2:
                        return False
        return True
    else:
        return False

@jit
def mat_update(mat:ECG_Matrix, seq1, seq2, p1, p2, index1, index2, mask,
               shiftAmount, rowStart, colStart):
    rowTarget, colTarget = index_to_row_col(index1, index2, rowStart, colStart, mat.mat.shape[0])
    index1New = index1 - str_len(p1)
    index2New = index2 - str_len(p2)
    rowSource, colSource = index_to_row_col(index1New, index2New, rowStart, colStart, mat.mat.shape[0])
    isMask = BitArray(mat.bit_len)
    if is_case(seq1, seq2, p1, p2) and (abs(index1New - index2New) <= mat.wid_win):
        isMask.bit_inverse()

    mask = isMask.bit_and(mask)

    mat.transition(rowSource, colSource, rowTarget, colTarget, mask.arr, shiftAmount)



@jit
def ecg_alg(s1, s2, startIndex, endIndex, mat: ECG_Matrix, startRow,
            s_pattern, ind_list, max_len, masks, shiftAmounts):
    wid_win = mat.wid_win  # window size = col_num / 2
    d_bp = mat.dep_back  # depth back = row_num - 1
    for i in range(startIndex, endIndex):
        d = min(i, wid_win)
        for j in range(-2 * d, 1):

            index1 = (i + (j // 2) * (j % 2))
            index2 = (i + (j // 2) * ((j + 1) % 2))
            # // 发现index1和index2计算有误，请把中间计算过程打印出来
            c1, c2 = s1[index1], s2[index2]
            row, col = index_to_row_col(index1, index2, 1 - startIndex + startRow, wid_win, d_bp + 1)
            mat.set_arr(row, col, BitArray(mat.bit_len).arr)  # Set the corresponding bit array to zero
            b_arr = BitArray(mat.bit_len)  # Initialize b_arr as zero
            if c1 == c2:
                # print('match!')
                b_arr.bit_inverse()  # Set b_arr to all ones
                mat_update(mat, c1, c2, '*', '*', index1, index2, b_arr, 0, 1 + startRow - startIndex, wid_win)
            else:
                # print('not match!')
                for i_th in range(len(ind_list) - 1):
                    mask = masks[i_th]
                    shiftAmount = shiftAmounts[i_th]
                    ch_bound = ind_list[i_th + 1]
                    for j_th in range(ch_bound):
                        # i_th is index of the pattern
                        # j_th is index of the pair in the pattern
                        # p_id = 0, 1 is the index of the sequence in the pair
                        p1 = extract_serialized_string(s_pattern, ind_list, max_len, i_th, j_th, 0)  # Extract the serialized string
                        p2 = extract_serialized_string(s_pattern, ind_list, max_len, i_th, j_th, 1)
                        # 打印指标，包括i, j, index1, index2, row, col，并标注变量名，方便调试
                        # print('Outside  i:', i, 'j:', j, 'index1:', index1, 'index2:', index2, 'row:', row, 'col:', col)
                        # print('m', shiftAmount)
                        mat_update(mat, s1[max(0, index1 - str_len(p1) + 1): index1 + 1],
                                   s2[max(0, index2 - str_len(p2) + 1): index2 + 1],
                                   p1, p2, index1, index2, mask, shiftAmount, 1 + startRow - startIndex, wid_win)
    return mat, (startRow + endIndex - startIndex) % (d_bp + 1)  # Return the updated matrix and the next start row

@jit
def augment_ecg(seq1, seq2, augAmount, mat, startRow, s_a, p_s, i_l, m_l, msks):
    widthWindow = mat.wid_win
    seq1In = seq1[max(0, len(seq1) - widthWindow - augAmount):]
    seq2In = seq2[max(0, len(seq2) - widthWindow - augAmount):]
    mat, next_start = ecg_alg(seq1In, seq2In, len(seq1In) - augAmount,
            len(seq1In), mat, startRow, p_s, i_l, m_l, msks, s_a)
    return mat, next_start


@njit(parallel = True)
def batch_seq_match(seq, codebook, residue:ECG_Matrix, s_a, p_s, i_l, m_l, msks):
    mats = [residue.clone() for i in range(len(codebook))]
    row_start = 0
    Loss = [item.cal_loss(row_start) for item in mats]
    seq_length = len(codebook[0])

    for i in prange(len(codebook)):
        item = codebook[i]
        _, row_start = augment_ecg(item, seq, seq_length, mats[i], 0, s_a, p_s, i_l, m_l, msks)
        Loss[i] = mats[i].loss()
    ans_index = Loss.index(max(Loss))
    return ans_index, mats[ans_index], Loss



if __name__ == '__main__':
    mat = ECG_Matrix(1, 3, 3)
    charSeq = [(3, ('*','*'), ("*", ''), ("", "*")),]
    msks, s_a, p_s, i_l, m_l = serialization(charSeq)
    mat.initialize()
    # ecg_alg('AGCT','AGAT', 0, 4, mat, 0, p_s, i_l, m_l, msks, s_a)
    augment_ecg('ab', 'ab', 2, mat, 0, s_a, p_s, i_l, m_l, msks)
    print(mat.to_string())
    print(msks[0].to_string())
    print(m_l)

    # mat.initialize()
    # print(mat.to_string())
    #
    # s1 = 'A'
    # s2 = 'A'
    # mat, sr = augment_ecg(s1, s2, 1, mat, 0, s_a, p_s, i_l, m_l, msks)
    # print(mat.to_string())
    # print(sr)
    # s1 = 'AG'
    # s2 = 'AT'
    # mat, sr = augment_ecg(s1, s2, 1, mat, sr, s_a, p_s, i_l, m_l, msks)
    # print(mat.to_string())
    # print(sr)
    # s1 = 'AGA'
    # s2 = 'ATA'
    # mat, sr = augment_ecg(s1, s2, 1, mat, sr, s_a, p_s, i_l, m_l, msks)
    # print(mat.to_string())
    # print(sr)
    # s1 = 'AGAG'
    # s2 = 'ATAC'
    # mat1, sr = augment_ecg(s1, s2, 1, mat, sr, s_a, p_s, i_l, m_l, msks)
    # print()
    # print(mat.to_string())
    # print(mat1.to_string())
    # print(sr)