import numpy
from Version3.BaseUtils import BitArray
from numba import jit

def index2tuple(index, basis):
    ans = []
    for value in basis[:: -1]:
        ans.append(index % value)
        index = index // value
    return ans[::-1]

def serialization(char_seq):
    upper_limit = []  # List to store (item[0] + 1) for each item in char_seq
    pattern_seq = []  # List to store all second and subsequent elements of char_seq
    index_list = [0]  # Index list, initial value is 0
    total_bits = 1  # Total number of bits, initial value is 1
    max_len = 0  # Maximum length, initial value is 0

    # Iterate through each item in char_seq
    for item in char_seq:
        print('item', item)
        upper_limit.append(item[0] + 1)  # Append (item[0] + 1) to upper_limit
        index_list.append(index_list[-1] + len(item[1:]))  # Update index_list
        pattern_seq += list(item[1:])  # Append item[1:] to pattern_seq
        # Update max_len
        max_len = max(max_len, *map(len, item[1:]))
        # Update total_bits
        total_bits *= (item[0] + 1)

    # Create a list of BitArray instances
    masks = [BitArray(total_bits) for _ in range(len(char_seq))]
    for i in range(total_bits):
        # Get the tuple form of the current index
        tup = index2tuple(i, upper_limit)
        for j in range(len(char_seq)):
            # Set the corresponding bit if condition is met
            if tup[j] < upper_limit[j] - 1:
                masks[j].set_bit(i, 1)

    shiftamount = [1]  # Initialize shiftamount
    # Calculate each shiftamount value
    for i in range(len(upper_limit) - 1):
        shiftamount.append(shiftamount[i] * upper_limit[-1 - i])
    shiftamount.reverse()  # Reverse shiftamount

    s_pattern = ''  # Initialize s_pattern
    for item in pattern_seq:
        for i in range(2):
            # Append item[i] and '\0' padding to s_pattern
            s_pattern += item[i] + '\0' * (max_len - len(item[i]))

    return masks, shiftamount, s_pattern, index_list, max_len

@jit
def extract_serialized_string(ser, ind_list, max_len, u_id, c_id, p_id):
    """
       Extracts a substring from a serialized string based on provided indices and maximum length.

       Parameters:
       ser (str): The serialized string from which to extract the substring.
       ind_list (list of int): The list of starting indices for each sequence category.
       max_len (int): The maximum length of the substring to extract.
       c_id (int): The category ID to select the starting index from ind_list.
       t_id (int): The offset within the selected category sequence.

       Returns:
       str: The extracted substring from the serialized string.

       Example:
       >>> ser = "ATG\\0\\0\\0T\\0*\\0\\0\\0A\\0T\\0"
       >>> ind_list = [0, 1, 2, 4]
       >>> max_len = 2
       >>> extract_serialized_string(ser, ind_list, max_len, 0, 0, 0)
       'AT'
       >>> extract_serialized_string(ser, ind_list, max_len, 0, 0, 1)
       'G\\x00'
       >>> extract_serialized_string(ser, ind_list, max_len, 1, 0, 1)
       'T\\x00'
       >>> extract_serialized_string(ser, ind_list, max_len, 2, 1, 0)
       'A\\x00'
       """
    index = max_len * (2 * (ind_list[u_id] + c_id) + p_id)
    p = ser[index: index + max_len]
    return p

def extract_serialized_feature(ser, ind_list, max_len, c_id, t_id, p_id):
    """
    Extracts a feature from a serialized feature sequence based on provided indices.

    Parameters:
    ser (list of lists): The serialized feature sequences.
    ind_list (list of int): The list of starting indices for each sequence category.
    c_id (int): The category ID to select the starting index from ind_list.
    t_id (int): The offset within the selected category sequence.
    p_id (int): The position ID within the selected sequence to extract the feature.

    Returns:
    The extracted feature from the serialized feature sequence.

    Example:
    >>> ser = "ATG\\0\\0\\0T\\0*\\0\\0\\0A\\0T\\0"
    >>> ind_list = [0, 1, 2, 4]
    >>> extract_serialized_feature(ser, ind_list, 0, 0, 1)
    'T'
    """
    index = max_len * ( 2 * ind_list[c_id] + t_id)
    p = ser[index + p_id]
    return p


@jit
def str_len(s):
    ans = 0
    for i in s:
        if i == '\0':
            break
        ans += 1
    return ans

def symmetrization(c_seq):
    ans = []
    for item in c_seq:
        tmp = []
        tmp.append(item[0] * 2)
        for c in item[1:]:
            tmp.append(c)
            c_tmp = (c[1], c[0])
            if c_tmp not in tmp[1:]:
                tmp.append(c_tmp)
        ans.append(tmp)
    return ans

if __name__ == '__main__':
    charSeq = [(2, ('AT', 'G')),
               (3, ('', 'T')),
               (4, ('*', ''), ('A', 'T')),
               (1, ('*','*'))]
    msks, s_a, p_s, i_l, m_l = serialization(charSeq)
    print(symmetrization(charSeq))
    print('masks: ',msks)
    print('shiftamount: ', s_a)
    print('patterns: ',p_s)
    print('index list: ', i_l)
    print(m_l)
    print('************')
    print(extract_serialized_feature(p_s, i_l, m_l, 1, 0, 0))
    print('**')
    for i in '\0\0\0\0\0':
        print(+1)
    # print(index2tuple(11, [3,4,5]))