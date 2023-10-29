import random
import pandas as pd

# サブストリングを生成する関数
def dev_generate_substrings(input_string):
    substrings = set()
    for i in range(2, len(input_string) + 1):
        for j in range(len(input_string) - i + 1):
            substring = input_string[j:j + i]
            substrings.add(substring)
    substrings_list = " ".join(list(substrings))
    return substrings
