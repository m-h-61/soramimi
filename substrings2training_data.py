import random
import pandas as pd
from excel2substring import dev_generate_substrings

def dev_generate_dilution(substrings):
    # Excelファイルを読み込む
    df = pd.read_excel('medicine_library.xlsx')

    result_list = []
    # 左から2列目の各値に関数を適用して結果をリストに追加
    for value in df.iloc[:, 1]:
        substrings = dev_generate_substrings(value)
        result_list.append(substrings)

    # 左から2列目の各値をそのまま格納するリスト
    medicines = df.iloc[:, 1].tolist()
    medicine = []
    labels = []
    # enumerateを使って要素とインデックスを取得し、リストに格納
    for index, value in enumerate(medicines):
        medicine.append(value)
        labels.append(index)

    # medicine = ['housuikuroraaru', 'esutazoramu', 'fururazepamu',...]となっている
    # labels = [0, 1, 2, 3,...]となっている

    # dilution_labelという空のリストを作成
    dilution_label = []
    dilution_list = []
    for i, sample_text in enumerate(result_list):
        for _ in range(5):
            tokens = sample_text.split()
            # 5つのトークンをランダムに削除
            if len(tokens) > 5:
                delete_indices = random.sample(range(len(tokens)), 5)
                for index in sorted(delete_indices, reverse=True):
                    del tokens[index]
            modified_text = ' '.join(tokens)
            dilution_list.append(modified_text)

        # iを5回ずつdilution_labelに追加
        dilution_label.extend([i] * 5)

    return dilution_label,dilution_list
