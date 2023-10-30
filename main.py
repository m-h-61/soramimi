import pickle
import streamlit as st
import torch.nn.functional as F
import re
import openpyxl
import torch
import requests
from sklearn.feature_extraction.text import CountVectorizer
from convert import prd_generate_substrings

@st.cache(allow_output_mutation=True)
def load_model():
        file_path = "model.pkl2"
        with open(file_path, "rb") as file:
        loaded_data = pickle.load(file)
        return loaded_data　

def predict(input_tensor):
    net = load_model()
    # 推論モードに切り替え
    net.eval()
    # 推論の実行
    with torch.no_grad():
        y = net(input_tensor)
    # 推論ラベルを取得
    y = torch.argmax(F.softmax(y, dim=-1))
    return y

def process_and_replace_nouns(text):
    # 半角スペースで区切られたサブストリングが登場した名詞順に全て入っている
    tmp_result = prd_generate_substrings(text)
    noun_list = tmp_result[0]
    nested_substrings_as_strings = tmp_result[1]
    # 半角スペースで区切られたサブストリングをテンソルにして順番に推論し、予測ラベルを出す。
    results = []
    for substrings in nested_substrings_as_strings:
        vectorizer = CountVectorizer()
        sample_bow = vectorizer.fit_transform([substrings]).toarray()
        sample_tensor = torch.tensor(sample_bow, dtype=torch.float32)
        prediction = predict(sample_tensor)
        results.append(prediction[0])
    # Excelファイルを開く
    workbook = openpyxl.load_workbook('medicine_library.xlsx')
    # 一番左のシートを選択
    sheet = workbook.active
    values = []
    for row_number in results:
        # 指定したセルの値を取得 (A列を表す列番号は1で固定)
        cell_value = sheet.cell(row=row_number + 2, column=1).value
        values.append(cell_value)
    # このとき、valuesには、元のテキストのうち、名詞と判断されたものをアルファベットにして、テンソルにして、推論させてラベルが出てそのラベルに対応する薬品名がリストになっている。
    # つまり、ここから元のテキストリストnoun_listと、valuesリストの値を置き換えて表示すれば、置き換え表示完了！
    # 名詞を正規表現を使用して抽出
    nouns = re.findall(r'\b\w+\b', input_text)  # 単語の境界を使った単語の抽出
    # 抽出した名詞と noun_list の一致を探し、一致した名詞を置き換え
    for i in range(len(noun_list)):
        replace_text = input_text.replace(noun_list[i], values[i])
    return replace_text



if __name__ == '__main__':
    st.title('なんでも薬の名前に空耳する薬剤師に何か言ってみて。')
    # ユーザーが入力するテキストボックス
    input_text = st.text_area('（テキストボックスに文を入れてね！）', '')

    # 変換ボタン
    if st.button('いざ！'):
        replace_result = process_and_replace_nouns(input_text)
        st.write('変換結果↓')
        st.write(replace_result)
        
