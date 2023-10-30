import MeCab
import re
from pykakasi import kakasi

def prd_generate_substrings(text):
    # MeCabの初期化
    tagger = MeCab.Tagger('-r /dev/null -d C:/Users/Harada/AppData/Local/Programs/Python/Python38/Lib/site-packages/unidic/dicdir')
    # tagger = MeCab.Tagger('-r /dev/null')
    # 品詞分解を行う
    result = tagger.parse(text)

    # Kakasiの設定
    kakasi_instance = kakasi()
    kakasi_instance.setMode('H', 'a')  # Hiragana to Alphabet
    kakasi_instance.setMode('K', 'a')  # Katakana to Alphabet
    kakasi_instance.setMode('J', 'a')  # Japanese Kanji to Alphabet
    kakasi_instance.setMode("r","Passport")
    # Kakasiのコンバータの設定
    conv = kakasi_instance.getConverter()

    # 名詞（名詞の全てのサブカテゴリ）の9番目の要素をローマ字に変換
    lines = result.split('\n')
    noun_pattern = r'.*名詞.*'
    pronunciation_list = []
    noun_list = []
    for line in lines:
        if line == 'EOS':
            break
        parts = re.split(r'[,\t]', line)  # タブとコンマの両方を区切り文字として使う
        if len(parts) >= 10:
            if '名詞' in parts[1] or '名詞' in parts[2] or '名詞' in parts[3]:
                _conv= conv.do(parts[9])
                pronunciation_list.append(_conv)
                noun_list.append(parts[0])

    nested_substrings = []
    for string in pronunciation_list:
        substrings = []
        length = len(string)
        for i in range(2, length + 1):
            for j in range(length - i + 1):
                substr = string[j:j + i]
                substrings.append(substr)
        nested_substrings.append(substrings)

    # 入れ子の中のリストを半角スペースで区切って文字列に変換
    nested_substrings_as_strings = [' '.join(substrings) for substrings in nested_substrings]

    return noun_list,nested_substrings_as_strings
