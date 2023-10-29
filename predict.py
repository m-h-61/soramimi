import streamlit as st
import torch.nn.functional as F
from learning_model import net

@st.cache(allow_output_mutation=True)
def load_model():
        return torch.load(f)
        loaded_model = torch.load('nn_classifier.pt')

@st.cache(allow_output_mutation=True)
def predict(input_tensor):
    # 推論モードに切り替え
    net.to(device).eval()

    # 推論の実行
    with torch.no_grad():
        y = net(input_tensor.to(device).unsqueeze(0))

    # 推論ラベルを取得
    y = torch.argmax(F.softmax(y, dim=-1))

    return y


