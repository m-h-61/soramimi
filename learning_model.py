import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from sklearn.feature_extraction.text import CountVectorizer
from torchmetrics.functional import accuracy
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from excel2substring import dev_generate_substrings
import random
import pandas as pd
import numpy as np

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
        sample_list = " ".join(list(sample_text))
        tokens = sample_list.split()
        # 5つのトークンをランダムに削除
        if len(tokens) > 5:
            delete_indices = random.sample(range(len(tokens)), 5)
            for index in sorted(delete_indices, reverse=True):
                del tokens[index]
        modified_text = ' '.join(tokens)
        dilution_list.append(modified_text)

    # iを5回ずつdilution_labelに追加
    dilution_label.extend([i] * 5)




text_train_val, text_test,  t_train_val, t_test = train_test_split(dilution_list, dilution_label, test_size=0.3, random_state=0, stratify=np.array(dilution_label))

vectorizer = CountVectorizer(min_df=30)
bow_train_val = vectorizer.fit_transform(text_train_val).toarray()
bow_test = vectorizer.transform(text_test).toarray()

t_train_val = np.array(t_train_val)
t_test = np.array(t_test)

# tensor形式へ変換
x_train_val = torch.tensor(bow_train_val, dtype=torch.float32)
x_test = torch.tensor(bow_test, dtype=torch.float32)

t_train_val = torch.tensor(t_train_val, dtype=torch.int64)
t_test = torch.tensor(t_test, dtype=torch.int64)

dataset_train_val = torch.utils.data.TensorDataset(x_train_val, t_train_val)
dataset_test = torch.utils.data.TensorDataset(x_test, t_test)

pl.seed_everything(0)
# train と val に分割
n_train = int(len(dataset_train_val)*0.7)
n_val = int(len(dataset_train_val) - n_train)
train, val = torch.utils.data.random_split(dataset_train_val, [n_train, n_val])

# バッチサイズの定義
batch_size = 32

# Data Loader を定義
train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val, batch_size)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size)

class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(bow_test.shape[1], 8000) # 単語数が入力のサイズになる。
        self.fc2 = nn.Linear(8000, 3818) # 分類したいラベルの種類が出力のサイズになる。

    def forward(self, x):
        h = self.fc1(x)
        h = F.relu(h)
        h = self.fc2(h)
        return h

    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy(y.softmax(dim=-1), t, task='multiclass', num_classes=3818, top_k=1), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', accuracy(y.softmax(dim=-1), t, task='multiclass', num_classes=3818, top_k=1), on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', accuracy(y.softmax(dim=-1), t, task='multiclass', num_classes=3818, top_k=1), on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer

# 学習の実行
pl.seed_everything(0)
net = Net()
trainer = pl.Trainer(max_epochs=500, accelerator='gpu', deterministic=False)
trainer.fit(net, train_loader, val_loader)

# 学習済みモデルの保存
torch.save(net.state_dict(), 'nn_classifier.pt')

