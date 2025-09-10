import json
import torch
import torch.nn as nn
from torchcrf import CRF
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ----------------------
# 1. Dataset
# ----------------------
class FusedNERDataset(Dataset):
    def __init__(self, json_path, label2id):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data_list = json.load(f)
        self.label2id = label2id

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        fused_vectors = torch.tensor(sample['fused_vectors'], dtype=torch.float32)
        labels = torch.tensor([self.label2id[l] for l in sample['label']], dtype=torch.long)
        # ⚠️ 对齐 fused_vectors 和 labels
        min_len = min(fused_vectors.size(0), labels.size(0))
        fused_vectors = fused_vectors[:min_len]
        labels = labels[:min_len]
        return fused_vectors, labels

# ----------------------
# 2. Collate 函数 (padding + mask)
# ----------------------
def collate_fn(batch):
    fused_vecs, labels = zip(*batch)
    fused_vecs_padded = pad_sequence(fused_vecs, batch_first=True)  # [batch, max_len, hidden_dim]
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)  # padding 用 0
    mask = torch.zeros_like(labels_padded, dtype=torch.bool)
    for i, l in enumerate(labels):
        mask[i, :len(l)] = 1
    return fused_vecs_padded, labels_padded, mask

# ----------------------
# 3. MLP + CRF
# ----------------------
class MLP_CRF_NER(nn.Module):
    def __init__(self, hidden_dim, num_labels):
        super().__init__()
        self.mlp = nn.Linear(hidden_dim, num_labels)
        self.crf = CRF(num_labels)

    def forward(self, fused_vectors, labels=None, mask=None):
        # fused_vectors: [batch, seq, hidden_dim]
        emissions = self.mlp(fused_vectors)  # [batch, seq, num_labels]
        emissions = emissions.transpose(0, 1)  # [seq, batch, num_labels]

        if labels is not None:
            labels = labels.transpose(0, 1)  # [seq, batch]
        if mask is not None:
            mask = mask.transpose(0, 1)      # [seq, batch]

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=mask)
            return loss
        else:
            return self.crf.decode(emissions, mask=mask)

# ----------------------
# 4. 标签映射
# ----------------------
labels_list = [
    'I-organization', 'I-book', 'I-game', 'B-game', 'I-scene', 'I-movie', 'I-position', 'O',
    'I-company', 'B-movie', 'B-scene', 'I-government', 'B-company', 'B-name', 'B-position',
    'I-name', 'I-address', 'B-government', 'B-book', 'B-organization', 'B-address'
]
label2id = {l:i for i,l in enumerate(labels_list)}
id2label = {i:l for l,i in label2id.items()}

# ----------------------
# 5. 超参数
# ----------------------
json_path = "/root/autodl-tmp/shiyan/dataset/result/fused.json"
batch_size = 2
num_epochs = 5
learning_rate = 1e-3
hidden_dim = 512
num_labels = len(labels_list)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------
# 6. DataLoader
# ----------------------
dataset = FusedNERDataset(json_path, label2id)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# ----------------------
# 7. 模型 & 优化器
# ----------------------
model = MLP_CRF_NER(hidden_dim, num_labels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ----------------------
# 8. 训练循环
# ----------------------
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for fused_vectors, labels_tensor, mask in dataloader:
        fused_vectors = fused_vectors.to(device)
        labels_tensor = labels_tensor.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        loss = model(fused_vectors, labels_tensor, mask)  # shape: scalar or [batch]?
        if loss.dim() > 0:
            loss = loss.sum()  # ⚠️ 关键：求和成标量
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

# ----------------------
# 9. 预测
# ----------------------
model.eval()
with torch.no_grad():
    for fused_vectors, labels_tensor, mask in dataloader:
        fused_vectors = fused_vectors.to(device)
        mask = mask.to(device)
        pred_tags_batch = model(fused_vectors, labels=None, mask=mask)
        for pred_tags in pred_tags_batch:
            pred_labels = [id2label[i] for i in pred_tags]
            print("预测标签:", pred_labels)
