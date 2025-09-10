import json
import torch
import torch.nn as nn

class TokenSemanticGatedFusion(nn.Module):
    def __init__(self, token_dim=512, sem_dim=256, hidden_dim=512):
        super().__init__()
        self.sem_proj = nn.Linear(sem_dim, token_dim)
        self.W_t = nn.Linear(token_dim, hidden_dim)
        self.W_s = nn.Linear(token_dim, hidden_dim)
        self.W_t_ = nn.Linear(token_dim, hidden_dim)
        self.W_s_ = nn.Linear(token_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, token_vecs, sem_matrix):
        sem_proj = self.sem_proj(sem_matrix)
        attn = torch.softmax(token_vecs @ sem_proj.T, dim=-1)
        sem_aligned = attn @ sem_proj
        gate = self.sigmoid(self.W_t(token_vecs) + self.W_s(sem_aligned))
        fused = gate * self.tanh(self.W_t_(token_vecs) + self.W_s_(sem_aligned))
        return fused

# -----------------------------
# 处理列表形式 JSON
# -----------------------------
def process_json_list(json_path, save_path, hidden_dim=512):
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)  # data_list 是一个 list

    for sample in data_list:
        token_vecs = torch.tensor(sample['token_vectors'], dtype=torch.float32)
        if 'semantic_matrix' in sample:
            sem_matrix = torch.tensor(sample['semantic_matrix'], dtype=torch.float32)
        else:
            sem_matrix = token_vecs.mean(dim=0, keepdim=True)

        token_dim = token_vecs.size(1)
        sem_dim = sem_matrix.size(1)

        fusion_model = TokenSemanticGatedFusion(token_dim, sem_dim, hidden_dim)
        fused_output = fusion_model(token_vecs, sem_matrix)
        sample['fused_vectors'] = fused_output.detach().cpu().tolist()

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)

# -----------------------------
# 测试
# -----------------------------
if __name__ == "__main__":
    json_path = "/root/autodl-tmp/shiyan/dataset/result/new_token_vectors.json"
    save_path = "/root/autodl-tmp/shiyan/dataset/result/gru.json"
    process_json_list(json_path, save_path)
    print(f"Fused vectors saved to {save_path}")









