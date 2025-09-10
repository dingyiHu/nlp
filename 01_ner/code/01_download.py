import os
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import AutoTokenizer, AutoModelForCausalLM
  
# 设置 Hugging Face 的镜像源（可选，用于国内网络加速）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 可选的镜像源

# 设置模型下载的缓存目录为 'autodl-tmp'
cache_dir = os.path.expanduser('/root/autodl-tmp/shiyan/model')

# tokenizer = AutoTokenizer.from_pretrained("THUDM/GLM-4-9B-0414",cache_dir=cache_dir)
# model = AutoModelForCausalLM.from_pretrained("THUDM/GLM-4-9B-0414",cache_dir=cache_dir)
from transformers import AutoModel, AutoTokenizer

# 下载并加载 bert-base-chinese 模型和分词器
model = AutoModel.from_pretrained("bert-base-chinese")
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# 保存模型和分词器到本地
model.save_pretrained("./bert-base-chinese",cache_dir=cache_dir)
tokenizer.save_pretrained("./bert-base-chinese",cache_dir=cache_dir)

print("BERT-base-chinese 模型和分词器已下载并保存到本地 ./bert-base-chinese 目录")