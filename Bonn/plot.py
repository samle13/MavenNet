import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


model_path = 'models/model_epoch_50.pt'  # 替换为实际的模型文件路径

train_loader, test_loader = get_loaders(args)
model = CNN2DModel(args.num_class).to(device)
# 加载已保存的模型状态字典
checkpoint = torch.load(model_path)

# 从检查点中恢复模型状态
model.load_state_dict(checkpoint['model_state_dict'])