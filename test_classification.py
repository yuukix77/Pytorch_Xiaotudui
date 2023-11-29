import torch


outputs = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

preds = outputs.argmax(1)
print(preds)

targets = torch.tensor([0,1])
print(preds == targets)