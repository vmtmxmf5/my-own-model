import torch
import torch.nn as nn

ln = nn.LayerNorm(512)
sample = torch.randn((2, 4, 512))
print(ln(sample).size())

# 결론
# LN 가중치는 생성할 때 입력차원 기준이다
# emb_dim마다 각각 dropout을 적용?
print(ln.weight.size())
print(ln.bias.size())