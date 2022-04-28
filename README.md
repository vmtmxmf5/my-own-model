# kcc
transformer-r

## autograd
autograd 수정 버전 사용하고 싶으면 EncoderLayer class와 from import 수정할 것

+ d_trim 256 -> 512


## small 
from roberta import * ==> from small_model import *

CUDA_VISIBLE_DEVICES=1 python main.py -layer=4 -head=4 --d_model=512 -ff=2048 -max_len=512 -vocab=30522 --dataset=sst2 -B=320 --num_workers=8 --device=cuda --epochs=50 --lr=5e-5
