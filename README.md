# 加速decoder 解码
通过添加 attention cache 并转换为 onnx，加速线上解码速度。当前 repo 以[simbert](https://github.com/ZhuiyiTechnology/simbert) 和[t5-pegasus](https://github.com/ZhuiyiTechnology/t5-pegasus) 为例。

# 使用方法
## t5-pegasus
1. 首先将t5_encoder/t5_decoder 转为onnx
```shell
python convert2onnx.py -m t5_encoder -c /data/pretrain/chinese_t5_pegasus_base/config.json -p /data/pretrain/chinese_t5_pegasus_base/model.ckpt

python convert2onnx.py -m t5_decoder_with_cache -c /data/pretrain/chinese_t5_pegasus_base/config.json -p /data/pretrain/chinese_t5_pegasus_base/model.ckpt
```

2. 测试
```
python test_t5_with_cache_onnx.py
```

## simbert(roformer_unilm)
1. 将roformer 转为onnx
```shell
python convert2onnx.py -m roformer_unilm -c /data/pretrain/chinese_roformer-sim-char-ft_L-12_H-768_A-12/bert_config.json -p /data/pretrain/chinese_roformer-sim-char-ft_L-12_H-768_A-12/bert_model.ckpt
```

2. 测试
```shell
python test_roformer_with_cache_onnx.py
```

# 主要思路
对attention 进行cache，可以避免重复计算之前结果，降低计算量；tensorflow 对显卡是全占用，而转为onnx 可以只占用模型对应大小的显存，避免显存浪费。

# 实现
对attention 层的k/v 进行cache，并在进行attention 计算前，将cache 中的k/v 与当前的k/v 进行拼接，此外，增加cache 后，q 为当前时刻，需要修改其对应的position 信息，并取消attention mask。除了模型层面的改动，解码函数上也需要进行调整，即解码时每次输入改为当前时刻的token/output-id，同时拼接k/v cache，为下个时刻计算做准备