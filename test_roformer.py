import numpy as np

from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding

from snippets import AutoRegressiveDecoderV2
from models import build_transformer_model

maxlen = 64

# 模型配置
config_path = '/data/pretrain/chinese_roformer-sim-char-ft_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/data/pretrain/chinese_roformer-sim-char-ft_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/data/pretrain/chinese_roformer-sim-char-ft_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

# 建立加载模型
roformer = build_transformer_model(
    config_path,
    checkpoint_path,
    model='roformer',
    application='unilm',
)
roformer.summary()

class SynonymsGenerator(AutoRegressiveDecoderV2):
    """seq2seq解码器
    """
    @AutoRegressiveDecoderV2.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, step,flag=None, with_cache=False):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        ret = self.last_token(roformer).predict([token_ids, segment_ids])
        return ret

    def generate(self, text, n=1, topp=0.95, mask_idxs=[]):
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        for i in mask_idxs:
            token_ids[i] = tokenizer._token_mask_id
        output_ids = self.random_sample([token_ids, segment_ids], n, topk=1, topp=topp)  # 基于随机采样
        return [tokenizer.decode(ids) for ids in output_ids]


synonyms_generator = SynonymsGenerator(
    start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen
)

print(synonyms_generator.generate(u'广州今天的天气如何', n=1))
