import numpy as np
from bert4keras.backend import keras, K
from models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding
from snippets import AutoRegressiveDecoderV2

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

# encoder = keras.models.Model(roformer.inputs, roformer.outputs[0])
# seq2seq = keras.models.Model(roformer.inputs, roformer.outputs[1])

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


def gen_synonyms(text, n=100, k=20, mask_idxs=[]):
    ''''含义： 产生sent的n个相似句，然后返回最相似的k个。
    做法：用seq2seq生成，并用encoder算相似度并排序。
    '''
    r = synonyms_generator.generate(text, n, mask_idxs=mask_idxs)
    r = [i for i in set(r) if i != text]
    r = [text] + r
    X, S = [], []
    for t in r:
        x, s = tokenizer.encode(t)
        X.append(x)
        S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    Z = roformer.predict([X, S])
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
    argsort = np.dot(Z[1:], -Z[0]).argsort()
    return [r[i + 1] for i in argsort[:k]]

print(synonyms_generator.generate(u'广州今天的天气如何', n=1))
