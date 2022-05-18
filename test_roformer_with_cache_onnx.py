import numpy as np
import json

from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, string_matching
from bert4keras.layers import Input, Dropout, LayerNormalization
import onnxruntime as rt

from snippets import AutoRegressiveDecoderV2


maxlen = 64

# 模型配置
roformer_onnx_path = 'roformer_unilm.onnx'
dict_path = '/data/pretrain/chinese_roformer-sim-char-ft_L-12_H-768_A-12/vocab.txt'
config_path = '/data/pretrain/chinese_roformer-sim-char-ft_L-12_H-768_A-12/bert_config.json'

config = json.load(open(config_path))
num_hidden_layers = config['num_hidden_layers']
hidden_size = config['hidden_size']

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

roformer = rt.InferenceSession(roformer_onnx_path, providers=rt.get_all_providers())


class SynonymsGenerator(AutoRegressiveDecoderV2):
    """seq2seq解码器, 当模型含有attention cache 时，因为模型是unilm，所以第一次时使用所有inputs 进行解码，剩余时间都使用output_ids 的最后一个时刻
    """
    @AutoRegressiveDecoderV2.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states, flag=None, with_cache=False):
        token_ids, segment_ids = inputs
        token_ids = np.array(token_ids, dtype=np.float32)
        segment_ids = np.array(segment_ids, dtype=np.float32)
        output_ids = np.array(output_ids, dtype=np.float32)
        not_first = output_ids.size > 0
        if with_cache:
            output_ids = output_ids if not_first else np.array(token_ids)
            segment_ids = np.ones_like(output_ids) if not_first else np.array(segment_ids)
        else:
            output_ids = np.concatenate([token_ids, output_ids], 1)
            segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        batch_size = output_ids.shape[0]
        cache_batch_size = self.cache_outputs[0].shape[0]
        # sample 时需要调整维度
        if flag is not None and len(flag) <= cache_batch_size:
            # 保留未完成的
            self.cache_outputs = [c[flag] for c in self.cache_outputs]    
            
        elif batch_size > cache_batch_size:
            self.cache_outputs = [c.repeat(batch_size, axis=0) for c in self.cache_outputs]

        if with_cache:
            # when use cache, first time, use all inputs, else only need the last output_id
            if not_first:
                cur_inputs = [output_ids[:,-1:], segment_ids[:,-1:]] + self.cache_outputs
            else:
                cur_inputs = [output_ids, segment_ids] + self.cache_outputs
        else:
            cur_inputs = [output_ids, segment_ids]
            
        # build input/output names
        input_names = ['Input-Token', 'Input-Segment']
        output_names = ['MLM-Activation']

        if with_cache:
            # when use cache, need add cache inputs and cache outputs 
            for i in range(num_hidden_layers):
                input_names.append(f'Transformer-{i}-Key-Cache-Input')
                input_names.append(f'Transformer-{i}-Value-Cache-Input')
                output_names.append(f'Transformer-{i}-Cache-Output')

        ret = roformer.run(output_names=output_names, input_feed=dict(zip(input_names, cur_inputs)))
        output_logits = ret[0] if with_cache else ret 
        
        if with_cache:
            # build cache
            output_caches = ret[1:]
            cur_cache = []

            # copy to feed k/v cache-inputs
            for c in output_caches:
                cur_cache.extend([c, c])
            self.cache_outputs = [np.concatenate([c, o], axis=1) for c, o in zip(self.cache_outputs, cur_cache)]
        return output_logits[:,-1]

    def generate(self, text, n=1, topp=None, mask_idxs=[]):
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        self.cache_outputs = [np.empty([1, 0, hidden_size], dtype=np.float32) for _ in range(num_hidden_layers * 2)]
        output_ids = self.random_sample([token_ids, segment_ids], n, topp=topp, topk=1, with_cache=True)  # 基于随机采样
        return [tokenizer.decode(ids) for ids in output_ids]


synonyms_generator = SynonymsGenerator(
    start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen
)

print(synonyms_generator.generate('广州今天的天气如何', n=1))
