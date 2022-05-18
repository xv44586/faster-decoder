from ast import Mult
import json
import numpy as np

from bert4keras.backend import K, keras
from keras.models import Model
import tensorflow as tf
from bert4keras.layers import Input, Embedding, Masking, Dense, Lambda, LayerNormalization, Dropout, Concatenate1D, MultiHeadAttention
from bert4keras.layers import Add, FeedForward, Activation, SinusoidalPositionEmbedding
from keras import initializers, activations
from bert4keras.models import LM_Mask, T5_Base, Transformer, T5_Encoder, extend_with_language_model, extend_with_unified_language_model, NEZHA
from bert4keras.snippets import string_matching, is_string

from layers import RelativePositionEmbeddingT5, MultiHeadAttentionCache


# -- re build model
class RoFormer(NEZHA):
    """旋转式位置编码的BERT模型
    链接：https://kexue.fm/archives/8265
    """
    def apply_main_layers(self, inputs, index):
        """RoFormer的主体是基于Self-Attention的模块
        顺序：Att --> Add --> LN --> FFN --> Add --> LN
        """
        x = inputs
        z = self.layer_norm_conds[0]

        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Transformer-%d-FeedForward' % index
        attention_mask = self.compute_attention_bias(index)

        xi = x
        if self.attention_caches:
            k_cache, v_cache = self.attention_caches[attention_name]
            x_key = Concatenate1D(name=attention_name + '-Cache')([k_cache, x])
            
            position_bias = self.compute_position_bias(x_key)
            inputs = [x, x, x, position_bias]
            arguments = {'a_bias': None, 'p_bias': 'rotary'}
        else:
            position_bias = self.compute_position_bias(x)
            inputs = [x, x, x, position_bias]
            arguments = {'a_bias': None, 'p_bias': 'rotary'}
            if attention_mask is not None:
                arguments['a_bias'] = True
                inputs.insert(3, attention_mask)

        # Self Attention

        x = self.apply(
            inputs=inputs,
            layer=MultiHeadAttentionCache,
            arguments=arguments,
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            out_dim=self.hidden_size,
            key_size=self.attention_key_size,
            attention_dropout=self.attention_dropout_rate,
            kernel_initializer=self.initializer,
            name=attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % attention_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % attention_name
        )

        # Feed Forward
        xi = x
        x = self.apply(
            inputs=x,
            layer=FeedForward,
            units=self.intermediate_size,
            activation=self.hidden_act,
            kernel_initializer=self.initializer,
            name=feed_forward_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % feed_forward_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % feed_forward_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % feed_forward_name
        )

        return x

    def compute_position_bias(self, inputs=None):
        """Sinusoidal位置编码（直接返回）
        """
        if self.position_bias is None:

            x = inputs
            self.position_bias = self.apply(
                inputs=x,
                layer=SinusoidalPositionEmbedding,
                output_dim=self.attention_key_size,
                merge_mode='zero',
                name='Embedding-Rotary-Position'
            )

        return self.position_bias


class T5_Decoder(LM_Mask, T5_Base):
    """Google的T5模型（Decoder）
    """
    def __init__(self, with_lm=True, **kwargs):
        super(T5_Decoder, self).__init__(**kwargs)
        self.with_lm = with_lm

    def get_inputs(self):
        """T5的Decoder的输入为context序列和token_ids
        """
        c_in = self.apply(
            layer=Input,
            shape=(self.sequence_length, self.hidden_size),
            name='Input-Context'
        )
        x_in = self.apply(
            layer=Input,
            shape=(self.sequence_length,),
            name='Decoder-Input-Token'
        )
        return [c_in, x_in]

    def apply_embeddings(self, inputs):
        """T5的embedding只有token embedding，
        并把relative position embedding准备好，待attention使用。
        """
        c, x = inputs[:2]

        c = self.apply(
            inputs=c, layer=Masking, mask_value=0.0, name='Masked-Context'
        )
        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token'
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Decoder-Embedding-Dropout'
        )
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Decoder-Embedding-Mapping'
            )

        return [c, x]

    def apply_main_layers(self, inputs, index):
        """T5的Dencoder主体是基于Self-Attention、Cross-Attention的模块
        顺序：LN --> Att1 --> Add --> LN --> Att2 --> Add -->  LN --> FFN --> Add
        """
        c, x = inputs[:2]
        z = self.layer_norm_conds[0]

        self_attention_name = 'Decoder-Transformer-%d-MultiHeadSelfAttention' % index
        cross_attention_name = 'Decoder-Transformer-%d-MultiHeadCrossAttention' % index
        feed_forward_name = 'Decoder-Transformer-%d-FeedForward' % index
        attention_mask = self.compute_attention_bias(index)
        # build position bias
        if self.attention_caches:
            k_cache, v_cache = self.attention_caches[self_attention_name]
            x_k = Concatenate1D(name=self_attention_name + '-Cache')([k_cache, x])
            position_bias = self.compute_position_bias([x_k, c], last_one=True)  # only query last one token position bias
        else:
            position_bias = self.compute_position_bias([x, c])        

        # Self Attention
        xi = x
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            center=False,
            epsilon=1e-6,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % self_attention_name
        )

        if self.attention_caches:
            inputs = [x, x, x, position_bias[0]]
            a_bias = None
        else:                
            inputs = [x, x, x, attention_mask, position_bias[0]]
            a_bias = True

        x = self.apply(
            inputs=inputs,
            layer=MultiHeadAttention,
            arguments={
                'a_bias': a_bias,
                'p_bias': 't5_relative'
            },
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            out_dim=self.hidden_size,
            key_size=self.attention_key_size,
            use_bias=False,
            attention_scale=False,
            kernel_initializer=self.initializer,
            name=self_attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % self_attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % self_attention_name
        )

        # Cross Attention
        xi = x
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            center=False,
            epsilon=1e-6,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % cross_attention_name
        )
        x = self.apply(
            inputs=[x, c, c, position_bias[1]],
            layer=MultiHeadAttention,
            arguments={
                'a_bias': None,
                'p_bias': 't5_relative'
            },
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            out_dim=self.hidden_size,
            key_size=self.attention_key_size,
            use_bias=False,
            attention_scale=False,
            kernel_initializer=self.initializer,
            name=cross_attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % cross_attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % cross_attention_name
        )

        # Feed Forward
        xi = x
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            center=False,
            epsilon=1e-6,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % feed_forward_name
        )
        x = self.apply(
            inputs=x,
            layer=FeedForward,
            units=self.intermediate_size,
            activation=self.hidden_act,
            use_bias=False,
            kernel_initializer=self.initializer,
            name=feed_forward_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % feed_forward_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % feed_forward_name
        )

        return [c, x]

    def apply_final_layers(self, inputs):
        """剩余部分
        """
        c, x = inputs[:2]
        z = self.layer_norm_conds[0]

        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            center=False,
            epsilon=1e-6,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='Decoder-Output-Norm'
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Decoder-Output-Dropout'
        )
        x = self.apply(
            inputs=x,
            layer=Lambda,
            function=lambda x: x / self.hidden_size**0.5,
            mask=lambda i, m: m,
            name='Decoder-Output-Scale'
        )

        if self.with_lm:
            # 预测token概率部分
            if self.embedding_size != self.hidden_size:
                x = self.apply(
                    inputs=x,
                    layer=Dense,
                    units=self.embedding_size,
                    kernel_initializer=self.initializer,
                    name='Decoder-Output-Mapping'
                )
            lm_activation = 'softmax' if self.with_lm is True else self.with_lm
            if self.version == 't5.1.0':
                x = self.apply(
                    inputs=x,
                    layer=Embedding,
                    arguments={'mode': 'dense'},
                    name='Embedding-Token'
                )
                x = self.apply(
                    inputs=x,
                    layer=Activation,
                    activation=lm_activation,
                    name='Dencoder-Output-LM-Activation'
                )
            else:
                x = self.apply(
                    inputs=x,
                    layer=Dense,
                    units=self.vocab_size,
                    activation=lm_activation,
                    use_bias=False,
                    kernel_initializer=self.initializer,
                    name='Decoder-Output-LM'
                )

        return x

    def compute_attention_bias(self, inputs=None):
        """修改LM Mask的序列长度（从 self.inputs[0] 改为 self.inputs[1] ）
        """
        old_inputs = self.inputs[:]
        self.inputs = [old_inputs[1]]
        mask = super(T5_Decoder, self).compute_attention_bias(inputs)
        self.inputs = old_inputs
        return mask

    def compute_position_bias(self, inputs=None, last_one=False):
        """T5相对位置编码
        """
        if self.position_bias is None:

            x, c = inputs
            p1 = self.apply(
                inputs=[x, x],
                layer=RelativePositionEmbeddingT5,
                input_dim=32,
                output_dim=self.num_attention_heads,
                bidirectional=False,
                embeddings_initializer=self.initializer,
                name='Decoder-Embedding-Relative-Position',
                last_one=last_one
            )
            p2 = self.apply(
                inputs=[x, c],
                layer=RelativePositionEmbeddingT5,
                input_dim=32,
                output_dim=self.num_attention_heads,
                bidirectional=False,
                embeddings_initializer=self.initializer,
                name='Decoder-Embedding-Relative-Position',
                last_one=last_one
            )
            self.position_bias = (p1, p2)

        return self.position_bias


class T5(T5_Base):
    """Google的T5模型（Encoder-Decoder）
    """
    def delete_kwargs_before_encoder(self, kwargs):
        # t5-encoder 无需重复cache
        for cache_input in ['attention_caches', 'additional_input_layers']:
            if cache_input  in kwargs:
                del kwargs[cache_input]

        return kwargs

    def __init__(self, **kwargs):
        super(T5, self).__init__(**kwargs)
        kwargs['layers'] = self.layers
        e_name, d_name = 'Encoder', 'Decoder'
        if 'name' in kwargs:
            e_name = '%s_%s' % (kwargs['name'], e_name)
            d_name = '%s_%s' % (kwargs['name'], d_name)
            del kwargs['name']  # 防止重复传参
        
        self._decoder = T5_Decoder(name=d_name, **kwargs)
        kwargs = self.delete_kwargs_before_encoder(kwargs)
        self._encoder = T5_Encoder(name=e_name, **kwargs)

    def build(self, **kwargs):
        """同时构建Encoder和Decoder
        """
        
        self._decoder.build(**kwargs)
        kwargs = self.delete_kwargs_before_encoder(kwargs)
        self._encoder.build(**kwargs)
        self.encoder = self._encoder.model
        self.decoder = self._decoder.model
        self.inputs = self.encoder.inputs + self.decoder.inputs[1:]
        self.outputs = self.decoder(
            self.encoder.outputs + self.decoder.inputs[1:]
        )
        self.model = Model(self.inputs, self.outputs)


def build_transformer_model(
    config_path=None,
    checkpoint_path=None,
    model='bert',
    application='encoder',
    return_keras_model=True,
    **kwargs
):
    """根据配置文件构建模型，可选加载checkpoint权重
    """
    configs = {}
    if config_path is not None:
        configs.update(json.load(open(config_path)))
    configs.update(kwargs)
    if 'max_position' not in configs:
        configs['max_position'] = configs.get('max_position_embeddings', 512)
    if 'dropout_rate' not in configs:
        configs['dropout_rate'] = configs.get('hidden_dropout_prob')
    if 'segment_vocab_size' not in configs:
        configs['segment_vocab_size'] = configs.get('type_vocab_size', 2)

    models = {
        't5': T5,
        't5_encoder': T5_Encoder,
        't5_decoder': T5_Decoder,
        't5.1.0': T5,
        't5.1.0_encoder': T5_Encoder,
        't5.1.0_decoder': T5_Decoder,
        't5.1.1': T5,
        't5.1.1_encoder': T5_Encoder,
        't5.1.1_decoder': T5_Decoder,
        'roformer': RoFormer,
    }

    if is_string(model):
        model = model.lower()
        MODEL = models[model]
        if model.startswith('t5.1.1'):
            configs['version'] = 't5.1.1'
    else:
        MODEL = model

    application = application.lower()
    if application in ['lm', 'unilm'] and model in ['electra', 't5']:
        raise ValueError(
            '"%s" model can not be used as "%s" application.\n' %
            (model, application)
        )

    if application == 'lm':
        MODEL = extend_with_language_model(MODEL)
    elif application == 'unilm':
        MODEL = extend_with_unified_language_model(MODEL)

    transformer = MODEL(**configs)
    transformer.build(**configs)

    if checkpoint_path is not None:
        transformer.load_weights_from_checkpoint(checkpoint_path)

    if return_keras_model:
        return transformer.model
    else:
        return transformer


def build_base_model(config_path: str, checkpoint_path: str):
    # 建立并加载模型
    t5 = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='t5.1.1',
                                 return_keras_model=False, with_lm=True, name='T5')
    return t5


def build_t5_encoder_model(config_path: str, checkpoint_path: str):
    # 建立并加载模型
    t5 = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model='t5.1.1',
        return_keras_model=False,
        with_lm='linear',
        name='T5',
    )
    encoder = keras.models.Model(t5.encoder.inputs[:1], t5.encoder.outputs)
    return encoder


def build_t5_decoder_with_cache_model(config_path: str, checkpoint_path: str):
    configs = json.load(open(config_path))
    # build cache input layer 
    cache_inputs = []
    attention_caches = {}
    for i in range(configs['num_hidden_layers']):
        name = 'Decoder-Transformer-%d-MultiHeadSelfAttention' % i
        _input_layers = [Input(shape=(None, configs['hidden_size']), name=name + '-Key'), Input(shape=(None, configs['hidden_size']), name=name+'-Value')]
        cache_inputs.extend(_input_layers)
        attention_caches[name] = _input_layers

    t5 = build_transformer_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            model='t5.1.1',
            return_keras_model=False,
            with_lm='linear',
            name='T5',
            attention_caches=attention_caches,
            additional_input_layers=cache_inputs
        )
    layer_norm_outputs=[]
    for layer in t5.decoder.layers:
        if isinstance(layer, LayerNormalization):
            if not string_matching(layer.name, ['MultiHeadSelfAttention']):
                continue
            
            layer_norm_outputs.append(layer.output)
    
    decoder = keras.models.Model(t5.decoder.inputs, t5.decoder.outputs + layer_norm_outputs)
    return decoder
    
def build_t5_decoder_model(config_path: str, checkpoint_path: str):
    # 建立并加载模型
    t5 = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model='t5.1.1',
        return_keras_model=False,
        with_lm='linear',
        name='T5',
    )
    decoder = t5.decoder
    return decoder

def build_roformer_unilm_with_cache_model(config_path, checkpoint_path):
    configs = json.load(open(config_path))
    # build cache input layer 
    cache_inputs = []
    attention_caches = {}
    for i in range(configs['num_hidden_layers']):
        name = 'Transformer-%d-MultiHeadSelfAttention' % i
        _input_layers = [Input(shape=(None, configs['hidden_size']), name=name + '-Key'), Input(shape=(None, configs['hidden_size']), name=name+'-Value')]
        cache_inputs.extend(_input_layers)
        attention_caches[name] = _input_layers


    roformer = build_transformer_model(
        config_path,
        checkpoint_path,
        model='roformer',
        application='unilm',
        attention_caches=attention_caches,
        additional_input_layers=cache_inputs
    )

    layer_norm_outputs=[]
    for layer in roformer.layers:
        if isinstance(layer, LayerNormalization) or isinstance(layer, Dropout):
            if not string_matching(layer.name, ['Embedding-Dropout', 'FeedForward-Norm']):
                continue
            
            layer_norm_outputs.append(layer.output)

    roformer_unilm = keras.models.Model(roformer.inputs, roformer.outputs + layer_norm_outputs[:-1])
    return roformer_unilm
