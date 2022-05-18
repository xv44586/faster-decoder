"""
将ckpt 转为onnx
1.将模型保存为pb 格式，ps:注意关闭每个sess，避免后续的pb 中带有sess 中非当前graph 部分
2.将pb 模型转为onnx格式，注意输入输出中的名称，保证转换最终输出的名称与定义的一致，模型pb 文件大于2GB，注意添加--large_model，--opset 参数默认9，当前选择为12，其他值可能会有后续问题
3.模型超过2GB后，转onnx 在onnxruntime 中需要适配，暂时没搞定，所以，尽量让模型不要超过2GB

requirements:
!pip install -U tf2onnx
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import argparse
import tensorflow as tf

import sys
sys.path.append(os.path.abspath('../samantha'))

from models import build_roformer_unilm_with_cache_model, build_t5_decoder_model, build_t5_decoder_with_cache_model, build_t5_encoder_model


roformer_config_path = '/data/pretrain/chinese_roformer-sim-char-ft_L-12_H-768_A-12/bert_config.json'
roformer_checkpoint_path = '/data/pretrain/chinese_roformer-sim-char-ft_L-12_H-768_A-12/bert_model.ckpt'
t5_config_path = '/data/pretrain/chinese_t5_pegasus_base/config.json'
t5_checkpoint_path = '/data/pretrain/chinese_t5_pegasus_base/model.ckpt'


def t5encoder2onnx(config_path=t5_config_path, checkpoint_path=t5_checkpoint_path):
    encoder = build_t5_encoder_model(config_path, checkpoint_path)
    
    # save to pb
    # with tf.keras.backend.get_session() as sess:
    sess = tf.keras.backend.get_session()
    tf.saved_model.simple_save(sess, '/tmp/t5_encoder', 
                            inputs={'Encoder-Input-Token': encoder.inputs[0]}, 
                            outputs={'Encoder-Output': encoder.outputs[0]})
    print('save to pb done!')

    # convert to onnx
    os.system('python -m tf2onnx.convert --opset 13 --saved-model /tmp/t5_encoder --output t5_encoder.onnx')

    print('convert done！')

def t5decoder2onnx(config_path=t5_config_path, checkpoint_path=t5_checkpoint_path):
    decoder = build_t5_decoder_model(config_path, checkpoint_path)

    # build inputs
    decoder_inputs = {'Input-Context': decoder.inputs[0], 'Decoder-Input-Token': decoder.inputs[1]}   

    # build outputs
    decoder_outputs = {'Decoder-Output': decoder.outputs[0]}

    # save to pb
    # with tf.keras.backend.get_session() as sess:
    sess = tf.keras.backend.get_session()
    tf.saved_model.simple_save(sess, '/tmp/t5_decoder',
                                inputs=decoder_inputs, outputs=decoder_outputs)
    print('save to pb done!')

    # convert to onnx
    os.system('python -m tf2onnx.convert --opset 13 --saved-model /tmp/t5_decoder --output t5_decoder.onnx')
    print('convert done!')

def t5decoder_wich_cache_2onnx(config_path=t5_config_path, checkpoint_path=t5_checkpoint_path):
    decoder = build_t5_decoder_with_cache_model(config_path, checkpoint_path)
    # build inputs
    decoder_inputs = {'Input-Context': decoder.inputs[0], 'Decoder-Input-Token': decoder.inputs[1]}
    for i in range((len(decoder.inputs) - 2)//2):
        decoder_inputs[f'Transformer-{i}-Key-Cache-Input'] = decoder.inputs[i*2+2]
        decoder_inputs[f'Transformer-{i}-Value-Cache-Input'] = decoder.inputs[i*2+3]
    
    # build outputs
    decoder_outputs = {'Decoder-Output': decoder.outputs[0]}
    for i in range(len(decoder.outputs) - 1):
        decoder_outputs[f'Transformer-{i}-Cache-Output'] = decoder.outputs[i+1]

    # save to pb
    # with tf.keras.backend.get_session() as sess:
    sess = tf.keras.backend.get_session()
    tf.saved_model.simple_save(sess, '/tmp/t5_decoder_with_cache',
                                inputs=decoder_inputs, outputs=decoder_outputs)
    print('save to pb done!')

    # convert to onnx
    os.system('python -m tf2onnx.convert --opset 13 --saved-model /tmp/t5_decoder_with_cache --output t5_decoder_with_cache.onnx')
    print('convert done!')

def roformer_unilm_2onnx(config_path=roformer_config_path, checkpoint_path=roformer_checkpoint_path):
    roformer = build_roformer_unilm_with_cache_model(config_path, checkpoint_path)
    # build inputs
    decoder_inputs = {'Input-Token': roformer.inputs[0], 'Input-Segment': roformer.inputs[1]}
    for i in range((len(roformer.inputs) - 2)//2):
        decoder_inputs[f'Transformer-{i}-Key-Cache-Input'] = roformer.inputs[i*2+2]
        decoder_inputs[f'Transformer-{i}-Value-Cache-Input'] = roformer.inputs[i*2+3]
    
    # build outputs
    decoder_outputs = {'MLM-Activation': roformer.outputs[0]}
    for i in range(len(roformer.outputs) - 1):
        decoder_outputs[f'Transformer-{i}-Cache-Output'] = roformer.outputs[i+1]

    # save to pb
    # with tf.keras.backend.get_session() as sess:
    sess = tf.keras.backend.get_session()
    tf.saved_model.simple_save(sess, '/tmp/roformer_unilm',
                                inputs=decoder_inputs, outputs=decoder_outputs)
    print('save to pb done!')

    # convert to onnx
    os.system('python -m tf2onnx.convert --opset 13 --saved-model /tmp/roformer_unilm --output roformer_unilm.onnx')
    print('convert done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='model name to convert, t5_encoder/t5_decoder/t5_decoder_with_cache/roformer_unilm', default='t5_encoder')
    parser.add_argument('-c', '--config_path', help='model config path', default='')
    parser.add_argument('-p', '--checkpoint_path', help='model checkpoint path', default='')
    args = parser.parse_args()

    model = args.model 
    config_path = args.config_path
    checkpoint_path = args.checkpoint_path
    if model == 't5_encoder':
        config_path = config_path or t5_config_path
        checkpoint_path = checkpoint_path or t5_checkpoint_path
        t5encoder2onnx(config_path, checkpoint_path)
    elif model == 't5_decoder':
        config_path = config_path or t5_config_path
        checkpoint_path = checkpoint_path or t5_checkpoint_path
        t5decoder2onnx(config_path, checkpoint_path)
    elif model=='t5_decoder_with_cache':
        config_path = config_path or t5_config_path
        checkpoint_path = checkpoint_path or t5_checkpoint_path
        t5decoder_wich_cache_2onnx(config_path, checkpoint_path)
    elif model=='roformer_unilm':
        config_path = config_path or roformer_config_path
        checkpoint_path = checkpoint_path or roformer_checkpoint_path
        roformer_unilm_2onnx(config_path, checkpoint_path)
    
    