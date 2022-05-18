import numpy as np
import warnings

from bert4keras.snippets import AutoRegressiveDecoder, softmax


class AutoRegressiveDecoderV2(AutoRegressiveDecoder):
    """增加重复惩罚（repetition_penalty）
    """
    @staticmethod
    def wraps(default_rtype='probas', use_states=False):
        """用来进一步完善predict函数
        目前包含：1. 设置rtype参数，并做相应处理；
                  2. 确定states的使用，并做相应处理；
                  3. 设置温度参数，并做相应处理。
        """
        def actual_decorator(predict):
            def new_predict(
                self,
                inputs,
                output_ids,
                states,
                temperature=1,
                rtype=default_rtype,
                repetition_penalty=1.,
                flag=None,
                with_cache=False,
            ):
                assert rtype in ['probas', 'logits']
                if repetition_penalty != 1. and default_rtype != 'logits':
                    warnings.warn('repetition only used at logits..., repetition penalty reset to 1.')
                    repetition_penalty = 1.
                prediction = predict(self, inputs, output_ids, states, flag, with_cache)

                if not use_states:
                    prediction = (prediction, None)
                
                if default_rtype == 'logits':
                    if repetition_penalty != 1.:
                        penalty = np.take_along_axis(prediction[0], output_ids, axis=1)
                        penalty = np.where(penalty < 0, penalty * repetition_penalty, penalty / repetition_penalty)
                        np.put_along_axis(prediction[0], output_ids, penalty, axis=1)
                        
                    prediction = (
                        softmax(prediction[0] / temperature), prediction[1]
                    )
                elif temperature != 1:
                    probas = np.power(prediction[0], 1.0 / temperature)
                    probas = probas / probas.sum(axis=-1, keepdims=True)
                    prediction = (probas, prediction[1])

                if rtype == 'probas':
                    return prediction
                else:
                    return np.log(prediction[0] + 1e-12), prediction[1]

            return new_predict

        return actual_decorator
    
    def random_sample(
        self,
        inputs,
        n,
        topk=None,
        topp=None,
        states=None,
        temperature=1,
        min_ends=1,
        repetition_penalty=1.,
        with_loss=False,
        with_cache=False,
    ):
        """随机采样n个结果
        说明：非None的topk表示每一步只从概率最高的topk个中采样；而非None的topp
             表示每一步只从概率最高的且概率之和刚好达到topp的若干个token中采样。
             with_loss 为True 时，返回结果的同时会返回对应结果的loss
        返回：n个解码序列组成的list。
        """
        inputs = [np.array([i]) for i in inputs]
        output_ids = self.first_output_ids
        results = []
        flag = None  # 判断需要保留的batch index, 默认是step 0,此时需要repeat
        losses = []
        output_losses = np.empty(shape=(n, 0))
        for step in range(self.maxlen):
            probas, states = self.predict(
                inputs, output_ids, states, temperature, 'probas', repetition_penalty, flag=flag, with_cache=with_cache
            )  # 计算当前概率
            probas /= probas.sum(axis=1, keepdims=True)  # 确保归一化
            probas_original = probas  # 保存一份原始概率
            if step == 0:  # 第1步预测后将结果重复n次
                probas = np.repeat(probas, n, axis=0)
                inputs = [np.repeat(i, n, axis=0) for i in inputs]
                output_ids = np.repeat(output_ids, n, axis=0)
            if topk is not None:
                k_indices = probas.argpartition(-topk,
                                                axis=1)[:, -topk:]  # 仅保留topk
                probas = np.take_along_axis(probas, k_indices, axis=1)  # topk概率
                probas /= probas.sum(axis=1, keepdims=True)  # 重新归一化
            if topp is not None:
                p_indices = probas.argsort(axis=1)[:, ::-1]  # 从高到低排序
                probas = np.take_along_axis(probas, p_indices, axis=1)  # 排序概率
                cumsum_probas = np.cumsum(probas, axis=1)  # 累积概率
                flag_ = np.roll(cumsum_probas >= topp, 1, axis=1)  # 标记超过topp的部分
                flag_[:, 0] = False  # 结合上面的np.roll，实现平移一位的效果
                probas[flag_] = 0  # 后面的全部置零
                probas /= probas.sum(axis=1, keepdims=True)  # 重新归一化
            sample_func = lambda p: np.random.choice(len(p), p=p)  # 按概率采样函数
            sample_ids = np.apply_along_axis(sample_func, 1, probas)  # 执行采样
            sample_ids = sample_ids.reshape((-1, 1))  # 对齐形状
            if topp is not None:
                sample_ids = np.take_along_axis(
                    p_indices, sample_ids, axis=1
                )  # 对齐原id
            if topk is not None:
                sample_ids = np.take_along_axis(
                    k_indices, sample_ids, axis=1
                )  # 对齐原id
            output_ids = np.concatenate([output_ids, sample_ids], 1)  # 更新输出
            
            sample_probas = np.take_along_axis(probas_original, sample_ids, axis=1)
            output_losses = np.concatenate([output_losses, -np.log(sample_probas)], 1) # 更新loss

            is_end = output_ids[:, -1] == self.end_id  # 标记是否以end标记结束
            end_counts = (output_ids == self.end_id).sum(1)  # 统计出现的end标记
            if output_ids.shape[1] >= self.minlen:  # 最短长度判断
                flag_ = is_end & (end_counts >= min_ends)  # 标记已完成序列
                if flag_.any():  # 如果有已完成的
                    for ids in output_ids[flag_]:  # 存好已完成序列
                        results.append(ids)
                    for _loss in output_losses[flag_]:
                        losses.append(_loss)

                    flag = (flag_ == False)  # 标记未完成序列
                    inputs = [i[flag] for i in inputs]  # 只保留未完成部分输入
                    output_ids = output_ids[flag]  # 只保留未完成部分候选集
                    output_losses = output_losses[flag]
                    end_counts = end_counts[flag]  # 只保留未完成部分end计数
                    if len(output_ids) == 0:
                        break
        # 如果还有未完成序列，直接放入结果
        for ids in output_ids:
            results.append(ids)
        for loss in output_losses:
            losses.append(loss)
        
        # 返回结果
        return results if not with_loss else (results, losses)