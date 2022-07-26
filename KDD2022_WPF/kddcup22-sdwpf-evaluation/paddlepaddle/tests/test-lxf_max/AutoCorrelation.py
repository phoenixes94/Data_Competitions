import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math


def repeat(tensor, sizes):
    src_dim = tensor.dim()
    while len(sizes)>src_dim:
        tensor = tensor.unsqueeze(0)
        src_dim+=1
    src_shape = list(tensor.shape)
    tgt_shape = [0 for _ in range(src_dim)]
    for i,s in enumerate(sizes):
        tensor = paddle.concat([tensor for _ in range(s)],axis=i)
        tgt_shape[i]=src_shape[i]*s
    return tensor.reshape(tgt_shape)

def gather(x, dim, index):
    index_shape = index.shape
    index_flatten = index.flatten()
    if dim < 0:
        dim = len(x.shape) + dim
    nd_index = []
    for k in range(len(x.shape)):
        if k == dim:
            nd_index.append(index_flatten)
        else:
            reshape_shape = [1] * len(x.shape)
            reshape_shape[k] = x.shape[k]
            x_arange = paddle.arange(x.shape[k], dtype=index.dtype)
            x_arange = x_arange.reshape(reshape_shape)
            dim_index = paddle.expand(x_arange, index_shape).flatten()
            nd_index.append(dim_index)
    ind2 = paddle.transpose(paddle.stack(nd_index), [1, 0]).astype("int64")
    paddle_out = paddle.gather_nd(x, ind2).reshape(index_shape)
    return paddle_out

class AutoCorrelation(nn.Layer):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """
    def __init__(self, mask_flag=True, factor=3, scale=None, attention_dropout=0.05, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        # self.training = True
        self.training = False

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = paddle.mean(paddle.mean(corr, axis=1), axis=1)
        index = paddle.topk(paddle.mean(mean_value, axis=0), top_k, axis=-1)[1]
        weights = paddle.stack([mean_value[:, index[i]] for i in range(top_k)], axis=-1)
        # update corr
        tmp_corr = F.softmax(weights, axis=-1)
        # aggregation
        tmp_values = values
        delays_agg = paddle.zeros_like(values,dtype='float32')
        for i in range(top_k):
            pattern = paddle.roll(tmp_values, -int(index[i]), -1)
            # delays_agg = delays_agg + pattern * \
            #              (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))

            delays_agg = delays_agg + pattern * \
                            repeat(tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1), [1, head, channel, length])
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        # init_index = paddle.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
        #     .repeat(batch, head, channel, 1).to(values.device)
        init_index = repeat(paddle.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0), [batch, head, channel, 1])
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = paddle.mean(paddle.mean(corr, axis=1), axis=1)
        weights, delay = paddle.topk(mean_value, top_k, axis=-1)
        # update corr
        tmp_corr = F.softmax(weights, axis=-1)
        # aggregation
        tmp_values = repeat(values, [1, 1, 1, 2])
        delays_agg = paddle.zeros_like(values,dtype='float32')
        for i in range(top_k):
            tmp_delay = init_index + repeat(delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1),[1, head, channel, length])
            pattern = gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         repeat(tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1),[1, head, channel, length])
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = repeat(paddle.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0), [batch, head, channel, 1])
        # find top k
        top_k = int(self.factor * math.log(length))
        weights, delay = paddle.topk(corr, top_k, axis=-1)
        # update corr
        tmp_corr = F.softmax(weights, axis=-1)
        # aggregation
        tmp_values = repeat(values, [1, 1, 1, 2])
        delays_agg = paddle.zeros_like(values, dtype='float32')
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = paddle.zeros_like(queries[:, :(L - S), :],dtype='float32')
            values = paddle.concat([values, zeros], axis=1)
            keys = paddle.concat([keys, zeros], axis=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = paddle.fft.rfft(paddle.transpose(queries, [0, 2, 3, 1]), axis=-1)
        k_fft = paddle.fft.rfft(paddle.transpose(keys, [0, 2, 3, 1]), axis=-1)
        res = q_fft * paddle.conj(k_fft)
        corr = paddle.fft.irfft(res, axis=-1)

        # time delay agg
        # if self.training:
        #     V = self.time_delay_agg_training(paddle.transpose(values, [0, 2, 3, 1]), corr)
        #     V = paddle.transpose(V, [0, 3, 1, 2])
        # else:
        #     V = self.time_delay_agg_inference(paddle.transpose(values, [0, 2, 3, 1]), corr)
        #     V = paddle.transpose(V, [0, 3, 1, 2])

        V = self.time_delay_agg_full(paddle.transpose(values, [0, 2, 3, 1]), corr)
        V = paddle.transpose(V, [0, 3, 1, 2])

        if self.output_attention:
            return (V, paddle.transpose(corr, [0, 3, 1, 2]))
        else:
            return (V, None)


class AutoCorrelationLayer(nn.Layer):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).reshape([B, L, H, -1])
        keys = self.key_projection(keys).reshape([B, S, H, -1])
        values = self.value_projection(values).reshape([B, S, H, -1])

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.reshape([B, L, -1])

        return self.out_projection(out), attn

if __name__=='__main__':
    import torch
    x = torch.tensor([1, 2, 3])
    print(x.shape)
    print('='*42)
    print(x.repeat(4,2))
    print('='*42)
    print(repeat(paddle.to_tensor(x.numpy()), [4,2]))