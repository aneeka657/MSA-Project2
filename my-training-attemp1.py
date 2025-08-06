import os
# Use GPUs 2 and 3 (the free ones!)
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import numpy as np
import tensorflow as tf
from scipy.signal import find_peaks
from scipy.ndimage import median_filter, gaussian_filter1d, filters
from tensorflow.keras import backend
import mir_eval
import librosa
import math

# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Added: Allow GPU memory growth

global_frame_size = 0.5


def get_spectral_mask(n_batch, seq_len, n_head, n_fct=1, n_mel=80, n_chroma=12):
    # Within only
    mel_mask = tf.concat(
        [tf.ones([n_mel, n_mel], dtype=tf.bool), tf.zeros([n_mel, n_chroma], dtype=tf.bool)],
        axis=1,
    ) # [80, 80+12]
    chroma_mask = tf.concat(
        [tf.zeros([n_chroma, n_mel], dtype=tf.bool), tf.ones([n_chroma, n_chroma], dtype=tf.bool)],
        axis=1,
    ) # [12, 80+12]
    mask = tf.concat([mel_mask, chroma_mask], axis=0) # [80+12, 80+12]
    mask = tf.tile(mask[tf.newaxis, :, :], [n_head, 1, 1]) # [h, 80+12, 80+12]

    # FCT see all
    mask = tf.pad(mask, [(0, 0), (0, 0), (n_fct, 0)], constant_values=False) # [h, 80+12, 1+80+12]
    mask = tf.pad(mask, [(0, 0), (n_fct, 0), (0, 0)], constant_values=True) # [h, 1+80+12, 1+80+12]
    mask = tf.tile(mask[tf.newaxis, :, :, :], [n_batch*seq_len, 1, 1, 1]) # [bn, h, 1+80+12, 1+80+12]
    mask = tf.concat(tf.split(mask, n_head, axis=1), axis=0) # [hbn, 1, 1+80+12, 1+80+12]
    return tf.squeeze(mask, axis=1) # [hbn, 1+80+12, 1+80+12]


def get_temporal_mask(valid_len, max_len, n_heads=8):
    def partition_len(l, div):
        assert div < l
        return [l // div + (1 if x < l % div else 0) for x in range(div)]

    valid_len = tf.cast(valid_len, tf.float32)
    b_seq = []
    for l in valid_len:
        h_seq = []
        for i_h in range(n_heads):
            div = i_h // 2 + 1
            ids = tf.concat([tf.ones([part], dtype=tf.int32)*i for i, part in enumerate(partition_len(l, div))], axis=0)
            ids = tf.pad(ids, [(0, max_len - l)], constant_values=-1) # [n]
            h_seq.append(ids)
        b_seq.append(tf.stack(h_seq)) # [h, n]
    b_seq = tf.stack(b_seq) # [b, h, n]
    mask = (b_seq[:, :, :, tf.newaxis] == b_seq[:, :, tf.newaxis, :]) # [b, h, n, n]
    mask = tf.concat(tf.split(mask, n_heads, axis=1), axis=0) # [hb, 1, n, n]
    return tf.squeeze(mask, axis=1) # [hb, n, n]


def peak_picking_MSAF(x, median_len=9, offset_rel=0.05, sigma=4.0):
    """Peak picking strategy following MSFA using an adaptive threshold (https://github.com/urinieto/msaf)
    # median_len=9, offset_rel=0.05, sigma=4.0

    Notebook: C6/C6S1_PeakPicking.ipynb

    Args:
        x (np.ndarray): Input function
        median_len (int): Length of media filter used for adaptive thresholding (Default value = 16)
        offset_rel (float): Additional offset used for adaptive thresholding (Default value = 0.05)
        sigma (float): Variance for Gaussian kernel used for smoothing the novelty function (Default value = 4.0)

    Returns:
        peaks (np.ndarray): Peak positions
        x (np.ndarray): Local threshold
        threshold_local (np.ndarray): Filtered novelty curve
    """
    offset = x.mean() * offset_rel
    x = gaussian_filter1d(x, sigma=sigma)
    threshold_local = median_filter(x, size=median_len) + offset
    peaks = []
    for i in range(1, x.shape[0] - 1):
        if x[i - 1] < x[i] and x[i] > x[i + 1]:
            if x[i] > threshold_local[i]:
                peaks.append(i)
    peaks = np.array(peaks, dtype=np.int32)
    return peaks


def peak_picking_boeck(activations, threshold=0.5, fps=100, include_scores=False, combine=False,
                       pre_avg=12, post_avg=6, pre_max=6, post_max=6):
    """Detects peaks.

    | Implements the peak-picking method described in:
    | "Evaluating the Online Capabilities of Onset Detection Methods"
    | Sebastian Boeck, Florian Krebs and Markus Schedl
    | Proceedings of the 13th International Society for Music Information Retrieval Conference (ISMIR), 2012

    Modified by Jan Schlueter, 2014-04-24

    Args:
        activations (np.nadarray): Vector of activations to process
        threshold (float): Threshold for peak-picking (Default value = 0.5)
        fps (scalar): Frame rate of onset activation function in Hz (Default value = 100)
        include_scores (bool): Include activation for each returned peak (Default value = False)
        combine (bool): Only report 1 onset for N seconds (Default value = False)
        pre_avg (float): Use N past seconds for moving average (Default value = 12)
        post_avg (float): Use N future seconds for moving average (Default value = 6)
        pre_max (float): Use N past seconds for moving maximum (Default value = 6)
        post_max (float): Use N future seconds for moving maximum (Default value = 6)

    Returns:
        peaks (np.ndarray): Peak positions
    """

    activations = activations.ravel()

    # detections are activations equal to the moving maximum
    max_length = int((pre_max + post_max) * fps) + 1
    if max_length > 1:
        max_origin = int((pre_max - post_max) * fps / 2)
        mov_max = filters.maximum_filter1d(activations, max_length, mode='constant', origin=max_origin)
        detections = activations * (activations == mov_max)
    else:
        detections = activations

    # detections must be greater than or equal to the moving average + threshold
    avg_length = int((pre_avg + post_avg) * fps) + 1
    if avg_length > 1:
        avg_origin = int((pre_avg - post_avg) * fps / 2)
        mov_avg = filters.uniform_filter1d(activations, avg_length, mode='constant', origin=avg_origin)
        detections = detections * (detections >= mov_avg + threshold)
    else:
        # if there is no moving average, treat the threshold as a global one
        detections = detections * (detections >= threshold)

    # convert detected onsets to a list of timestamps
    if combine:
        stamps = []
        last_onset = 0
        for i in np.nonzero(detections)[0]:
            # only report an onset if the last N frames none was reported
            if i > last_onset + combine:
                stamps.append(i)
                # save last reported onset
                last_onset = i
        stamps = np.array(stamps)
    else:
        stamps = np.where(detections)[0]

    return stamps


def shape_list(input_tensor):
    """Return list of dims, statically where possible."""
    tensor = tf.convert_to_tensor(input_tensor)

    # If unknown rank, return dynamic shape
    if tensor.get_shape().dims is None:
        return tf.shape(tensor)

    static = tensor.get_shape().as_list()
    shape = tf.shape(tensor)

    ret = []
    for i, dim in enumerate(static):
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


class Norm(tf.keras.layers.Layer):
    def __init__(self, axes=[1, 2], adaptive=False):
        super().__init__()
        self.axes = axes
        self.rank = None
        self.adpative = adaptive

    def build(self, input_shape):
        d = int(input_shape[-1])
        self.rank = len(input_shape)

        if not self.adpative:
            self.gamma = self.add_weight(name='gamma',
                                         shape=[d],
                                         initializer=tf.keras.initializers.Ones,
                                         trainable=True)

            self.beta = self.add_weight(name='beta',
                                        shape=[d],
                                        initializer=tf.keras.initializers.Zeros,
                                        trainable=True)

    def call(self, inputs, valid_len=None, epsilon=1e-7):
        if 1 in self.axes: # containing temporal dimension
            mask = tf.sequence_mask(valid_len, maxlen=shape_list(inputs)[1], dtype=tf.float32) # [b, n]
            mask = mask[:, :, tf.newaxis] if self.rank == 3 else mask[:, :, tf.newaxis, tf.newaxis]
            mean, variance = tf.nn.weighted_moments(inputs, axes=self.axes, frequency_weights=mask, keepdims=True)
        else:
            mean, variance = tf.nn.moments(inputs, axes=self.axes, keepdims=True)

        normalized = (inputs - mean) * tf.math.rsqrt(variance + epsilon)

        if self.adpative:
            '''Adaptive Normalization. ref: Understanding and Improving Layer Normalization (NIPS 2019)'''
            C = 1
            k = 0.1
            adapter = C * (1 - k * tf.stop_gradient(normalized))
            return adapter * normalized
        else:
            return self.gamma * normalized + self.beta


class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, axes=[0, 1, 2], momentum=0.99, epsilon=0.001):
        super().__init__()
        self.axes = axes
        self.rank = None
        self.epsilon = epsilon
        self.momentum = momentum


    def build(self, input_shape):
        d = int(input_shape[-1])
        self.rank = len(input_shape)

        self.gamma = self.add_weight(name='gamma',
                                     shape=[d],
                                     initializer=tf.keras.initializers.Ones,
                                     trainable=True)

        self.beta = self.add_weight(name='beta',
                                    shape=[d],
                                    initializer=tf.keras.initializers.Zeros,
                                    trainable=True)

        self.moving_mean = self.add_weight(name='moving_mean',
                                           shape=[d],
                                           initializer=tf.keras.initializers.Zeros,
                                           trainable=False)

        self.moving_var = self.add_weight(name='moving_var',
                                           shape=[d],
                                           initializer=tf.keras.initializers.Ones,
                                           trainable=False)

    def call(self, inputs, valid_len, training=False):
        mask = tf.sequence_mask(valid_len, maxlen=shape_list(inputs)[1], dtype=tf.float32) # [b, n]
        mask = mask[:, :, tf.newaxis, tf.newaxis] # [b, n, 1, 1]
        batch_mean, batch_var = tf.nn.weighted_moments(inputs, axes=self.axes, frequency_weights=mask, keepdims=False) # [d]

        # Update moving momentum
        update_mean = tf.cond(
            training,
            lambda: self.moving_mean * self.momentum + batch_mean * (1 - self.momentum),
            lambda: self.moving_mean
        )
        update_var = tf.cond(
            training,
            lambda: self.moving_var * self.momentum + batch_var * (1 - self.momentum),
            lambda: self.moving_var
        )
        self.moving_mean.assign(update_mean)
        self.moving_var.assign(update_var)

        # Normalization
        mean, var = tf.cond(
            training,
            lambda: (batch_mean, batch_var),
            lambda: (self.moving_mean, self.moving_var)
        )
        normalized = (inputs - mean) * tf.math.rsqrt(var + self.epsilon)
        return self.gamma * normalized + self.beta


class DuoConv2D(tf.keras.layers.Layer):
    def __init__(self, n_units, kernel_size, padding='same'):
        super().__init__()
        self.n_units = n_units
        self.kernel_size = kernel_size
        self.padding = padding
        self.out_dense = tf.keras.layers.Dense(n_units)

    def build(self, input_shape):
        depth_multiplier = int(self.n_units // input_shape[-1])
        self.conv0 = tf.keras.layers.SeparableConv2D(
            self.n_units,
            kernel_size=(1, 1),
            dilation_rate=(1, 1),
            depth_multiplier=depth_multiplier,
            padding=self.padding,
        )
        self.conv1 = tf.keras.layers.SeparableConv2D(
            self.n_units,
            kernel_size=self.kernel_size,
            dilation_rate=(1, 1),
            depth_multiplier=depth_multiplier,
            padding=self.padding,
        )
        self.conv2 = tf.keras.layers.SeparableConv2D(
            self.n_units,
            kernel_size=self.kernel_size,
            dilation_rate=(2, 1),
            depth_multiplier=depth_multiplier,
            padding=self.padding,
        )

    def call(self, x):
        '''x = [b, n, f, c]'''
        if self.padding == 'same':
            enc0 = self.conv0(x)
            enc1 = self.conv1(x)
            enc2 = self.conv2(x)
        elif self.padding == 'valid':
            n_pad = self.kernel_size[0] // 2
            enc1 = self.conv1(tf.pad(x, [(0,0), (n_pad,n_pad), (0,0), (0,0)]))
            enc2 = self.conv2(tf.pad(x, [(0,0), (2*n_pad,2*n_pad), (0,0), (0,0)]))
        return self.out_dense(enc0 + enc1 + enc2)


class SE_block(tf.keras.layers.Layer):
    '''Squeeze and excitation block'''
    def __init__(self, alpha=0.5, activation_func='relu', axis=[1]):
        super().__init__()

        self.alpha = alpha
        self.activation_func = activation_func
        self.axis = axis
        self.inner = None
        self.outer = None

    def build(self, input_shape):
        self.inner = tf.keras.layers.Dense(int(input_shape[-1] * self.alpha), activation=self.activation_func)
        self.outer = tf.keras.layers.Dense(int(input_shape[-1]), activation=tf.sigmoid)
        self.shape = input_shape

    def call(self, input, valid_len):
        seq_mask = tf.sequence_mask(valid_len, maxlen=shape_list(input)[1], dtype=tf.float32) # [b, n]
        seq_mask = seq_mask[:, :, tf.newaxis] if int(tf.rank(input)) == 3 else seq_mask[:, :, tf.newaxis, tf.newaxis]
        valid_len = valid_len[:,  tf.newaxis, tf.newaxis] if int(tf.rank(input)) == 3 else valid_len[:, tf.newaxis, tf.newaxis, tf.newaxis]

        if self.axis == [1, 2]:
            gap = tf.reduce_sum(input*seq_mask, axis=self.axis, keepdims=True) / (tf.cast(valid_len, tf.float32) * self.shape[2]) # [b, c]
        elif self.axis == [1]:
            gap = tf.reduce_sum(input*seq_mask, axis=self.axis, keepdims=True) / tf.cast(valid_len, tf.float32) # [b, (f), c]
        elif self.axis == [2]:
            gap = tf.reduce_mean(input, axis=self.axis, keepdims=True) # [b, n, c]
        else:
            print('invalid axes.')
            exit(1)
        scale = self.outer(self.inner(gap))
        return scale * input


class SpecCNN(tf.keras.layers.Layer):
    def __init__(
        self,
        n_units=128,
        dropout_rate=0,
        activation_func='relu',
        kernel_size=(5, 5),
        is_ssm=False,
        freq_collapse=True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_units = n_units
        self.dropout_rate = dropout_rate
        self.activation = tf.keras.layers.Activation(activation_func)
        self.kernel_size = kernel_size
        self.is_ssm = is_ssm
        self.freq_collapse = freq_collapse

        alpha = 0.5
        self.conv1 = DuoConv2D(n_units//4, kernel_size=kernel_size)
        self.conv2 = DuoConv2D(n_units//2, kernel_size=kernel_size)
        self.conv3 = DuoConv2D(n_units, kernel_size=kernel_size)
        self.se1 = SE_block(alpha=alpha, axis=[2])
        self.se2 = SE_block(alpha=alpha, axis=[2])
        self.se3 = SE_block(alpha=alpha, axis=[2])

        self.norm1 = Norm(axes=[1], adaptive=False)
        self.norm2 = Norm(axes=[1], adaptive=False)
        self.norm3 = Norm(axes=[1], adaptive=False)

        self.reduce_dense = tf.keras.layers.Dense(1, name='reduce_dense')
        self.out_dense = tf.keras.layers.Dense(n_units, name='out_dense')
        self.out_norm_t = Norm(axes=[1], adaptive=False)
        self.out_norm_c = Norm(axes=[-1], adaptive=False)
        self.out_se = SE_block(alpha=alpha, axis=[1])

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def shuffle_channel(self, x, groups):
        b, n, d = shape_list(x)
        output = tf.reshape(x, [b, n, groups, d // groups])
        output = tf.transpose(output, [0, 1, 3, 2])
        return tf.reshape(output, [b, n, d])

    def call(self, input, valid_len):
        # input = [b, n, f, c]

        enc1 = self.conv1(input) # [b, n, f, d]
        enc1 = self.norm1(enc1, valid_len)
        enc1 = self.activation(enc1)
        enc1 = self.se1(enc1, valid_len)

        enc2 = self.conv2(enc1) # [b, n, f, d]
        enc2 = self.norm2(enc2, valid_len)
        enc2 = self.activation(enc2)
        enc2 = self.se2(enc2, valid_len)
        enc2 = self.dropout(enc2)

        enc3 = self.conv3(enc2) # [b, n, f, d]
        enc3 = self.norm3(enc3, valid_len)
        enc3 = self.activation(enc3)
        enc3 = self.se3(enc3, valid_len)

        if not self.freq_collapse:
            return enc3

        if not self.is_ssm:
            # Summarize the frequency dimension
            enc_max = tf.reduce_max(enc3, axis=2) # [b, n, d]
            enc_dense = tf.squeeze(self.reduce_dense(tf.transpose(enc3, [0, 1, 3, 2])), axis=-1) # [b, n, d]
            output = enc_max + enc_dense # [b, n, d]
            output = self.out_dense(output) # [b, n, d]
            output = self.out_norm_t(output, valid_len)
            b, n, d = shape_list(output)
            output = tf.reshape(output, [b, n, 20, d//20])
            output = self.out_norm_c(output, valid_len)
            output = tf.reshape(output, [b, n, d])
            output = self.activation(output)
            output = self.out_se(output, valid_len)
            output = self.dropout(output)
        else:
            output = tf.reduce_max(enc3, axis=2) # [b, n, d]
        return output


class SelfCNN(tf.keras.layers.Layer):
    """Multi-head attention keras layer wrapper"""
    def __init__(
        self,
        n_units=32,
        dropout_rate=0,
        activation_func='relu',
        kernel_size=(7, 7),
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_units = n_units
        self.dropout_rate = dropout_rate
        self.activation_func = activation_func
        self.activation = tf.keras.activations.deserialize(self.activation_func)

        self.conv1 = tf.keras.layers.Conv2D(2*n_units, kernel_size=kernel_size, activation=None, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(n_units, kernel_size=kernel_size, activation=None, padding='same')

        self.norm1 = Norm(axes=[1,2], adaptive=False)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, input, valid_len):
        '''input has shape = [b, n, n, d]'''
        output = self.conv1(input) # [b, n, n, d]
        output = self.norm1(output, valid_len)
        output = self.activation(output)
        output = self.conv2(output)
        return output


# === NEW: Positional Encoding ===
class SinusoidalPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=935, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        
        # Create positional encoding
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term[:d_model//2])
        
        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
        
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]


# === NEW: Drum Encoder ===
class DrumEncoder(tf.keras.layers.Layer):
    def __init__(self, mel_dim=80, chroma_dim=12, d_model=256, num_heads=8, 
                 num_layers=2, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.mel_dim = mel_dim
        self.chroma_dim = chroma_dim
        self.d_model = d_model
        
        # Separate projections for mel and chroma
        self.mel_projection = tf.keras.layers.Dense(d_model // 2, name='drum_mel_proj')
        self.chroma_projection = tf.keras.layers.Dense(d_model // 2, name='drum_chroma_proj')
        
        # Combined input normalization
        self.input_norm = tf.keras.layers.LayerNormalization(name='drum_input_norm')
        
        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, name='drum_pos_enc')
        self.dropout = tf.keras.layers.Dropout(dropout, name='drum_dropout')
        
        # Transformer encoder layers
        self.encoder_layers = []
        for i in range(num_layers):
            self.encoder_layers.append(
                tf.keras.layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=d_model // num_heads,
                    dropout=dropout,
                    name=f'drum_mha_{i}'
                )
            )
            self.encoder_layers.append(
                tf.keras.layers.LayerNormalization(name=f'drum_norm1_{i}')
            )
            self.encoder_layers.append(
                tf.keras.Sequential([
                    tf.keras.layers.Dense(d_model * 4, activation='gelu'),
                    tf.keras.layers.Dropout(dropout),
                    tf.keras.layers.Dense(d_model)
                ], name=f'drum_ffn_{i}')
            )
            self.encoder_layers.append(
                tf.keras.layers.LayerNormalization(name=f'drum_norm2_{i}')
            )
        
    def call(self, drum_mel, drum_chroma, mask=None, training=None):
        # ADD THIS DEBUG LINE FIRST:
        # tf.print("🔍 DrumEncoder Debug:")
        # tf.print("  drum_mel shape:", tf.shape(drum_mel))
        # tf.print("  drum_chroma shape:", tf.shape(drum_chroma))
        # tf.print("  expected d_model:", self.d_model)
        # Project mel and chroma features
        mel_proj = self.mel_projection(drum_mel)      # (B, T, d_model//2)
        chroma_proj = self.chroma_projection(drum_chroma)  # (B, T, d_model//2)

        # ADD THESE DEBUG LINES:
        # tf.print("  mel_proj shape:", tf.shape(mel_proj))
        # tf.print("  chroma_proj shape:", tf.shape(chroma_proj))
        
        # Concatenate to form full feature vector
        x = tf.concat([mel_proj, chroma_proj], axis=-1)  # (B, T, d_model)

        # ADD THIS DEBUG LINE:
        # tf.print("  concatenated x shape:", tf.shape(x))
        # tf.print("  positional encoding shape:", tf.shape(self.pos_encoding.pe))
            
        # Add positional encoding
        x = self.input_norm(x)
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        
        # Apply transformer encoder layers
        for i in range(0, len(self.encoder_layers), 4):
            mha = self.encoder_layers[i]
            norm1 = self.encoder_layers[i + 1]
            ffn = self.encoder_layers[i + 2]
            norm2 = self.encoder_layers[i + 3]
            
            # Multi-head attention
            attn_output = mha(x, x, attention_mask=mask, training=training)
            x = norm1(x + attn_output)
            
            # Feed forward
            ffn_output = ffn(x, training=training)
            x = norm2(x + ffn_output)
        
        return x


# === NEW: Cross-Attention Block ===
class CrossAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.cross_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
            name='cross_attn'
        )
        
        self.norm1 = tf.keras.layers.LayerNormalization(name='cross_norm1')
        self.norm2 = tf.keras.layers.LayerNormalization(name='cross_norm2')
        
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model * 4, activation='gelu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(d_model)
        ], name='cross_ffn')
        
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    def call(self, tgt, memory, tgt_mask=None, memory_mask=None, training=None):
        # Cross-attention with drums
        tgt_norm = self.norm1(tgt)
        tgt2, attn_weights = self.cross_attn(
            tgt_norm, memory, attention_mask=memory_mask, 
            return_attention_scores=True, training=training
        )
        tgt = tgt + self.dropout(tgt2, training=training)
        
        # FFN
        tgt_norm = self.norm2(tgt)
        tgt2 = self.ffn(tgt_norm, training=training)
        tgt = tgt + self.dropout(tgt2, training=training)
        
        return tgt, attn_weights


class CAMHSA(tf.keras.layers.Layer):
    """Convolution-Augmented Multi-Head Self-Attention"""
    def __init__(
        self,
        n_units=32,
        n_heads=8,
        max_len=540,
        attn_dropout_rate=0,
        cnn_dropout_rate=0,
        activation_func=None,
        self_mask=False,
        shared_pos=False,
        return_maps=False,
        attn_mask=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_units = n_units
        self.n_heads = n_heads
        self.max_len = max_len
        self.attn_dropout_rate = attn_dropout_rate
        self.cnn_dropout_rate = cnn_dropout_rate
        self.activation_func = activation_func
        self.self_mask = self_mask
        self.shared_pos = shared_pos
        self.return_maps = return_maps
        self.attn_mask = attn_mask
        self.pos_clip = 60

        self.proj_q = tf.keras.layers.Dense(n_units, activation=activation_func, use_bias=True, name='proj_q')
        self.proj_k = tf.keras.layers.Dense(n_units, activation=activation_func, use_bias=True, name='proj_k')
        self.proj_v = tf.keras.layers.Dense(n_units, activation=activation_func, use_bias=True, name='proj_v')
        self.proj_h = tf.keras.layers.Dense(n_units, activation=activation_func, use_bias=True, name='proj_h')
        self.attn_dropout = tf.keras.layers.Dropout(attn_dropout_rate)
        self.cnn_dropout = tf.keras.layers.Dropout(cnn_dropout_rate)
        self.layer_norm = Norm(axes=[-1], adaptive=False)

        self.selfcnn = SelfCNN(
            n_units=n_heads,
            dropout_rate=cnn_dropout_rate,
            activation_func='relu',
            kernel_size=(5, 5),
        )

    def build(self, input_shape):
        if not self.shared_pos:
            # pos embedding
            self.pos_k = self.add_weight(name='pos_k',
                                         shape=[2 * self.pos_clip + 1, self.n_units // self.n_heads],
                                         initializer=tf.random_uniform_initializer,
                                         trainable=True)

    def call(self, query, valid_len, attn_mask=None):
        # Query has shape [b, n, d]
        b, n, d_in = shape_list(query)
        seq_mask = tf.sequence_mask(valid_len, maxlen=n, dtype=tf.bool) # [b, n]

        # Relative position encodings
        max_len = shape_list(query)[1]
        '''Self-Attention with Relative Position Representations (NAACL-HLT 2018)'''
        rel_pos_idx = tf.range(max_len)[tf.newaxis, :] - tf.range(max_len)[:, tf.newaxis] # [n, n]
        rel_pos_idx = tf.clip_by_value(rel_pos_idx, -self.pos_clip, self.pos_clip)
        rel_pos_idx += self.pos_clip
        pos_enc_k = tf.nn.embedding_lookup(self.pos_k, rel_pos_idx) # [n, n, d/h]

        # Projection
        q_emb = self.proj_q(query) # [b, n, d]
        k_emb = self.proj_k(query) # [b, n, d]
        v_emb = self.proj_v(query) # [b, n, d]

        # Head splitting
        q_emb = tf.concat(tf.split(q_emb, self.n_heads, axis=-1), 0) # [hb, n, d/h]
        k_emb = tf.concat(tf.split(k_emb, self.n_heads, axis=-1), 0) # [hb, n, d/h]
        v_emb = tf.concat(tf.split(v_emb, self.n_heads, axis=-1), 0) # [hb, n, d/h]

        # Attention computation
        QK = tf.matmul(q_emb, k_emb, transpose_b=True) # [hb, n, n]

        QR_K = tf.matmul(tf.transpose(q_emb, [1,0,2]), pos_enc_k, transpose_b=True) # [n, hb, n]
        QR_K = tf.transpose(QR_K, [1,0,2]) # [hb, n, n]
        attn_map = QK + QR_K # [hb, n, n]
        attn_map = attn_map / (shape_list(k_emb)[-1]**0.5) # [hb, n, n]

        # Convolution on attention maps
        attn_map = tf.stack(tf.split(attn_map, self.n_heads, 0), -1) # [b, n, n, h]
        attn_map = self.selfcnn(attn_map, valid_len) # [b, n, n, h]
        attn_map = tf.concat(tf.split(attn_map, self.n_heads, -1), 0) # [hb, n, n, 1]
        attn_map = tf.squeeze(attn_map, -1) # [hb, n, n]

        # Attention masking
        valid_mask = tf.tile(seq_mask[:, tf.newaxis, :], [self.n_heads, n, 1]) # [hb, n, n]
        if attn_mask is not None:
            valid_mask = tf.logical_and(valid_mask, attn_mask)

        attn_map = tf.where(valid_mask, attn_map, -1e12) # [hb, n, n]

        # Activation
        attn_map = tf.nn.softmax(attn_map) # [hb, n, n]

        # Combinatorial representation
        output = tf.matmul(attn_map, v_emb) # [hb, n, d/h]
        output = tf.concat(tf.split(output, self.n_heads, 0), -1) # [b, n, d]
        output = self.proj_h(output) # [b, n, d]
        output = self.attn_dropout(output)
        output += query # residual connection

        if self.return_maps:
            return self.layer_norm(output), attn_map
        else:
            return self.layer_norm(output)


class FeedForward(tf.keras.layers.Layer):
    """Feedfoward layer of the transformer model.
    Paramters
    ---------
    n_units: list[int, int]
        A two-element integer list. The first integer represents the output embedding size
        of the first convolution layer, and the second integer represents the embedding size
        of the second convolution layer.
    activation_func: str
        Activation function of the first covolution layer. Available options can be found
        from the tensorflow.keras official site.
    dropout_rate: float
        Dropout rate of all dropout layers.
    """
    def __init__(
            self,
            n_units=[1024, 256],
            activation_func='relu',
            dropout_rate=0,
            output_norm=True,
            residual=True,
            kernel_size=None,
    ):
        super().__init__()

        self.n_units = n_units
        self.activation_func = activation_func
        self.dropout_rate = dropout_rate
        self.residual = residual
        self.output_norm = output_norm
        self.kernel_size = kernel_size

        self.inner = tf.keras.layers.Dense(n_units[0], activation=activation_func)
        self.outer = tf.keras.layers.Dense(n_units[1], activation=None)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm = Norm(axes=[-1], adaptive=False)

    def call(self, input):
        output = self.inner(input)
        output = self.outer(output)
        output = self.dropout(output)
        if self.residual:
            output += input # residual connection
        return self.layer_norm(output) if self.output_norm else output


class Attention(tf.keras.layers.Layer):
    def __init__(
        self,
        n_units=128,
        n_heads=4,
        max_len=540,
        activation_func=None,
        dropout_rate=0,
        return_maps=False,
        with_pos_enc=True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_units = n_units
        self.n_heads = n_heads
        self.max_len = max_len
        self.activation_func = activation_func
        self.dropout_rate = dropout_rate
        self.pos_clip = max_len - 1
        self.return_maps = return_maps
        self.with_pos_enc = with_pos_enc

        self.proj_q = tf.keras.layers.Dense(n_units, activation=activation_func, name='proj_q')
        self.proj_k = tf.keras.layers.Dense(n_units, activation=activation_func, name='proj_k')
        self.proj_v = tf.keras.layers.Dense(n_units, activation=activation_func, name='proj_v')
        self.proj_h = tf.keras.layers.Dense(n_units, activation=activation_func, name='proj_h')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm = Norm(axes=[-1], adaptive=False)

    def build(self, input_shape):
        if self.with_pos_enc:
            # pos embedding
            self.pos_k = self.add_weight(name='pos_k',
                                         shape=[2 * self.pos_clip + 1, self.n_units // self.n_heads],
                                         initializer=tf.random_uniform_initializer,
                                         trainable=True)

    def call(self, query, key=None, value=None, valid_len=None, attn_mask=None):
        # query = [b, n, d_in]

        if key is None:
            key = query
        if value is None:
            value = key

        b, n, d_in = shape_list(query)

        # Projection
        q_emb = self.proj_q(query) # [b, n, d]
        k_emb = self.proj_k(key) # [b, n, d]
        v_emb = self.proj_v(value) # [b, n, d]

        # Head splitting
        q_emb = tf.concat(tf.split(q_emb, self.n_heads, axis=-1), 0) # [hb, n, d/h]
        k_emb = tf.concat(tf.split(k_emb, self.n_heads, axis=-1), 0) # [hb, n, d/h]
        v_emb = tf.concat(tf.split(v_emb, self.n_heads, axis=-1), 0) # [hb, n, d/h]

        # Attention computation
        QK = tf.matmul(q_emb, k_emb, transpose_b=True) # [hb, n, n]
        if self.with_pos_enc:
            # Relative position encodings
            max_len = shape_list(query)[1]
            '''Self-Attention with Relative Position Representations (NAACL-HLT 2018)'''
            rel_pos_idx = tf.range(max_len)[tf.newaxis, :] - tf.range(max_len)[:, tf.newaxis] # [n, n]
            rel_pos_idx = tf.clip_by_value(rel_pos_idx, -self.pos_clip, self.pos_clip)
            rel_pos_idx += self.pos_clip
            pos_enc_k = tf.nn.embedding_lookup(self.pos_k, rel_pos_idx) # [n, n, d/h]

            QR_K = tf.matmul(tf.transpose(q_emb, [1, 0, 2]), pos_enc_k, transpose_b=True)  # [n, hb, n]
            QR_K = tf.transpose(QR_K, [1, 0, 2])  # [hb, n, n]

            attn_map = QK + QR_K # [hb, n, n]
        else:
            attn_map = QK

        attn_map = attn_map / (shape_list(k_emb)[-1]**0.5) # [hb, n, n]

        if valid_len is not None:
            # Attention masking
            seq_mask = tf.sequence_mask(valid_len, maxlen=n, dtype=tf.bool) # [b, n]
            valid_mask = tf.tile(seq_mask[:, tf.newaxis, :], [self.n_heads, n, 1]) # [hb, n, n]
            attn_map = tf.where(valid_mask, attn_map, -1e12) # [hb, n, n]

        if attn_mask is not None:
            attn_map = tf.where(attn_mask, attn_map, -1e12) # [hb, n, n]

        # Activation
        attn_map = tf.nn.softmax(attn_map) # [hb, n, n]

        # Combinatorial representation
        output = tf.matmul(attn_map, v_emb) # [hb, n, d/h]
        output = tf.concat(tf.split(output, self.n_heads, 0), -1) # [b, n, d]
        output = self.proj_h(output) # [b, n, d]
        output = self.dropout(output)
        output += query # residual connection

        if self.return_maps:
            return self.layer_norm(output), attn_map
        else:
            return self.layer_norm(output)


class CNNBase2D(tf.keras.layers.Layer):
    def __init__(
            self,
            n_units=[40, 80],
            activation_func='relu',
            kernel_size=(3, 3),
            dropout_rate=0,
            padding='same',
    ):
        super().__init__()

        self.norm1 = BatchNorm()
        self.norm2 = BatchNorm()

        self.conv1 = tf.keras.layers.Conv2D(n_units[0], kernel_size=kernel_size, padding=padding)
        self.conv2 = tf.keras.layers.Conv2D(n_units[1], kernel_size=kernel_size, padding=padding)

        self.activation = tf.keras.activations.deserialize(activation_func)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.half_t = kernel_size[0] // 2
        self.padding = padding

    def call(self, x, valid_len, training=False):
        '''x = [b, n, f, d]'''
        if self.padding == 'valid':
            x = tf.pad(x, [(0,0), (self.half_t, self.half_t), (0,0), (0,0)], constant_values=0)
        enc = self.conv1(x) # [b, n, f, d1]
        enc = self.norm1(enc, valid_len, training=training) # [b, n, f, d1]
        enc = self.activation(enc) # [b, n, f, d1]
        enc = self.dropout(enc) # [b, n, f, d1]

        if self.padding == 'valid':
            enc = tf.pad(enc, [(0,0), (self.half_t, self.half_t), (0,0), (0,0)], constant_values=0)
        enc = self.conv2(enc) # [b, n, f, d2]
        enc = self.norm2(enc, valid_len, training=training) # [b, n, f, d2]
        enc = self.activation(enc) # [b, n, f, d2]
        enc = self.dropout(enc) # [b, n, f, d2]
        return enc


class ResBlock2D(tf.keras.layers.Layer):
    def __init__(
            self,
            n_units,
            activation_func='relu',
            kernel_size=(3, 3),
            dropout_rate=0,
    ):
        super().__init__()

        self.norm1 = BatchNorm()
        self.norm2 = BatchNorm()

        self.conv1 = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding='same', depth_multiplier=1)
        self.conv2 = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding='same', depth_multiplier=1)

        self.activation = tf.keras.activations.deserialize(activation_func)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, valid_len, training=False):
        '''x = [b, n, f, d]'''
        enc = self.conv1(x) # [b, n, f, d1]
        enc = self.norm1(enc, valid_len, training=training) # [b, n, f, d1]
        enc = self.activation(enc) # [b, n, f, d1]
        enc = self.dropout(enc) # [b, n, f, d1]

        enc = self.conv2(enc) # [b, n, f, d2]
        enc = self.norm2(enc, valid_len, training=training) # [b, n, f, d2]
        enc = self.activation(enc) # [b, n, f, d2]
        enc = self.dropout(enc) # [b, n, f, d2]
        return x + enc # [b, n, f, d2]


class ChromaCNNBase2D(tf.keras.layers.Layer):
    def __init__(
            self,
            n_units=[40, 80],
            activation_func='relu',
            kernel_size=(5, 12),
            dropout_rate=0,
            padding='valid',
    ):
        super().__init__()

        self.norm1 = BatchNorm()
        self.norm2 = BatchNorm()

        self.conv1 = tf.keras.layers.Conv2D(n_units[0], kernel_size=kernel_size, padding=padding)
        self.conv2 = tf.keras.layers.Conv2D(n_units[1], kernel_size=kernel_size, padding=padding)

        self.activation = tf.keras.activations.deserialize(activation_func)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.half_w = kernel_size[0] // 2
        self.padding = padding

    def call(self, x, valid_len, training=False):
        '''x = [b, n, 12, d]'''
        # Chroma expansion
        x_extend = tf.concat([x, x[:, :, :11, :]], axis=2) # [b, n, 23, d]
        x_extend = tf.pad(x_extend, [(0,0), (self.half_w, self.half_w), (0,0), (0,0)]) # [b, n+w, 23, d]

        # 1st conv
        enc = self.conv1(x_extend) # [b, n, 12, d1]
        enc = self.norm1(enc, valid_len, training=training) # [b, n, 12, d1]
        enc = self.activation(enc) # [b, n, 12, d1]
        enc = self.dropout(enc) # [b, n, 12, d1]

        # Chroma expansion
        enc = tf.concat([enc, enc[:, :, :11, :]], axis=2) # [b, n, 23, d]
        enc = tf.pad(enc, [(0,0), (self.half_w, self.half_w), (0,0), (0,0)]) # [b, n+w, 23, d]

        # 2nd conv
        enc = self.conv2(enc) # [b, n, 12, d2]
        enc = self.norm2(enc, valid_len, training=training) # [b, n, 12, d2]
        enc = self.activation(enc) # [b, n, 12, d2]
        enc = self.dropout(enc) # [b, n, 12, d2]
        return enc


class ChromaCNN2D(tf.keras.layers.Layer):
    def __init__(
            self,
            activation_func='relu',
            dropout_rate=0,
            kernel_size=[5, 12],
            padding='valid',
    ):
        super().__init__()

        self.norm1 = BatchNorm()
        self.norm2 = BatchNorm()

        self.half_w = kernel_size[0] // 2
        self.conv1 = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding=padding, depth_multiplier=1)
        self.conv2 = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding=padding, depth_multiplier=1)

        self.activation = tf.keras.activations.deserialize(activation_func)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, valid_len, training=False):
        '''x = [b, n, 12, d]'''

        # Chroma expansion
        x_extend = tf.concat([x, x[:, :, :11, :]], axis=2) # [b, n, 23, d]
        x_extend = tf.pad(x_extend, [(0,0), (self.half_w, self.half_w), (0,0), (0,0)]) # [b, n+w, 23, d]

        enc = self.conv1(x_extend) # [b, n, 12, d]
        enc = self.norm1(enc, valid_len, training=training) # [b, n, 12, d]
        enc = self.activation(enc) # [b, n, 12, d]
        enc = self.dropout(enc) # [b, n, 12, d]

        # Chroma expansion
        enc = tf.concat([enc, enc[:, :, :11, :]], axis=2) # [b, n, 23, d]
        enc = tf.pad(enc, [(0,0), (self.half_w, self.half_w), (0,0), (0,0)]) # [b, n+w, 23, d]

        enc = self.conv2(enc) # [b, n, 12, d]
        enc = self.norm2(enc, valid_len, training=training) # [b, n, 12, d]
        enc = self.activation(enc) # [b, n, 12, d]
        enc = self.dropout(enc) # [b, n, 12, d]
        return x + enc # [b, n, 12, d]


# === MODIFIED: SpecTNT with Cross-Attention ===
class SpecTNT_CAMHSA(tf.keras.layers.Layer):
    def __init__(
        self,
        n_units=80,
        n_heads_f=4,
        n_heads_t=8,
        max_len=540,
        attn_dropout_rate=0,
        cnn_dropout_rate=0,
        activation_func=None,
        shared_pos=False,
        return_maps=False,
        n_fct=1,
        n_mel=80,
        n_chroma=12,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_units = n_units
        self.n_heads_f = n_heads_f
        self.n_heads_t = n_heads_t
        self.max_len = max_len
        self.n_fct = n_fct
        self.n_mel = n_mel
        self.n_chroma = n_chroma
        self.attn_dropout_rate = attn_dropout_rate
        self.cnn_dropout_rate = cnn_dropout_rate
        self.activation_func = activation_func
        self.shared_pos = shared_pos
        self.return_maps = return_maps
        self.pos_clip = max_len - 1

        # Spectral
        self.attn_f = Attention(
            n_units=n_units//2,
            n_heads=n_heads_f,
            max_len=max_len,
            dropout_rate=attn_dropout_rate,
            return_maps=return_maps,
            with_pos_enc=False,
        )
        self.ffn_f = FeedForward(
            n_units=[4 * (n_units//2), n_units//2],
            dropout_rate=attn_dropout_rate,
        )
        self.f2t = tf.keras.layers.Dense(n_units)

        # === NEW: Replace temporal self-attention with cross-attention ===
        self.cross_attn_t = CrossAttentionBlock(
            d_model=n_units,
            num_heads=n_heads_t,
            dropout=attn_dropout_rate,
        )
        self.ffn_t = FeedForward(
            n_units=[4 * n_units, n_units],
            dropout_rate=attn_dropout_rate,
        )
        self.t2f = tf.keras.layers.Dense(n_units//2)

    def call(self, S, FCT, drum_features, valid_len):
        # S = [b, n, f, d/2], FCT = [b, n, 1, d], drum_features = [b, n, d_drum]
        b, n, f, half_d = shape_list(S)

        attn_mask_f = get_spectral_mask(
            n_batch=b, seq_len=n, n_head=self.n_heads_f,n_fct=self.n_fct, n_mel=self.n_mel, n_chroma=self.n_chroma
        ) # [hbn, 93, 93]

        # Concat and reshape
        enc_FCT = self.t2f(FCT) # FCT = [b, n, 1, d/2]
        enc = tf.concat([enc_FCT, S], axis=2) # [b, n, 1+f, d/2]
        enc = tf.reshape(enc, [b*n, 1+f, half_d]) # [b*n, 1+f, d/2]

        # Spectral Attention
        if self.return_maps:
            enc, map_S = self.attn_f(enc, valid_len=None, attn_mask=attn_mask_f) # [b*n, 1+f, d/2], [h_f*b*n, 1+f, 1+f]
        else:
            enc = self.attn_f(enc, valid_len=None, attn_mask=attn_mask_f) # [b*n, 1+f, d/2]
        enc = self.ffn_f(enc) # [b*n, 1+f, d/2]
        enc = tf.reshape(enc, [b, n, 1+f, half_d]) # [b, n, 1+f, d/2]

        # Split
        enc_FCT = enc[:, :, 0, :] # [b, n, d/2]
        enc_S = enc[:, :, 1:, :] # [b, n, f, d/2]

        # === MODIFIED: Cross-attention with drums instead of self-attention ===
        enc_FCT = self.f2t(enc_FCT) # [b, n, d]
        
        # Cross-attention: query=full_song, key/value=drums
        if self.return_maps:
            enc_FCT, map_T = self.cross_attn_t(enc_FCT, drum_features, training=True) # [b, n, d], attention_weights
        else:
            enc_FCT, _ = self.cross_attn_t(enc_FCT, drum_features, training=True) # [b, n, d]
            
        enc_FCT = self.ffn_t(enc_FCT) # [b, n, d]
        enc_FCT = tf.expand_dims(enc_FCT, axis=2) # [b, n, 1, d]

        if self.return_maps:
            return enc_S, enc_FCT, map_S, map_T # [b, n, f, d/2], [b, n, 1, d], [h_f*b*n, f+1, f+1], attention_weights
        else:
            return enc_S, enc_FCT # [b, n, f, d/2], [b, n, 1, d]


class FunctionalSegmentModel(tf.keras.Model):
    """Modified model with drum encoder and cross-attention"""
    def __init__(
        self,
        max_len=935,
        n_units=80,
        n_heads=8,
        cnn_dropout_rate=0,
        attn_dropout_rate=0,
        n_layers=2,
        steps_per_epoch=None,
        n_classes=7,
        return_maps=True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.max_len = max_len
        self.n_units = n_units
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.cnn_dropout_rate = cnn_dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.steps_per_epoch = steps_per_epoch
        self.n_classes = n_classes
        self.return_maps = return_maps
        self.flag = True
        self.confusion_matrix_train_max = tf.zeros([n_classes, n_classes], tf.int32)
        self.confusion_matrix_test_max = tf.zeros([n_classes, n_classes], tf.int32)
        self.confusion_matrix_train_boun = tf.zeros([n_classes, n_classes], tf.int32)
        self.confusion_matrix_test_boun = tf.zeros([n_classes, n_classes], tf.int32)

        self.w_b = 18
        self.w_f = 2

        # === NEW: Drum Encoder ===
        self.drum_encoder = DrumEncoder(
            mel_dim=80,
            chroma_dim=12,
            d_model=n_units,
            num_heads=n_heads,
            num_layers=2,
            dropout=attn_dropout_rate,
        )

        self.spec_prenorm = BatchNorm()

        # CNN feature extraction (unchanged)
        self.specCNNBase = CNNBase2D(
            n_units=[n_units//4, n_units//2],
            activation_func='relu',
            kernel_size=(7, 5),
            dropout_rate=cnn_dropout_rate,
        )
        self.specCNN = ResBlock2D(
            n_units=n_units//2,
            activation_func='relu',
            kernel_size=(7, 5),
            dropout_rate=cnn_dropout_rate,
        )
        self.chromaCNNBase = ChromaCNNBase2D(
            n_units=[n_units//4, n_units//2],
            activation_func='relu',
            kernel_size=(7, 12),
            dropout_rate=cnn_dropout_rate,
        )
        self.chromaCNN = ChromaCNN2D(
            activation_func='relu',
            dropout_rate=cnn_dropout_rate,
            kernel_size=(7, 12),
        )

        self.sepc_res_conv = tf.keras.layers.Conv2D(1, kernel_size=[5, 1], padding='same')
        self.chroma_res_conv = tf.keras.layers.Conv2D(1, kernel_size=[5, 1], padding='same')

        # CNN transition (unchanged)
        self.spec_transition = tf.keras.layers.Dense(n_units//2, name='spec_transition')
        self.spec_transition_norm = Norm(axes=[-1], adaptive=False)
        self.chroma_transition = tf.keras.layers.Dense(n_units//2, name='chroma_transition')
        self.chroma_transition_norm = Norm(axes=[-1], adaptive=False)

        self.fct_dense = tf.keras.layers.Dense(n_units, name='fct_dense')
        self.fct_dense_norm = Norm(axes=[-1], adaptive=False)

        # Spectro-temporal modeling (modified with cross-attention)
        self.specTNT_layers = [SpecTNT_CAMHSA(
            n_units=n_units,
            max_len=max_len,
            cnn_dropout_rate=cnn_dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            return_maps=return_maps,
        ) for _ in range(n_layers)]

        # Output layers (unchanged)
        self.boun1 = tf.keras.layers.Dense(n_units, name='boun1')
        self.boun2 = tf.keras.layers.Dense(n_units, name='boun2')
        self.boun_out = tf.keras.layers.Conv1D(1, kernel_size=5, padding='same')

        self.func1 = tf.keras.layers.Dense(n_units, name='func1')
        self.func2 = tf.keras.layers.Dense(n_units, name='func2')
        self.func_out = tf.keras.layers.Conv1D(self.n_classes, kernel_size=11, padding='same')

        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.result = {k: [] for k in
            [
                'Acc_max',
                'Acc_smooth',
                'P_seg', 'R_seg', 'F1_seg',
                'P_seg3', 'R_seg3', 'F1_seg3',
                'P_pair', 'R_pair', 'F1_pair',
                'loss',
                'loss_b', 'loss_f',
            ]
        }
        self.temp = {k: [] for k in
            [
                'b_ref', 'b_est', 'matched', 'n_b_ref', 'n_b_est', 'n_matched', 'b_ref_in_second', 'b_est_in_second',
                'f_ref', 'f_est',
            ]
        }

    def build(self, input_shape):
        # frequency positional embedding
        self.fpe_S = self.add_weight(name='fpe_S',
                                     shape=[1, 1, 92, self.n_units//2],
                                     initializer=tf.random_uniform_initializer,
                                     trainable=True)

        self.fpe_FCT = self.add_weight(name='fpe_FCT',
                                       shape=[1, 1, 1, self.n_units],
                                       initializer=tf.random_uniform_initializer,
                                       trainable=True)

    def clear_result(self):
        self.result = {k: [] for k in self.result.keys()}
        self.temp = {k: [] for k in self.temp.keys()}
        self.confusion_matrix_train_max = tf.zeros([self.n_classes, self.n_classes], tf.int32)
        self.confusion_matrix_test_max = tf.zeros([self.n_classes, self.n_classes], tf.int32)
        self.confusion_matrix_train_boun = tf.zeros([self.n_classes, self.n_classes], tf.int32)
        self.confusion_matrix_test_boun = tf.zeros([self.n_classes, self.n_classes], tf.int32)
        
    def average_result(self):
        result_dict = {}
        for k, v in self.result.items():
            if len(v) > 0:
                try:
                    # First, try to concatenate (works for arrays)
                    concatenated = tf.concat(v, axis=0)
                    result_dict[k] = tf.reduce_mean(concatenated)
                except tf.errors.InvalidArgumentError:
                    # If concat fails, it means we have scalars - use stack instead
                    stacked = tf.stack(v)
                    result_dict[k] = tf.reduce_mean(stacked)
        return result_dict

    def call(self, spec, chromagram, drum_spec, drum_chromagram, valid_len, training=False):
        '''
        === MODIFIED: Added drum inputs ===
        spec = [b, n, 80]
        chromagram = [b, n, 12]
        drum_spec = [b, n, 80]
        drum_chromagram = [b, n, 12]
        valid_len = [b]
        '''

        # Log compression (unchanged)
        spec = tf.math.log(1 + 100 * tf.nn.relu(spec + 80))
        spec = tf.expand_dims(spec, axis=-1) # [b, n, 80, 1]
        chromagram = tf.expand_dims(chromagram, axis=-1) # [b, n, 12, 1]

        # === NEW: Process drum features ===
        drum_spec = tf.math.log(1 + 100 * tf.nn.relu(drum_spec + 80))
        drum_features = self.drum_encoder(drum_spec, drum_chromagram, training=training) # [b, n, d_model]

        # Pre-Norm (unchanged)
        spec = self.spec_prenorm(spec, valid_len, training=training) # [b, n, 80, 1]

        with tf.name_scope("cnn") as scope_cnn:
            enc_spec = self.specCNNBase(spec, valid_len, training=training) # [b, n, 80, d/2]
            enc_spec = self.specCNN(enc_spec, valid_len, training=training) # [b, n, 80, d/2]
            enc_spec_res = self.sepc_res_conv(tf.transpose(enc_spec, [0, 1, 3, 2])) # [b, n, d/2, 1]
            enc_spec_res = tf.reduce_mean(enc_spec_res, axis=[2, 3]) # [b, n]
            enc_spec = self.spec_transition(enc_spec) # [b, n, 80, d/2]
            enc_spec = self.spec_transition_norm(enc_spec) # [b, n, 80, d/2]

            enc_chroma = self.chromaCNNBase(chromagram, valid_len, training=training) # [b, n, 12, d/2]
            enc_chroma = self.chromaCNN(enc_chroma, valid_len, training=training) # [b, n, 12, d/2]
            enc_chroma_res = self.chroma_res_conv(tf.transpose(enc_chroma, [0, 1, 3, 2])) # [b, n, d/2, 1]
            enc_chroma_res = tf.reduce_mean(enc_chroma_res, axis=[2, 3]) # [b, n]
            enc_chroma = self.chroma_transition(enc_chroma) # [b, n, 12, d/2]
            enc_chroma = self.chroma_transition_norm(enc_chroma) # [b, n, 12, d/2]

        with tf.name_scope("attention") as scope_attn:
            b, n, _, _ = shape_list(spec)
            enc_S = tf.concat([enc_spec, enc_chroma], axis=2) # [b, n, 92, d/2]
            enc_FCT = self.fct_dense(tf.reduce_mean(enc_S, axis=2, keepdims=True)) # [b, n, 1, d]
            enc_FCT = self.fct_dense_norm(enc_FCT)

            enc_S += self.fpe_S # [b, n, 92, d/2]
            enc_FCT += self.fpe_FCT # [b, n, 1, d]

            map_S = None
            for l, specTNT in enumerate(self.specTNT_layers):
                if self.return_maps:
                    # === MODIFIED: Pass drum_features to SpecTNT layers ===
                    enc_S, enc_FCT, map_S, _ = specTNT(enc_S, enc_FCT, drum_features, valid_len=valid_len)
                else:
                    enc_S, enc_FCT = specTNT(enc_S, enc_FCT, drum_features, valid_len=valid_len)
            enc = tf.squeeze(enc_FCT, axis=2) # [b, n, d]

        with tf.name_scope('boundary_estimation') as scope_boun:
            logits_boun = self.boun1(enc) # [b, n, d]
            logits_boun = self.boun2(logits_boun) # [b, n, d]
            logits_boun = tf.squeeze(self.boun_out(logits_boun), axis=2) # [b, n]
            logits_boun = logits_boun + enc_spec_res + enc_chroma_res # [b, n]

        with tf.name_scope("function_estimation") as scope_func:
            logits_func = self.func1(enc) # [b, n, d]
            logits_func = self.func2(logits_func) # [b, n, d]
            logits_func = self.func_out(logits_func) # [b, n, k]

        return logits_boun, logits_func, enc_S, map_S

    def train_step(self, data):
        # === MODIFIED: Unpack drum data ===
        spec, chromagram, drum_spec, drum_chromagram, valid_len, boun_ref, func_ref, sec_ref = data

        with tf.GradientTape() as tape:
            # === MODIFIED: Pass drum inputs to model ===
            logits_boun, logits_func, _, _ = self.call(spec, chromagram, drum_spec, drum_chromagram, valid_len, training=True)

            prob_boun = tf.nn.sigmoid(logits_boun) # [b, n]

            # Estimation (unchanged)
            boun_est = self.decode_boundary(prob_boun, valid_len) # [b, n]
            func_est_max = tf.argmax(logits_func, axis=-1, output_type=tf.int32) # [b, n]
            func_est_smooth = self.decode_labeling(boun_est, logits_func, valid_len) # [b, n]

            self.confusion_matrix_train_max += self.compute_confusion_matrix(
                func_ref,
                func_est_max,
            )
            self.confusion_matrix_train_boun += self.compute_confusion_matrix(
                func_ref,
                func_est_smooth,
            )

            # Losses (unchanged)
            ce_b = self.w_b * self.bce_from_logits(boun_ref, logits_boun, valid_len)
            ce_f = self.w_f * self.cce_from_logits(func_ref, logits_func, valid_len)
            
            if self.flag:
                print('ce_b', ce_b.numpy())
                print('ce_f', ce_f.numpy())
                self.flag = False

            loss = ce_b + ce_f

            # Compute gradients
            trainable_vars = self.trainable_variables
            grads = tape.gradient(loss, trainable_vars)

        # Update weights (unchanged)
        learninig_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=self.steps_per_epoch,
            decay_rate=0.96,
            staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learninig_rate,
            clipnorm=0.1,
        )
        optimizer.apply_gradients(zip(grads, trainable_vars))

        # Update metrics (unchanged)
        score_dict = self.compute_classification_score(func_ref, func_est_max, valid_len, key='Acc_max')
        score_dict.update(
            self.compute_classification_score(func_ref, func_est_smooth, valid_len, key='Acc_smooth')
        )
        score_dict.update(
            self.compute_pairwise_score(boun_ref, func_ref, boun_est, func_est_smooth, valid_len, resolution=global_frame_size)
        )
        score_dict.update(
            self.compute_segment_score(boun_ref, boun_est, valid_len, resolution=global_frame_size)
        )

        self.loss_tracker.update_state(loss)

        score_dict.update(
            {
                'loss': loss,
                'loss_b': ce_b,
                'loss_f': ce_f,
            }
        )
        [self.result[k].append(v) for k, v in score_dict.items()]

    def test_step(self, data):
        # === MODIFIED: Unpack drum data ===
        spec, chromagram, drum_spec, drum_chromagram, valid_len, boun_ref, func_ref, sec_ref = data

        # === MODIFIED: Pass drum inputs to model ===
        logits_boun, logits_func, enc_S, map_S = self(spec, chromagram, drum_spec, drum_chromagram, valid_len, training=False)
        prob_boun = tf.nn.sigmoid(logits_boun) # [b, n]

        # Estimation (unchanged)
        boun_est = self.decode_boundary(prob_boun, valid_len) # [b, n]
        func_est_max = tf.argmax(logits_func, axis=-1, output_type=tf.int32) # [b, n]
        func_est_smooth = self.decode_labeling(boun_est, logits_func, valid_len) # [b, n]

        self.confusion_matrix_test_max += self.compute_confusion_matrix(
            func_ref,
            func_est_max,
        )
        self.confusion_matrix_test_boun += self.compute_confusion_matrix(
            func_ref,
            func_est_smooth,
        )

        # Losses (unchanged)
        ce_b = self.w_b * self.bce_from_logits(boun_ref, logits_boun, valid_len)
        ce_f = self.w_f * self.cce_from_logits(func_ref, logits_func, valid_len)

        loss = ce_b + ce_f

        # Update metrics (unchanged)
        score_dict = self.compute_classification_score(func_ref, func_est_max, valid_len, key='Acc_max')
        score_dict.update(
            self.compute_classification_score(func_ref, func_est_smooth, valid_len, key='Acc_smooth')
        )
        score_dict.update(
            self.compute_pairwise_score(boun_ref, func_ref, boun_est, func_est_smooth, valid_len, resolution=global_frame_size)
        )
        score_dict.update(
            self.compute_segment_score(boun_ref, boun_est, valid_len, resolution=global_frame_size)
        )

        self.loss_tracker.update_state(loss)

        score_dict.update(
            {
                'loss': loss,
                'loss_b': ce_b,
                'loss_f': ce_f,
            }
        )
        [self.result[k].append(v) for k, v in score_dict.items()]

        return boun_est, func_est_smooth

    # === ALL REMAINING METHODS UNCHANGED ===
    def bce_from_logits(self, gt, logits, valid_len, pos_weight=0.3):
        '''gt, logits = [b, n], valid_len = [b]'''
        gt_expaned = self.expand_boundary(gt, valid_len, value=0.5) # [b, n]
        seq_mask = tf.sequence_mask(valid_len, maxlen=shape_list(gt)[1], dtype=tf.float32) # [b, n]

        wbce = tf.nn.weighted_cross_entropy_with_logits(gt_expaned, logits, pos_weight=pos_weight) # [b, n]

        # Mean over time
        loss = tf.reduce_sum(wbce * seq_mask, axis=1) / tf.cast(valid_len, tf.float32) # [b]
        return tf.reduce_mean(loss)

    def cce_from_logits(self, gt, logits, valid_len):
        '''gt = [b, n], logits = [b, n, k], valid_len = [b]'''

        weights = tf.constant([
            6.0, # intro
            0.3, # verse
            0.4, # chorus
            1.4, # bridge
            1.2, # instrument
            0.1, # outro
            0.1, # silence
        ], tf.float32) # [k]

        seq_mask = tf.sequence_mask(valid_len, maxlen=tf.shape(gt)[1], dtype=tf.float32) # [b, n]
        gt_onehot = tf.one_hot(gt, depth=self.n_classes) # [b, n, k]

        # Cross entropy
        wbce = tf.nn.weighted_cross_entropy_with_logits(gt_onehot, logits, pos_weight=weights) # [b, n, k]

        # Mean over time
        loss = tf.reduce_sum(wbce * seq_mask[:, :, tf.newaxis], axis=1) / tf.cast(valid_len, tf.float32)[:, tf.newaxis] # [b, k]
        return tf.reduce_mean(loss)

    def decode_boundary(self, prob_boun, valid_len, method='librosa'):
        '''prob_boun = [b, n]'''
        assert method in ['msaf', 'librosa', 'boeck']

        seq_mask = tf.sequence_mask(valid_len, maxlen=shape_list(prob_boun)[1], dtype=tf.float32) # [b, n]
        prob_boun *= seq_mask # [b, n]

        prob_boun_numpy = prob_boun.numpy() # [b, n]
        peaks = np.zeros_like(prob_boun_numpy, dtype=np.int32) # [b, n]

        if method == 'msaf':
            peak_indices = [peak_picking_MSAF(seq, median_len=7, offset_rel=0.05, sigma=4) for seq in prob_boun_numpy]
        elif method == 'librosa':
            peak_indices = [
                librosa.util.peak_pick(seq, pre_max=10, post_max=10, pre_avg=20, post_avg=10, delta=0.03, wait=10) for seq in prob_boun_numpy
            ]
            peak_indices = [ids.astype(int) for ids in peak_indices]
        elif method == 'boeck':
            peak_indices = [
                peak_picking_boeck(seq, threshold=0.01, fps=2, combine=10, pre_max=10, post_max=10, pre_avg=20, post_avg=10) for seq in prob_boun_numpy
            ]

        for i in range(prob_boun_numpy.shape[0]):
            peaks[i, peak_indices[i]] = 1

        # Ensure each sequence begins with 1
        peaks[:, 0] = 1
        assert np.array_equal(prob_boun_numpy.shape, peaks.shape)
        return tf.constant(peaks, tf.int32) * tf.cast(seq_mask, tf.int32)

    def decode_labeling(self, boun_est, logits_func, valid_len):
        # Labeling based on the boundary prediction
        '''boun_est = [b, n], prob_func = [b, n, k]'''

        boun_est = boun_est.numpy()
        prob_func = tf.nn.sigmoid(logits_func).numpy()
        valid_len = valid_len.numpy()
        max_len = valid_len.max()
        func_est = []
        for i in range(valid_len.shape[0]):
            l = valid_len[i]
            b_i = np.where(np.equal(boun_est[i, :l], 1))[0]
            segments = [segment for segment in np.split(prob_func[i, :l], indices_or_sections=b_i) if len(segment)]
            centroids = np.stack([np.sum(segment, axis=0) for segment in segments]) # [n_segments, d]
            clusters = np.argmax(centroids, axis=-1) # [n_segments]
            label_frame = np.array([c for (segment, c) in zip(segments, clusters) for _ in range(len(segment))])

            if l < max_len:
                label_frame = np.pad(label_frame, (0, max_len-l), 'constant', constant_values=self.n_classes-1)
            func_est.append(label_frame)
        return tf.constant(func_est)

    def expand_boundary(self, boundary, valid_len, value=0.5, size=3):
        '''boundary = [b, n], valid_len = [b]'''
        seq_mask = tf.sequence_mask(valid_len, maxlen=shape_list(boundary)[1], dtype=tf.float32)
        boundary = tf.cast(boundary, tf.float32) # [b, n]

        filter = tf.ones([size, 1, 1]) # [size, 1, 1]
        boundary_expanded = tf.nn.conv1d(boundary[:, :, tf.newaxis], filters=filter, stride=1, padding='SAME') # [b, n, 1]
        boundary_expanded = tf.squeeze(boundary_expanded, axis=-1) # [b, n]

        cond = tf.logical_and((boundary_expanded != boundary), tf.logical_not(tf.cast(boundary, tf.bool))) # [b, n]
        boundary_expanded = tf.where(cond, value, boundary) # [b, n]
        return boundary_expanded * seq_mask # [b, n]

    def compute_segment_score(self, boun_ref, boun_est, valid_len, resolution):
        # gt = [b, n], pred = [b, n], valid_len = [b]

        seq_mask = tf.sequence_mask(valid_len, maxlen=shape_list(boun_ref)[1], dtype=tf.int32) # [b, n]

        boun_ref_expanded = tf.cast(self.expand_boundary(boun_ref, valid_len, value=1), tf.int32) # [b, n]
        matched = boun_est * boun_ref_expanded * seq_mask # [b, n]

        n_boun_ref = tf.reduce_sum(boun_ref, axis=1) # [b]
        n_boun_est = tf.reduce_sum(boun_est, axis=1) # [b]
        n_matched = tf.reduce_sum(matched, axis=1) # [b]

        precision, recall, fscore = [], [], []
        precision3, recall3, fscore3 = [], [], []
        for i in range(shape_list(boun_ref)[0]):
            l = valid_len[i].numpy()
            b_ref = boun_ref[i, :l].numpy() # [n]
            b_est = boun_est[i, :l].numpy() # [n]

            b_ref_in_second = np.where(b_ref == 1)[0] * resolution
            b_est_in_second = np.where(b_est == 1)[0] * resolution

            b_ref_in_interval = segmentFrame2interval(b_ref, frame_size=resolution)
            b_est_in_interval = segmentFrame2interval(b_est, frame_size=resolution)

            self.temp['b_ref'].append(b_ref)
            self.temp['b_est'].append(b_est)
            self.temp['matched'].append(matched[i, :l].numpy())
            self.temp['n_b_ref'].append(n_boun_ref[i].numpy())
            self.temp['n_b_est'].append(n_boun_est[i].numpy())
            self.temp['n_matched'].append(n_matched[i].numpy())
            self.temp['b_ref_in_second'].append(b_ref_in_second)
            self.temp['b_est_in_second'].append(b_est_in_second)

            P, R, F1 = mir_eval.segment.detection(b_ref_in_interval, b_est_in_interval, window=0.5, beta=1.0)
            precision.append(P)
            recall.append(R)
            fscore.append(F1)

            P3, R3, F3 = mir_eval.segment.detection(b_ref_in_interval, b_est_in_interval, window=3, beta=1.0)
            precision3.append(P3)
            recall3.append(R3)
            fscore3.append(F3)

        return {
            'P_seg': tf.constant(precision),
            'R_seg': tf.constant(recall),
            'F1_seg': tf.constant(fscore),
            'P_seg3': tf.constant(precision3),
            'R_seg3': tf.constant(recall3),
            'F1_seg3': tf.constant(fscore3)
        }

    def compute_confusion_matrix(self, func_ref, func_est):
        return tf.math.confusion_matrix(
            labels=tf.reshape(func_ref, [-1]),
            predictions=tf.reshape(func_est, [-1]),
            num_classes=self.n_classes,
            weights=None,
            dtype=tf.dtypes.int32,
        ) # [k, k]

    def compute_pairwise_score(self, boun_ref, func_ref, boun_est, func_est, valid_len, resolution):
        # boun_ref, func_ref, boun_est, func_est = [b, n]
        precision_pair, recall_pair, fscore_pair = [], [], []

        for i in range(shape_list(func_ref)[0]):
            # Ground truth
            l = valid_len[i].numpy()
            f_ref = func_ref[i, :l].numpy() # [n]
            b_ref = boun_ref[i, :l].numpy() # [n]
            f_est = func_est[i, :l].numpy() # [n]
            assert f_ref.shape == f_est.shape
            b_est = boun_est[i, :l].numpy() # [n]

            # Convert frame-level into interval-level
            interval_ref, label_ref = frame2interval(b_ref, f_ref, frame_size=resolution)
            interval_est, label_est = frame2interval(b_est, f_est, frame_size=resolution)

            self.temp['f_ref'].append(f_ref)
            self.temp['f_est'].append(f_est)

            # Pairwise agreement
            P_pair, R_pair, F1_pair = mir_eval.segment.pairwise(
                interval_ref, label_ref, interval_est, label_est, frame_size=0.1
            )
            precision_pair.append(P_pair)
            recall_pair.append(R_pair)
            fscore_pair.append(F1_pair)

        return {
            'P_pair': tf.constant(precision_pair), # [b]
            'R_pair': tf.constant(recall_pair), # [b]
            'F1_pair': tf.constant(fscore_pair), # [b]
        }

    def compute_classification_score(self, func_ref, func_est, valid_len, key='Accuracy'):
        # func_ref, func_est = [b, n]
        seq_mask = tf.sequence_mask(valid_len, maxlen=shape_list(func_ref)[1], dtype=tf.float32) # [b, n]
        matched = tf.cast(func_ref == func_est, tf.float32) * seq_mask # [b, n]
        accuracy = tf.reduce_sum(matched, axis=1) / tf.cast(valid_len, tf.float32) # [b]
        return {key: accuracy}


def get_train_test_data(test_dataset=None):
    all_datasets = ['beatles', 'salami', 'harmonix']
    assert test_dataset in all_datasets

    beatles_dir = './my_beatles_data'
    salami_dir = './my_salami_data'
    rwc_dir = './my_harmonix_data'
    dataset_dir = {'beatles': beatles_dir, 'salami': salami_dir, 'harmonix': rwc_dir}

    train_data = {}
    for dataset in all_datasets:
        if dataset == test_dataset:
            data_path = os.path.join(dataset_dir[dataset], 'test_data.npz')
            data = np.load(data_path, allow_pickle=True)
            test_data = {k: data[k] for k in data.files}
        else:
            data_path = os.path.join(dataset_dir[dataset], 'train_data.npz')
            data = np.load(data_path, allow_pickle=True)
            if len(train_data.keys()) == 0:
                train_data = {k: [] for k in data.files}
            [train_data[k].append(data[k]) for k in data.files]

        print(dataset, data[data.files[0]].shape)

    # Concat training data
    train_data = {k: np.concatenate(train_data[k]) for k in train_data.keys()}

    # Remove tracks with only one section or with too many silence frames (class=6)
    cond_train = [(len(set(seq)) > 1) and (sum(seq == 6) < seq.shape[0] * 0.2) for seq in train_data['function']]
    cond_test = [(len(set(seq)) > 1) and (sum(seq == 6) < seq.shape[0] * 0.2) for seq in test_data['function']]
    train_data = {k: train_data[k][cond_train] for k in train_data.keys()}
    test_data = {k: test_data[k][cond_test] for k in test_data.keys()}

    print('\ntrain dataset:', [dataset for dataset in all_datasets if dataset != test_dataset])
    [print((k, train_data[k].shape)) for k in sorted(train_data.keys())]

    [print((k, test_data[k].shape)) for k in sorted(test_data.keys())]
    return train_data, test_data


function_dict = {
    'intro': 0,
    'verse': 1,
    'chorus': 2,
    'bridge': 3,
    'inst': 4,
    'outro': 5,
    'silence': 6,
}

def class_conversion(i, reduced=True):
    if i == 4: return 'X'
    for k, v in function_dict.items():
        if v == i:
            return (k[0].upper()) if reduced else k.capitalize()


def format_cluster_sequence(seq):
    out = ''
    for c in seq:
        tag = class_conversion(c)
        sty = str(c%7 + 41)
        format = ';'.join([sty])
        out += '\x1b[%sm%s\x1b[0m' % (format, tag)
    return out


def print_temp(temp, sample=0, print_len=190):
    just_len = 15
    b_ref, b_est, matched = temp['b_ref'][sample], temp['b_est'][sample], temp['matched'][sample]
    n_b_ref, n_b_est, n_matched = temp['n_b_ref'][sample], temp['n_b_est'][sample], temp['n_matched'][sample]

    print('n_b_ref %d n_b_est %d n_matched %d' % (n_b_ref, n_b_est, n_matched))
    b_ref_in_second = temp['b_ref_in_second'][sample]
    b_est_in_second = temp['b_est_in_second'][sample]
    print('b_ref_in_second', ' '.join(["{:.2f}".format(s) for s in b_ref_in_second]))
    print('b_est_in_second', ' '.join(["{:.2f}".format(s) for s in b_est_in_second]))

    max_len = b_ref.shape[0]
    n_split = int(np.ceil(max_len/print_len))

    f_ref, f_est = temp['f_ref'][sample], temp['f_est'][sample]


def print_confusion_matrix(cm1, cm2=None, n_just=9, norm=True, epsilon=1e-7):
    n_classes = len(function_dict.keys())

    if norm:
        n_just = 6
        sum_row = np.sum(cm1, axis=1, keepdims=True)
        cm1_norm = cm1 / (sum_row + epsilon)
        if cm2 is not None:
            sum_row = np.sum(cm2, axis=1, keepdims=True)
            cm2_norm = cm2 / (sum_row + epsilon)

    def str_value(v):
        return '%.3f' % v if norm else '%d' % v

    def multi_display(vs, just):
        return '/'.join([str_value(v).rjust(just) for v in vs])

    if cm2 is None:
        if norm:
            rows = [class_conversion(i) + ''.join([str_value(v).rjust(n_just) for v in cm1_norm[i]]) for i in range(n_classes)]
        else:
            rows = [class_conversion(i) + ''.join([str_value(v).rjust(n_just) for v in cm1[i]]) for i in range(n_classes)]

    else:
        if norm:
            rows = [
                class_conversion(i) + ''.join([multi_display(vs, n_just) for vs in zip(cm1_norm[i], cm2_norm[i])]) for i in range(n_classes)
            ]
            col_n_just = 2 * n_just + 1
        else:
            rows = [
                class_conversion(i) + ''.join([multi_display(vs, n_just) for vs in zip(cm1_norm[i], cm2[i])]) for i in
                range(n_classes)
            ]
            col_n_just = n_just
    col_names = ''.join([class_conversion(i).rjust(col_n_just) for i in range(n_classes)])

    print('confusion matrix:')
    print('', col_names)
    [print(row) for row in rows]

    # Compute P, R, and F1 scores
    for cm in [cm1, cm2]:
        if cm is not None:
            sum_row = np.sum(cm, axis=1)
            sum_col = np.sum(cm, axis=0)
            diag = np.diag(cm)
            P = diag / (sum_col + epsilon)
            R = diag / (sum_row + epsilon)
            F1 = 2*P*R / (P + R + epsilon)
            print(' ', ''.join([class_conversion(i).rjust(n_just) for i in range(n_classes)]))
            print('P', ''.join([str_value(v).rjust(n_just) for v in P]))
            print('R', ''.join([str_value(v).rjust(n_just) for v in R]))
            print('F', ''.join([str_value(v).rjust(n_just) for v in F1]))


def color_text(color):
    color_dict = {
        "PURPLE": "\033[95m",
        "CYAN": "\033[96m",
        "DARKCYAN": "\033[36m",
        "BLUE": "\033[94m",
        "GREEN": "\033[92m",
        "YELLOW": "\033[93m",
        "RED": "\033[91m",
        "BOLD": "\033[1m",
        "UNDERLINE": "\033[4m",
        "END": "\033[0m"}
    return color_dict[color]


def train():
    print("Starting training with drum encoder and cross-attention...")
    all_datasets = ['beatles', 'salami', 'harmonix']
    test_dataset = all_datasets[0]
    save_model = True

    model_path = './my_model'
    train_data, test_data = get_train_test_data(test_dataset=test_dataset)

    print("After concatenation:")
    for key in ['len', 'section']:
        if key in train_data:
            print(f"{key}: shape={train_data[key].shape}, dtype={train_data[key].dtype}")
            print(f"  First element: {train_data[key][0]}, type={type(train_data[key][0])}")
            if hasattr(train_data[key][0], 'shape'):
                print(f"  First element shape: {train_data[key][0].shape}")

    # === MODIFIED: Updated generator for drum data with robust type conversion ===
    def generator(data):
        for spec, chromagram, drum_spec, drum_chromagram, valid_len, boundary, function, section in \
                zip(data['spec'], data['chromagram'], data['drum_spec'], data['drum_chromagram'],
                    data['len'], data['boundary'], data['function'], data['section']):
            

            # Ensure valid_len is a scalar integer
            if hasattr(valid_len, 'shape') and len(valid_len.shape) > 0:
                valid_len_scalar = int(valid_len[0]) if len(valid_len) > 0 else int(valid_len.item())
            else:
                valid_len_scalar = int(valid_len)
            
            # Ensure section is a scalar string
            if hasattr(section, 'shape') and len(section.shape) > 0:
                section_str = str(section[0]) if len(section) > 0 else "unknown"
            else:
                section_str = str(section)
            
            yield spec, chromagram, drum_spec, drum_chromagram, valid_len_scalar, boundary, function, section_str

    # === MODIFIED: Updated output types and shapes ===
    output_types = (tf.float32, tf.float32, tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.string)
    output_shapes = (
        tf.TensorShape([None, 80]),    # spec
        tf.TensorShape([None, 12]),    # chromagram
        tf.TensorShape([None, 80]),    # drum_spec
        tf.TensorShape([None, 12]),    # drum_chromagram
        tf.TensorShape([]),            # len (scalar) - THIS MUST BE SCALAR!
        tf.TensorShape([None]),        # boundary
        tf.TensorShape([None]),        # function class
        tf.TensorShape([]),            # section (scalar string) - THIS MUST BE SCALAR!
    )


    # Verify drum data is different from full song
    print("🔍 Verifying drum data is different from full song:")
    print(f"Full song mean: {np.mean(train_data['spec'][0]):.3f}")
    print(f"Drum spec mean: {np.mean(train_data['drum_spec'][0]):.3f}")
    print(f"Are they identical? {np.array_equal(train_data['spec'][0], train_data['drum_spec'][0])}")

    # === PLACEHOLDER: Add drum data (you need to replace this with real drum separation) ===
    # print("WARNING: Using placeholder drum data - you need to replace with real drum-separated features!")
    # train_data['drum_spec'] = train_data['spec'].copy()  # PLACEHOLDER - replace with real drum mel-spectrogram
    # train_data['drum_chromagram'] = train_data['chromagram'].copy()  # PLACEHOLDER - replace with real drum chromagram
    
    # test_data['drum_spec'] = test_data['spec'].copy()  # PLACEHOLDER
    # test_data['drum_chromagram'] = test_data['chromagram'].copy()  # PLACEHOLDER

    # Add section data if missing
    if 'section' not in train_data:
        train_data['section'] = np.array([f"track_{i:03d}" for i in range(len(train_data['spec']))], dtype='<U20')
    if 'section' not in test_data:
        test_data['section'] = np.array([f"track_{i:03d}" for i in range(len(test_data['spec']))], dtype='<U20')

    tf_train_data = tf.data.Dataset.from_generator(
        lambda: generator(train_data),
        output_types=output_types,
        output_shapes=output_shapes,
    )

    tf_test_data = tf.data.Dataset.from_generator(
        lambda: generator(test_data),
        output_types=output_types,
        output_shapes=output_shapes,
    )

    # === MODIFIED: Create new model with drum encoder ===
    model = FunctionalSegmentModel(
        max_len=935,
        n_units=80,
        n_heads=8,
        n_layers=2,
        cnn_dropout_rate=0.5,
        attn_dropout_rate=0.5,
    )

    # Build the model
    print("🔄 Building model with drum encoder...")
    dummy_spec = tf.zeros((1, 100, 80))
    dummy_chroma = tf.zeros((1, 100, 12))
    dummy_drum_spec = tf.zeros((1, 100, 80))
    dummy_drum_chroma = tf.zeros((1, 100, 12))
    dummy_len = tf.constant([100])
    _ = model(dummy_spec, dummy_chroma, dummy_drum_spec, dummy_drum_chroma, dummy_len, training=False)
    print("✅ Model built successfully!")

    # Checkpoint setup
    checkpoint = tf.train.Checkpoint(model=model)
    all_epochs_manager = tf.train.CheckpointManager(
        checkpoint, 
        directory=f'{model_path}/all_epochs', 
        max_to_keep=50,
        checkpoint_name='epoch'
    )
    best_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=f'{model_path}/best_models',
        max_to_keep=3,
        checkpoint_name='best'
    )
        
    TRAIN_BATCH_SIZE = 6
    TEST_BATCH_SIZE = 6
    TRAIN_SHUFFLE_SIZE = train_data['spec'].shape[0]
    N_EPOCHS = 100

    model.steps_per_epoch = int(np.ceil(TRAIN_SHUFFLE_SIZE / TRAIN_BATCH_SIZE))
    tf_train_data = tf_train_data.shuffle(TRAIN_SHUFFLE_SIZE, reshuffle_each_iteration=True)
    tf_train_data = tf_train_data.padded_batch(TRAIN_BATCH_SIZE, output_shapes)
    tf_test_data = tf_test_data.padded_batch(TEST_BATCH_SIZE, output_shapes)

    best_train_epoch, best_test_epoch = 0, 0
    supervised_metrics = ['F1_seg']
    best_train_result, best_test_result = {k: 0 for k in supervised_metrics}, {k: 0 for k in supervised_metrics}
    
    for epoch in range(1, N_EPOCHS+1):
        print('epoch', epoch)
        print(color_text("RED") + "--training phase--" + color_text("END"))
        
        for i_batch, batch in enumerate(tf_train_data):
            model.train_step(batch)
        print_temp(model.temp)
        result = model.average_result()
        print_confusion_matrix(model.confusion_matrix_train_max.numpy(), model.confusion_matrix_train_boun.numpy())
        model.clear_result()
        train_F1 = sum([float(result[k]) for k in supervised_metrics])
        if train_F1 > sum([float(best_train_result[k]) for k in supervised_metrics]):
            best_train_epoch, best_train_result = epoch, result
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if 'loss' in k]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k.endswith('seg')]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k.endswith('seg3')]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k.endswith('pair')]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k == 'Acc_max']))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k == 'Acc_smooth']))

        print(color_text("GREEN") + "--testing phase--" + color_text("END"))
        for i_batch, batch in enumerate(tf_test_data):
            model.test_step(batch)
        print_temp(model.temp, sample=np.random.randint(TEST_BATCH_SIZE))
        result = model.average_result()
        print_confusion_matrix(model.confusion_matrix_test_max.numpy(), model.confusion_matrix_test_boun.numpy())
        model.clear_result()
        test_F1 = sum([float(result[k]) for k in supervised_metrics])
        
        # Save every epoch
        if save_model:
            try:
                all_epochs_path = all_epochs_manager.save()
                print(f"💾 Saved epoch {epoch}: {all_epochs_path}")
            except Exception as e:
                print(f'❌ Epoch saving failed: {e}')

        # Save best model
        if test_F1 > sum([float(best_test_result[k]) for k in supervised_metrics]):
            best_test_epoch, best_test_result = epoch, result
            if save_model:
                print(color_text("YELLOW") + f"🏆 NEW BEST MODEL at epoch {epoch}!" + color_text("END"))
                try:
                    best_path = best_manager.save()
                    print(f"✅ Best model saved: {best_path}")
                except Exception as e:
                    print(f'❌ Best model saving failed: {e}')

        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if 'loss' in k]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k.endswith('seg')]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k.endswith('seg3')]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k.endswith('pair')]))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k == 'Acc_max']))
        print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in result.items() if k == 'Acc_smooth']))
        print(color_text("CYAN") + '### best_test_F1 at epoch %d' % best_test_epoch,
              '  '.join([' '.join((k, '{:.3f}'.format(best_test_result[k].numpy()))) for k in supervised_metrics]), color_text("END"))
        print()

    print('best_train_epoch:', best_train_epoch)
    print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in best_train_result.items() if 'l' in k]))
    print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in best_train_result.items() if k == 'Acc_max']))
    print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in best_train_result.items() if k == 'Acc_smooth']))
    print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in best_train_result.items() if k.endswith('seg')]))
    print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in best_train_result.items() if k.endswith('seg3')]))
    print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in best_train_result.items() if k.endswith('pair')]))

    print('best_test_epoch:', best_test_epoch)
    print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in best_test_result.items() if 'l' in k]))
    print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in best_test_result.items() if k == 'Acc_max']))
    print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in best_test_result.items() if k == 'Acc_smooth']))
    print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in best_test_result.items() if k.endswith('seg')]))
    print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in best_test_result.items() if k.endswith('seg3')]))
    print('  '.join([' '.join((k.ljust(8), '{:.3f}'.format(v.numpy()))) for k, v in best_test_result.items() if k.endswith('pair')]))


def segmentFrame2interval(segment_frame, frame_size=0.5):
    segment_frame = np.array(segment_frame)
    segment_frame[0] = 1
    segment_frame = np.append(segment_frame, [1])
    boundary = np.where(segment_frame == 1)[0]
    interval = np.array(list(zip(boundary[:-1], boundary[1:]))) * frame_size
    return interval


def frame2interval(segment_frame, label_frame, frame_size=0.5):
    segment_frame = np.array(segment_frame)
    label_frame = np.array(label_frame)
    segment_frame[0] = 1
    label = label_frame[segment_frame == 1]
    segment_frame = np.append(segment_frame, [1])
    boundary = np.where(segment_frame == 1)[0]
    interval = np.array(list(zip(boundary[:-1], boundary[1:]))) * frame_size
    return interval, label

def main():
    train()
    
if __name__ == '__main__':
    main()