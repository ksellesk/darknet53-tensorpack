import tensorflow as tf


def quantize(x, k):
    n = float(2 ** k - 1)

    @tf.custom_gradient
    def _quantize(x):
        return tf.round(x * n) / n, lambda dy: dy

    return _quantize(x)


def bit_act(inputs, bits=8):
    x = tf.clip_by_value(inputs, 0.0, 1.0)
    return quantize(x, bits)


def quantize_getter_fn(conv_qbit=None, bn_qbit=None, fc_qbit=None, specified_layers=None, except_layers=None):
    def _quant_weight(x):
        name = x.op.name

        if "moving_mean" in name or "moving_var" in name:
            return x

        if except_layers:
            for _el in except_layers:
                if _el in name:
                    print("except_layer", name)
                    return x

        qbit = conv_qbit if "conv" in name else None
        qbit = fc_qbit if "dense" in name else qbit
        qbit = bn_qbit if "batch_normalization" in name else qbit
        if specified_layers:
            qbit = specified_layers.get(name) or qbit

        if qbit is None or qbit >= 32 or qbit < 1:
            return x

        if qbit == 1:
            E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))

            @tf.custom_gradient
            def _sign(x):
                return tf.sign(x / E) * E, lambda dy: dy

            return _sign(x)

        x = tf.tanh(x)
        x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
        return 2 * quantize(x, qbit) - 1


    def getter(getter, *args, **kwargs):
        x = getter(*args, **kwargs)
        return _quant_weight(x)

    return getter
