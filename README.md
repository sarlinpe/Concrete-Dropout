# Concrete Dropout

TensorFlow implementation of Concrete Dropout, as presented in the paper: "Concrete Dropout." Yarin Gal, Jiri Hron, Alex Kendall. [ArXiv 2017](https://arxiv.org/abs/1705.07832).


## Usage

The concrete dropout layer exposes the same API as other classes in `tf.layers`. It additionally returns a kernel/bias regularizer, which should be fed into the subsequent dense or convolutional layer. When the `training` argument is set to `False`, the layer is disabled, which is equivalent to setting the dropout rate to 0. See `mnist.py` for a concrete (!) example.

```python
from concrete_dropout import concrete_dropout
from tensorflow import layers as tfl

...

dropped, reg = concrete_dropout(
	inputs,
	weight_regularizer=wr,
	dropout_regularizer=dr,
	init_min=init_rate,
	init_max=init_rate,
	name='concrete_dropout',
	training=training)
outputs = tfl.conv2d(
	dropped,
	filters,
	kernel_size,
	kernel_regularizer=reg,
	name='conv')
```

Don't forget to add the regularisers to your loss when you're done with building the graph:

```python
loss += tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
```

Finally, you can obtain the summaries for the learned rates with:

```python
rates = tf.get_collection('DROPOUT_RATES')
for r in rates:
	tf.summary.scalar(r.name, r)
```


## Credits

[Original implementation](https://github.com/yaringal/ConcreteDropout) by Yarin Gal.
