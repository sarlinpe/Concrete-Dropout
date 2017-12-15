# Concrete Dropout

TensorFlow implementation of Concrete Dropout, as presented in the paper: "Concrete Dropout." Yarin Gal, Jiri Hron, Alex Kendall. [ArXiv 2017](https://arxiv.org/abs/1705.07832).



## Usage

The concrete dropout layer should precede a convolutional layer in order to feed the kernel regulariser in.

```python
from concre_dropout import concrete_dropout
from tensorflow import layers as tfl

...

dropped, reg = concrete_dropout(
	inputs,
	weight_regularizer=wr,
	dropout_regularizer=dr,
	init_min=init_rate,
	init_max=init_rate,
	name='concrete_dropout',
	reuse=reuse,
	training=training)
outputs = tfl.conv2d(
	dropped,
	filters,
	kernel_size,
	kernel_regularizer=reg,
	name='convolution',
	reuse=reuse)
```

Don't forget to add the regularisers to your loss when you're done with building the graph:

```python
loss += tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
```

Finally, you can get summaries for the learned rates with: 

```python
rates = tf.get_collection('DROPOUT_RATES')
for r in rates:
	tf.summary.scalar(r.name, r)
```



## Credits

[Original implementation](https://github.com/yaringal/ConcreteDropout) by Yarin Gal.