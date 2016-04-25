# Keras backends

## What is a "backend"?

Keras is a model-level library, providing high-level building blocks for developing deep learning models. It does not handle itself low-level operations such as tensor products, convolutions and so on. Instead, it relies on a specialized, well-optimized tensor manipulation library to do so, serving as the "backend engine" of Keras. Rather than picking one single tensor library and making the implementation of Keras tied to that library, Keras handles the problem in a modular way, and several different backend engines can be plugged seamlessly into Keras.

At this time, Keras has two backend implementations available: the **Theano** backend and the **TensorFlow** backend.

- [Theano](http://deeplearning.net/software/theano/) is an open-source symbolic tensor manipulation framework developed by LISA/MILA Lab at Université de Montréal.
- [TensorFlow](http://www.tensorflow.org/) is an open-source symbolic tensor manipulation framework developed by Google, Inc.

----

## Switching from one backend to another

If you have run Keras at least once, you will find the Keras configuration file at:

`~/.keras/keras.json`

If it isn't there, you can create it.

It probably looks like this:

`{"epsilon": 1e-07, "floatx": "float32", "backend": "theano"}`

Simply change the field `backend` to either `"theano"` or `"tensorflow"`, and Keras will use the new configuration next time you run any Keras code.

You can also define the environment variable ``KERAS_BACKEND`` and this will
override what is defined in your config file :

```bash
KERAS_BACKEND=tensorflow python -c "from keras import backend; print backend._BACKEND"
Using TensorFlow backend.
tensorflow
```

----

## Using the abstract Keras backend to write new code

If you want the Keras modules you write to be compatible with both Theano and TensorFlow, you have to write them via the abstract Keras backend API. Here's an intro.

You can import the backend module via:
```python
from keras import backend as K
```

The code below instantiates an input placeholder. It's equivalent to `tf.placeholder()` or `T.matrix()`, `T.tensor3()`, etc.

```python
input = K.placeholder(shape=(2, 4, 5))
# also works:
input = K.placeholder(shape=(None, 4, 5))
# also works:
input = K.placeholder(ndim=3)
```

The code below instantiates a shared variable. It's equivalent to `tf.variable()` or `theano.shared()`.

```python
val = np.random.random((3, 4, 5))
var = K.variable(value=val)

# all-zeros variable:
var = K.zeros(shape=(3, 4, 5))
# all-ones:
var = K.ones(shape=(3, 4, 5))
```

Most tensor operations you will need can be done as you would in TensorFlow or Theano:

```python
a = b + c * K.abs(d)
c = K.dot(a, K.transpose(b))
a = K.sum(b, axis=2)
a = K.softmax(b)
a = concatenate([b, c], axis=-1)
# etc...
```

----

## Backend functions


### floatx


```python
floatx()
```


Returns the default float type, as a string
(e.g. 'float16', 'float32', 'float64').

----

### cast_to_floatx


```python
cast_to_floatx(x)
```


Cast a Numpy array to floatx.

----

### variable


```python
variable(value, dtype='float32', name=None)
```


Instantiate a tensor variable.

----

### placeholder


```python
placeholder(shape=None, ndim=None, dtype='float32', name=None)
```


Instantiate an input data placeholder variable.

----

### shape


```python
shape(x)
```


Return the shape of a tensor.

- __Warning__: type returned will be different for
Theano backend (Theano tensor type) and TF backend (TF TensorShape).

----

### eval


```python
eval(x)
```


Run a graph.

----

### zeros


```python
zeros(shape, dtype='float32', name=None)
```


Instantiate an all-zeros variable.

----

### ones


```python
ones(shape, dtype='float32', name=None)
```


Instantiate an all-ones variable.

----

### count_params


```python
count_params(x)
```


Return number of scalars in a tensor.

- __Return__: numpy integer.

----

### batch_dot


```python
batch_dot(x, y, axes=None)
```


batchwise dot product
batch_dot results in a tensor with less dimensions than the input.
If the number of dimensions is reduced to 1, we use `expand_dims` to
make sure that ndim is at least 2.

__Example__

Assume x = [[1, 2]   and y = [[5, 6]
	[3, 4]]   [7, 8]]
batch_dot(x, y, axes=1) = [[17, 53]] which is the main diagonal
of x.dot(y.T), although we never have to calculate the off-diagonal
elements.


__Arguments__

x, y: tensors with ndim >= 2
- __axes__: list (or single) int with target dimensions

__Returns__

Tensor with ndim >= 2

----

### gather


```python
gather(reference, indices)
```


reference: a tensor.
- __indices__: an int tensor of indices.

- __Return__: a tensor of same type as reference.

----

### sum


```python
sum(x, axis=None, keepdims=False)
```


Sum of the values in a tensor, alongside the specified axis.

----

### prod


```python
prod(x, axis=None, keepdims=False)
```


Multiply the values in a tensor, alongside the specified axis.

----

### any


```python
any(x, axis=None, keepdims=False)
```


Bitwise reduction (logical OR).

----

### permute_dimensions


```python
permute_dimensions(x, pattern)
```


Transpose dimensions.

pattern should be a tuple or list of
dimension indices, e.g. [0, 2, 1].

----

### repeat_elements


```python
repeat_elements(x, rep, axis)
```


Repeat the elements of a tensor along an axis, like np.repeat.

If x has shape (s1, s2, s3) and axis=1, the output
will have shape (s1, s2 * rep, s3).

----

### resize_images


```python
resize_images(X, height_factor, width_factor, dim_ordering)
```


Resize the images contained in a 4D tensor of shape
- [batch, channels, height, width] (for 'th' dim_ordering)
- [batch, height, width, channels] (for 'tf' dim_ordering)
by a factor of (height_factor, width_factor). Both factors should be
positive integers.

----

### resize_volumes


```python
resize_volumes(X, depth_factor, height_factor, width_factor, dim_ordering)
```


Resize the volume contained in a 5D tensor of shape
- [batch, channels, depth, height, width] (for 'th' dim_ordering)
- [batch, depth, height, width, channels] (for 'tf' dim_ordering)
by a factor of (depth_factor, height_factor, width_factor).
Both factors should be positive integers.

----

### repeat


```python
repeat(x, n)
```


Repeat a 2D tensor.

If x has shape (samples, dim) and n=2,
the output will have shape (samples, 2, dim).

----

### batch_flatten


```python
batch_flatten(x)
```


Turn a n-D tensor into a 2D tensor where
the first dimension is conserved.

----

### expand_dims


```python
expand_dims(x, dim=-1)
```


Add a 1-sized dimension at index "dim".

----

### squeeze


```python
squeeze(x, axis)
```


Remove a 1-dimension from the tensor at index "axis".

----

### temporal_padding


```python
temporal_padding(x, padding=1)
```


Pad the middle dimension of a 3D tensor
with "padding" zeros left and right.

Appologies for the inane API, but Theano makes this
really hard.

----

### spatial_2d_padding


```python
spatial_2d_padding(x, padding=(1, 1), dim_ordering='th')
```


Pad the 2nd and 3rd dimensions of a 4D tensor
with "padding[0]" and "padding[1]" (resp.) zeros left and right.

----

### spatial_3d_padding


```python
spatial_3d_padding(x, padding=(1, 1, 1), dim_ordering='th')
```


Pad the 2nd, 3rd and 4th dimensions of a 5D tensor
with "padding[0]", "padding[1]" and "padding[2]" (resp.) zeros left and right.

----

### rnn


```python
rnn(step_function, inputs, initial_states, go_backwards=False, mask=None, constants=None, unroll=False, input_length=None)
```


Iterates over the time dimension of a tensor.

__Arguments__

- __inputs__: tensor of temporal data of shape (samples, time, ...)
(at least 3D).
- __step_function__:
- __Parameters__:
	- __input__: tensor with shape (samples, ...) (no time dimension),
	representing input for the batch of samples at a certain
	time step.
	- __states__: list of tensors.
- __Returns__:
	- __output__: tensor with shape (samples, ...) (no time dimension),
	- __new_states__: list of tensors, same length and shapes
	as 'states'.
- __initial_states__: tensor with shape (samples, ...) (no time dimension),
containing the initial values for the states used in
the step function.
- __go_backwards__: boolean. If True, do the iteration over
the time dimension in reverse order.
- __mask__: binary tensor with shape (samples, time),
with a zero for every element that is masked.
- __constants__: a list of constant values passed at each step.
- __unroll__: whether to unroll the RNN or to use a symbolic loop (`scan`).
- __input_length__: must be specified if using `unroll`.

__Returns__

A tuple (last_output, outputs, new_states).
- __last_output__: the latest output of the rnn, of shape (samples, ...)
- __outputs__: tensor with shape (samples, time, ...) where each
	entry outputs[s, t] is the output of the step function
	at time t for sample s.
- __new_states__: list of tensors, latest states returned by
	the step function, of shape (samples, ...).

----

### switch


```python
switch(condition, then_expression, else_expression)
```


condition: scalar tensor.

----

### conv2d


```python
conv2d(x, kernel, strides=(1, 1), border_mode='valid', dim_ordering='th', image_shape=None, filter_shape=None)
```



- __border_mode__: string, "same" or "valid".

----

### conv3d


```python
conv3d(x, kernel, strides=(1, 1, 1), border_mode='valid', dim_ordering='th', volume_shape=None, filter_shape=None)
```



Run on cuDNN if available.
- __border_mode__: string, "same" or "valid".






