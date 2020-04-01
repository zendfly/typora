### learning.py

包含用于训练模型的各种函数，主要有三种：

- mainpulation gradients
- creating a 'train_op'
- a training loop function

loop function和optimization去优化参数

#### `def clip_gradient_norms(gradients_to_variables,max_norm)`

用给定的value裁剪（跟新）梯度





### `def train_step(sess,train_op,global_step,train_step_kwargs):`

执行梯度步长并判断是否停止。



return: the total loss and a boolean indicating whether or not to stop training.





