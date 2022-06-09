# 场景检测

## 开发
Tensorflow 2.4 + opencv 

利用keras快速构建模型。

## 问题记录

1. opencv 加载模型问题
cv2.cnn.readNet('./model/saved_model.pb', 'saved_model.pbtxt')

自升级tensorflow 2.0后不在支持pbtxt生成。对于网络上所提供的生成pbtxt办法都是从原有的pb文件中读取graph然后写入，但现在pb文件中也已经不再携带graph。
参考至opencv文档：[here](https://answers.opencv.org/questions/198805/revisions/)

核心代码如下：
```python
def write_graph_to_pb():
    m = tf.saved_model.load('./model/')

    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    tfm = tf.function(lambda x: m(x))  # full model
    tfm = tfm.get_concrete_function(tf.TensorSpec(m.signatures['serving_default'].inputs[0].shape.as_list(),
                                                  m.signatures['serving_default'].inputs[0].dtype.name))
    frozen_func = convert_variables_to_constants_v2(tfm)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir="./model/", name="saved_model_graph.pb", as_text=False)

```

生成的saved_model_graph.pb文件可以直接用于cv.cnn.readNet();但是经测试无法正常使用，无报错，但是与预测值不符合预期。
最后还是使用`keras.models.load_model`加载模型进行预测。

2. 模型预测效率较低，现在仅仅是分类图片，明显感觉卡顿。

    解决方案：
   1. UI线程不参与预测等CPU密集任务
   2. 所有耗时操作在子线程执行，最终回调更新UI


