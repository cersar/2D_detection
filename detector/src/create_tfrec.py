import tensorflow as tf
import numpy as np

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def serialize_example(image_id, image_data, label):
    feature = {
        'image_id': _bytes_feature(image_id),
        'image': _bytes_feature(image_data),
        'label': _float_feature(label)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

class_num = 2
with tf.io.TFRecordWriter('../../test.tfrec') as f:
    for i in range(64):
        image_id = 'test{}'.format(i)
        box_num = np.random.randint(1,11)
        label = np.zeros((box_num,5),np.float32)
        for j in range(box_num):
            label[j,0] = np.random.randint(0,class_num)
            label[j,1:] = np.random.rand(4)
        label = label.reshape(box_num*5)
        image_data = tf.image.encode_jpeg(tf.cast(tf.random.uniform((512,512,3),0,256,dtype=tf.int32),tf.uint8))
        example = serialize_example(image_id.encode(), image_data, label)
        f.write(example)