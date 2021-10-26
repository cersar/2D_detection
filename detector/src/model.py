from efficientnet.tfkeras import EfficientNetB0
import tensorflow as tf
from detector.src.anchors import gen_anchors

FEATURE_LAYER_NAMES_DICT = {'EfficientNetB0':['block3b_add','block5c_add','block7a_project_bn']}

def fuse_conv(fpn_len):
    return tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(fpn_len,3,use_bias=True,padding='same'),
        tf.keras.layers.BatchNormalization(),
        ])


class BiFPN_Block(tf.keras.layers.Layer):
    def __init__(self,fpn_levels, fpn_len, **kwargs):
        super(BiFPN_Block, self).__init__(**kwargs)
        self.fpn_levels = fpn_levels
        self.fpn_len = fpn_len
        self.fuse_conv_1 = []
        self.fuse_conv_2 = []
        self.up_sample = tf.keras.layers.UpSampling2D()
        self.down_sample = tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')
        for i in self.fpn_levels[1:-1]:
            self.fuse_conv_1.append(fuse_conv(fpn_len))
        for i in self.fpn_levels:
            self.fuse_conv_2.append(fuse_conv(fpn_len))

        self.act = tf.keras.activations.swish

        self.w_l1 = []
        self.w_l2 = []
        for i in self.fpn_levels[1:-1]:
            w = []
            w.append(self.add_weight(name='w{}_1_1'.format(i)))
            w.append(self.add_weight(name='w{}_1_2'.format(i)))
            self.w_l1.append(w)

        for i in self.fpn_levels:
            w = []
            w.append(self.add_weight(name='w{}_2_1'.format(i)))
            w.append(self.add_weight(name='w{}_2_2'.format(i)))
            if i != self.fpn_levels[0] and i != self.fpn_levels[-1]:
                w.append(self.add_weight(name='w{}_2_3'.format(i)))
            self.w_l2.append(w)

    def build(self, input_shape):
        assert len(input_shape) == len(self.fpn_levels)


    def call(self, inputs):
        tf.assert_equal(len(inputs),len(self.fpn_levels))
        outputs_1 = []
        outputs_2 = []
        pre = inputs[-1]
        P1_inputs = inputs[1:-1]
        for i in reversed(range(len(P1_inputs))):
            fuse = tf.nn.relu(self.w_l1[i][0])*P1_inputs[i] +tf.nn.relu(self.w_l1[i][1])*self.up_sample(pre)
            P_1 = self.fuse_conv_1[i](self.act(fuse))
            outputs_1.insert(0,P_1)
            pre = P_1
        P2_inputs = inputs
        pre = None

        for i in range(len(P2_inputs)):
            if i == 0:
                fuse = tf.nn.relu(self.w_l2[i][0]) * P2_inputs[i] + tf.nn.relu(self.w_l2[i][1]) * self.up_sample(outputs_1[0])
            elif i == len(P2_inputs)-1:
                fuse = tf.nn.relu(self.w_l2[i][0]) * P2_inputs[i] + tf.nn.relu(self.w_l2[i][1]) * self.down_sample(pre)
            else:
                fuse = tf.nn.relu(self.w_l2[i][0]) * P2_inputs[i] + tf.nn.relu(self.w_l2[i][1]) * outputs_1[i-1] + tf.nn.relu(self.w_l2[i][2]) * self.down_sample(pre)
            P_2 = self.fuse_conv_2[i](self.act(fuse))
            outputs_2.append(P_2)
            pre = P_2
        return outputs_2

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'fpn_levels': self.fpn_levels,
            'fpn_len': self.fpn_len,
        })
        return config

def get_backbone(backbone_name,input_shape,weights='noisy-student'):
    if backbone_name == 'EfficientNetB0':
        backbone = EfficientNetB0(input_shape=input_shape, weights=weights, include_top=False, drop_connect_rate=0.0)
    else:
        raise ValueError('Invalid backbone {} !'.format(backbone_name))
    return backbone

class RetinaClsHead(tf.keras.layers.Layer):
    def __init__(self, num_classes, num_anchors, neck_width, detect_head_len, **kwargs):
        super(RetinaClsHead, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.neck_width = neck_width
        self.detect_head_len = detect_head_len
        self.head = tf.keras.layers.SeparableConv2D(num_anchors * num_classes, 3, use_bias=True,
                                               padding='same')
        self.detect_convs = []
        for i in range(detect_head_len):
            self.detect_convs.append(tf.keras.layers.SeparableConv2D(neck_width, 3, use_bias=True, padding='same'))

        self.act = tf.keras.activations.swish

    def build(self, input_shape):
        print(input_shape)
        self.bns = []
        for i in range(self.detect_head_len):
            tmp = []
            for j in range(len(input_shape)):
                tmp.append(tf.keras.layers.BatchNormalization())
            self.bns.append(tmp)


    def call(self, inputs):
        features = inputs.copy()
        for i in range(self.detect_head_len):
            for j in range(len(features)):
                features[j] = self.detect_convs[i](features[j])
                features[j] = self.bns[i][j](features[j])
                features[j] = self.act(features[j])
        for i in range(len(features)):
            x = self.head(features[i])
            features[i] = tf.reshape(x, (-1, x.shape[1] * x.shape[2] * self.num_anchors, self.num_classes))

        output = tf.concat(features, axis=1)
        output = tf.sigmoid(output)
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_classes': self.num_classes,
            'num_anchors': self.num_anchors,
            'neck_width': self.neck_width,
            'detect_head_len': self.detect_head_len,
        })
        return config


class RetinaRegHead(tf.keras.layers.Layer):
    def __init__(self, num_anchors, neck_width, detect_head_len, **kwargs):
        super(RetinaRegHead, self).__init__(**kwargs)
        self.num_anchors = num_anchors
        self.neck_width = neck_width
        self.detect_head_len = detect_head_len
        self.head = tf.keras.layers.SeparableConv2D(num_anchors * 4, 3, use_bias=True,
                                               padding='same')
        self.detect_convs = []
        for i in range(detect_head_len):
            self.detect_convs.append(tf.keras.layers.SeparableConv2D(neck_width, 3, use_bias=True, padding='same'))

        self.act = tf.keras.activations.swish

    def build(self, input_shape):
        print(input_shape)
        self.bns = []
        for i in range(self.detect_head_len):
            tmp = []
            for j in range(len(input_shape)):
                tmp.append(tf.keras.layers.BatchNormalization())
            self.bns.append(tmp)


    def call(self, inputs):
        features = inputs.copy()
        for i in range(self.detect_head_len):
            for j in range(len(features)):
                features[j] = self.detect_convs[i](features[j])
                features[j] = self.bns[i][j](features[j])
                features[j] = self.act(features[j])
        for i in range(len(features)):
            x = self.head(features[i])
            features[i] = tf.reshape(x, (-1, x.shape[1] * x.shape[2] * self.num_anchors, 4))

        output = tf.concat(features, axis=1)
        output = tf.sigmoid(output)
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_anchors': self.num_anchors,
            'neck_width': self.neck_width,
            'detect_head_len': self.detect_head_len,
        })
        return config

def get_detect_head(CFG):
    if CFG['detect_head'] == 'retina':
        cls_head = RetinaClsHead(CFG['num_classes'],CFG['num_anchors'], CFG['neck_width'], CFG['detect_head_len'])
        reg_head = RetinaRegHead(CFG['num_anchors'], CFG['neck_width'], CFG['detect_head_len'])
    else:
        raise ValueError('Invalid detect_head {} !'.format(CFG['detect_head']))
    return cls_head,reg_head

# def detect_cls_head(inputs,CFG):
#     features = inputs.copy()
#     if CFG['detect_head'] == 'retina':
#         for i in range(CFG['detect_head_len']):
#             detect_conv = tf.keras.layers.SeparableConv2D(CFG['neck_width'],3,use_bias=True,padding='same')
#             for j in range(len(features)):
#                 features[j] = detect_conv(features[j])
#                 features[j] = tf.keras.layers.BatchNormalization()(features[j])
#                 features[j] = tf.keras.activations.swish(features[j])
#         head = tf.keras.layers.SeparableConv2D(CFG['num_anchors']*CFG['num_classes'], 3, use_bias=True, padding='same')
#         for i in range(len(features)):
#             x = head(features[i])
#             features[i] = tf.reshape(x,(-1,x.shape[1]*x.shape[2]*CFG['num_anchors'],CFG['num_classes']))
#
#         output = tf.concat(features,axis=1)
#         output = tf.sigmoid(output)
#     else:
#         raise ValueError('Invalid detect_head {} !'.format(CFG['detect_head']))
#     return output

# def detect_reg_head(inputs,CFG):
#     features = inputs.copy()
#     if CFG['detect_head'] == 'retina':
#         for i in range(CFG['detect_head_len']):
#             detect_conv = tf.keras.layers.SeparableConv2D(CFG['neck_width'],3,use_bias=True,padding='same')
#             for j in range(len(features)):
#                 features[j] = detect_conv(features[j])
#                 features[j] = tf.keras.layers.BatchNormalization()(features[j])
#                 features[j] = tf.keras.activations.swish(features[j])
#         head = tf.keras.layers.SeparableConv2D(CFG['num_anchors']*4, 3, use_bias=True, padding='same')
#         for i in range(len(features)):
#             x = head(features[i])
#             features[i] = tf.reshape(x,(-1,x.shape[1]*x.shape[2]*CFG['num_anchors'],4))
#         output = tf.concat(features,axis=1)
#     else:
#         raise ValueError('Invalid detect_head {} !'.format(CFG['detect_head']))
#     return output

def detect_neck(features,CFG):
    if CFG['neck'] == 'bifpn':
        for i in range(CFG['neck_len']):
            features = BiFPN_Block(CFG['pyramid_levels'],CFG['neck_width'], name='bifpn_block{}'.format(i))(features)
    else:
        raise ValueError('Invalid neck {} !'.format(CFG['neck']))

    return features


def get_detector(CFG):
    backbone = get_backbone(CFG['backbone_name'],CFG['input_shape'])
    feature_layer_names = FEATURE_LAYER_NAMES_DICT[CFG['backbone_name']]
    assert len(CFG['pyramid_levels'])>=len(feature_layer_names)

    features = []
    for layer_name in feature_layer_names:
        x = backbone.get_layer(layer_name).output
        x = tf.keras.layers.Conv2D(CFG['neck_width'], 1, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        features.append(x)

    x = backbone.get_layer(feature_layer_names[-1]).output

    for i in range(len(CFG['pyramid_levels'])-len(feature_layer_names)):
        x = tf.keras.layers.Conv2D(CFG['neck_width'], 1, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        features.append(x)

    features = detect_neck(features,CFG)

    cls_head,reg_head = get_detect_head(CFG)
    cls_output = cls_head(features)
    bbox_output = reg_head(features)
    outputs = tf.concat([cls_output,bbox_output],-1)

    model = tf.keras.Model(backbone.inputs, outputs=outputs)

    anchors = gen_anchors(CFG)
    return model,anchors