import numpy as np
import tensorflow as tf

import lucid.modelzoo.vision_models as models
from lucid.misc.channel_reducer import ChannelReducer
import lucid.optvis.param as param
import lucid.optvis.objectives as objectives
import lucid.optvis.render as render
from lucid.misc.io import show, load
from lucid.misc.gradient_override import gradient_override_map

SSD = False
#SSD = True
import pdb; pdb.set_trace()
if SSD:
  model = models.SSDMobilenet_v1()
else:
  model = models.InceptionV1()
model.load_graphdef()

def neuron_groups(img, layer, n_groups=6, attr_classes=[]):
  # Compute activations
  import pdb; pdb.set_trace()
  with tf.Graph().as_default(), tf.Session():
    t_input = tf.placeholder_with_default(img, [None, None, 3])
    T = render.import_model(model, t_input, t_input)
    acts = T(layer).eval()

  # We'll use ChannelReducer (a wrapper around scikit learn's factorization tools)
  # to apply Non-Negative Matrix factorization (NMF).
  nmf = ChannelReducer(n_groups, "NMF")
  spatial_factors = nmf.fit_transform(acts)[0].transpose(2, 0, 1).astype("float32")
  channel_factors = nmf._reducer.components_.astype("float32")

  # Let's organize the channels based on their horizontal position in the image
  x_peak = np.argmax(spatial_factors.max(1), 1)
  ns_sorted = np.argsort(x_peak)
  spatial_factors = spatial_factors[ns_sorted]
  channel_factors = channel_factors[ns_sorted]

  # And create a feature visualziation of each group
  param_f = lambda: param.image(80, batch=n_groups)
  obj = sum(objectives.direction(layer, channel_factors[i], batch=i)
            for i in range(n_groups))
  group_icons = render.render_vis(model, obj, param_f, verbose=False)[-1]
  
  import cv2
  for i in range(len(group_icons)):
    png = (np.round(group_icons[i] * 255)).astype(int)
    cv2.imwrite('group' + str(i) + '.png', png)
  
  
if SSD:
  img = load("/home/thanatcha/object_recognition/uncluttered+cluttered/image_rgb/rgb_raw_0001.png")
  neuron_groups(img, "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6", 6)
else:
  img = load("https://storage.googleapis.com/lucid-static/building-blocks/examples/dog_cat.png")
  neuron_groups(img, "mixed4d", 6)
