import numpy as np
import tensorflow as tf

import lucid.modelzoo.vision_models as models
from lucid.misc.channel_reducer import ChannelReducer
import lucid.optvis.param as param
import lucid.optvis.objectives as objectives
import lucid.optvis.render as render
from lucid.misc.io import show, load
from lucid.misc.io.showing import _image_url
from lucid.misc.gradient_override import gradient_override_map

from lucid.modelzoo.vision_base import Model

def raw_class_group_attr(img, layer, label, group_vecs, output, override=None):
  """How much did spatial positions at a given layer effect a output class?"""

  # Set up a graph for doing attribution...
  with tf.Graph().as_default(), tf.Session(), gradient_override_map(override or {}):
    t_input = tf.placeholder_with_default(img, [None, None, 3])
    T = render.import_model(model, t_input, t_input)
    
    # Compute activations
    acts = T(layer).eval()
    
    if label is None: return np.zeros(acts.shape[1:-1])
    
    # Compute gradient
    score = T(output)[0, labels.index(label)]
    t_grad = tf.gradients([score], [T(layer)])[0]   
    grad = t_grad.eval({T(layer) : acts})
    
    # Linear approximation of effect of spatial position
    return [np.sum(group_vec * grad) for group_vec in group_vecs]

def neuron_groups(img, layer, n_groups=6, attr_classes=[]):
  # Compute activations
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

  spatial = [factor[..., None]/np.percentile(spatial_factors,99)*[1,0,0] for factor in spatial_factors]
  import cv2
  folder = 'out' + model.model_path + "_" + model.input_name + "_" + layer
  folder = folder.replace("/", "_")
  import os
  if not os.path.exists(folder):
    os.mkdir(folder)
  os.chdir(folder)
  scale = lambda k, x : cv2.resize(x, (x.shape[1] * k, x.shape[0] * k))
  for i in range(len(group_icons)):
    cv2.imwrite('group' + str(i) + '.png', np.round(scale(3, group_icons[i] * 255)).astype(int))
    cv2.imwrite('space' + str(i) + '.png', np.round(scale(8, spatial[i] * 255)).astype(int))
  os.chdir('..')

  
  if SSD:
    group_vecs = [spatial_factors[i, ..., None]*channel_factors[i]
                for i in range(n_groups)]
    attr_list = list()
    for i in range(1, 2):
      output = 'BoxPredictor_' + str(i) + '/ClassPredictor/BiasAdd' 
      attr_list.append(np.asarray([raw_class_group_attr(img, layer, attr_class, group_vecs, output) for attr_class in attr_classes]).astype(int))
    print(attr_list)


class SSDMobilenet_v1(Model):
  #model_path = '/home/thanatcha/lucid/mobilenetv1_graphdef_frozen.pb.modelzoo'
  #model_path = '/home/thanatcha/object_recognition/log/july5_dann_both_on_real2/trained/frozen_inference_graph.pb'
  model_path = '/home/thanatcha/object_recognition/models/sim_on_sim/trained/frozen_inference_graph.pb'
  labels_path = '/home/thanatcha/object_recognition/data/classes.txt'
  image_shape = [480, 640, 3]
  image_value_range = (-1, 1)
  input_name = 'Preprocessor/mul'

labels = ['tube', 'scrap', 'screwdriver', 'tape', 'hammer', 'wrench']
SSD = False
SSD = True
if SSD:
  model = SSDMobilenet_v1()
  model.load_graphdef()
  #img = load("/home/thanatcha/object_recognition/uncluttered+cluttered/image_rgb/rgb_raw_0001.png")
  img = load("/home/thanatcha/object_recognition/dataset_07_05_2018_2/rgb_1.png")
  neuron_groups(img, "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6", 8, ["screwdriver", "tape", "tube", "scrap"])
else:
  model = models.InceptionV1()
  model.load_graphdef()
  img = load("https://storage.googleapis.com/lucid-static/building-blocks/examples/dog_cat.png")
  neuron_groups(img, "mixed4d", 6, ["Labrador retriever", "tiger cat"])
