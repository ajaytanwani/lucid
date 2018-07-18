import numpy as np
import tensorflow as tf

import lucid.modelzoo.vision_models as models
from lucid.misc.io import show
import lucid.misc.io.showing as showing
from lucid.misc.channel_reducer import ChannelReducer
import lucid.optvis.param as param
import lucid.optvis.objectives as objectives
import lucid.optvis.render as render
from lucid.misc.io import show, load
from lucid.misc.io.reading import read
from lucid.misc.io.showing import _image_url
from lucid.misc.gradient_override import gradient_override_map

model = models.InceptionV1()
model.load_graphdef()

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
  

img = load("https://storage.googleapis.com/lucid-static/building-blocks/examples/dog_cat.png")
neuron_groups(img, "mixed4d", 6, ["Labrador retriever", "tiger cat"])
