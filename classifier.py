

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from flask_socketio import send, emit
import json

class MyEncoder(json.JSONEncoder):
  def default(self, obj):
      if isinstance(obj, np.integer):
          return int(obj)
      elif isinstance(obj, np.floating):
          return float(obj)
      elif isinstance(obj, np.ndarray):
          return obj.tolist()
      else:
          return super(MyEncoder, self).default(obj)

input_mean = 0
input_std = 255
input_layer = "Placeholder"
output_layer = "final_result"

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def run_model(model_file, file_name, input_height, input_width):
  graph = load_graph(model_file)
  t = read_tensor_from_image_file(
    file_name,
    input_height=input_height,
    input_width=input_width,
    input_mean=input_mean,
    input_std=input_std)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)

  with tf.Session(graph=graph) as sess:
    results = sess.run(output_operation.outputs[0], {
      input_operation.outputs[0]: t
    })
  results = np.squeeze(results)

  top_k = results.argsort()[-5:][::-1]
  return (top_k, results)


def damaged_or_whole(filepath, socketio):
  file_name = filepath
  model_file = "dow_output_graph.pb"
  label_file = "dow_output_labels.txt"

  top_k, results = run_model(model_file, file_name, 299, 299)
  labels = load_labels(label_file)

  final_output = []
  dow_output = []
  str_result = ""
  first = True
  for i in top_k:
    if not first:
      str_result += ", "

    if first:
      dow_top_label = labels[i]
      first = False
    in_prct = '%.2f' % (results[i] * 100) + "% "

    str_result += labels[i] + " : " + in_prct
    dow_output.append({labels[i]: results[i]})

  final_tpl = {"text": "Auto Status", "label": dow_top_label, "output": dow_output, "details": str_result}
  final_output.append(final_tpl)
  msg = json.dumps(final_tpl, cls=MyEncoder)
  socketio.emit('dow_finished', msg)

  if dow_top_label == "damage":
    model_file = "wp_output_graph.pb"
    label_file = "wp_output_labels.txt"

    top_k, results = run_model(model_file, file_name, 331, 331)
    labels = load_labels(label_file)

    wp_output = []
    first = True
    str_result = ""
    for i in top_k:
      if not first:
        str_result += ", "
      if first:
        wp_top_label = labels[i]
        first = False
      in_prct = '%.2f' % (results[i] * 100) + "% "

      str_result += labels[i] + " : " + in_prct
      wp_output.append((labels[i], results[i]))

    final_tpl = {"text": "Wo ist das Schaden", "label": wp_top_label, "output": wp_output, "details": str_result}
    final_output.append(final_tpl)
    msg = json.dumps(final_tpl, cls=MyEncoder)
    socketio.emit('wp_finished', msg)

    model_file = "s_output_graph.pb"
    label_file = "s_output_labels.txt"

    top_k, results = run_model(model_file, file_name, 331, 331)
    labels = load_labels(label_file)

    s_output = []
    str_result = ""
    first = True
    for i in top_k:
      if not first:
        str_result += ", "
      if first:
        s_top_label = labels[i]
        first = False
      in_prct = '%.2f' % (results[i] * 100) + "% "

      str_result += labels[i] + " : " + in_prct
      s_output.append((labels[i], results[i]))


    final_tpl = {"text": "Wie gross ist das Schaden", "label": s_top_label, "output": s_output, "details": str_result}
    final_output.append(final_tpl)
    msg = json.dumps(final_tpl, cls=MyEncoder)
    socketio.emit('s_finished', msg)

  return final_output