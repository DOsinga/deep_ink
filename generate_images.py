#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import PIL.Image
import argparse
import tensorflow as tf
import os
import sys
import shutil

BLACK = -2
WHITE = 2
IMAGE_PREFIX = 'img_'

def in_circle(x, y):
  dx = x - 112
  dy = y - 112
  r2 = dx * dx + dy * dy
  return r2 < 30


def save_image(img, fn):
    a = np.uint8(np.clip(img, 0, 1) * 255)
    PIL.Image.fromarray(a).save(open(fn, 'wb'))


def T(graph, layer):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("import/%s:0"%layer)


def update_image(g, img1):
  g1 = np.mean(g, 2)
  todo = sorted(((g1[x][y], x, y) for x in range(224) for y in range(224)), key=lambda t: abs(t[0]))
  flips = 0
  while flips < 50 and todo:
    score, x, y = todo.pop()
    if score > 0 and img1[x][y][0] == BLACK:
      img1[x][y] = [WHITE, WHITE, WHITE]
      flips += 1
    elif score < 0 and img1[x][y][0] == WHITE:
      img1[x][y] = [BLACK, BLACK, BLACK]
      flips += 1
  return flips


def parse_channel(channel_def):
  for chunk in channel_def.split(','):
    chunk = chunk.strip()
    if '-' in chunk:
      begin, end = (int(x) for x in chunk.split('-'))
      for i in range(begin, end + 1):
        yield i
    else:
      yield int(chunk)

def create_drawing(model_fn, layer, channel, drawings):
  graph = tf.Graph()
  sess = tf.InteractiveSession(graph=graph)
  with tf.gfile.FastGFile(model_fn, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
  t_input = tf.placeholder(np.float32, name='input') # define the input tensor
  imagenet_mean = 117.0
  t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
  tf.import_graph_def(graph_def, {'input': t_preprocessed})

  t_obj = T(graph, layer)[:,channel]
  t_score = tf.reduce_mean(t_obj)
  t_grad = tf.gradients(t_score, t_input)[0]

  checkpoint_path = os.path.join(drawings, 'checkpoints')
  if os.path.isdir(checkpoint_path):
    shutil.rmtree(checkpoint_path)
  os.mkdir(checkpoint_path)

  img_path = os.path.join(drawings, '%s-%d.png' % (layer, channel))

  image = np.array([[([BLACK, BLACK, BLACK] if in_circle(x, y) else [WHITE, WHITE, WHITE]) for x in range(224)] for y in range(224)])

  score = 0
  for index in range(0, 1001):
    if index > 0:
      g, score = sess.run([t_grad, t_score], {t_input:image})
      update_image(g, image)
    if index % 25 == 0:
      print(index, score)
      save_image(image, img_path)
      shutil.copy(img_path, os.path.join(checkpoint_path, '%04d.png' % index))


def path_for_index(index, path):
  return os.path.join(path, '%s%d.png' % (IMAGE_PREFIX, index))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Doodle images')
  parser.add_argument('--model_path', type=str, default='inception5h/tensorflow_inception_graph.pb',
                      help='Where the unpacked model dump is')
  parser.add_argument('--layer', type=str, default='output2',
                      help='Which layer to pick the feature from')
  parser.add_argument('--channel', type=str, default='139',
                      help='Which channel contains the feature')
  parser.add_argument('--drawings', type=str, default='drawings',
                      help='Where to store the drawings')
  args = parser.parse_args()

  if not os.path.isfile(args.model_path):
    print('%s is not a file. You can download inception from:' % args.model_path)
    print('https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip')
    sys.exit(1)

  for channel in parse_channel(args.channel):
    create_drawing(args.model_path, args.layer, channel, args.drawings)





