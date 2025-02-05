import pandas as pd
import numpy as np
import copy
from collections import namedtuple
import ast
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

PRIMITIVES = [
    'conv',            # tf.keras.layers.Conv2D
    'dil_conv_d2',     # tf.keras.layers.Conv2D dilation_rate = 2
    'dil_conv_d3',     # tf.keras.layers.Conv2D dilation_rate = 3
    'dil_conv_d4',     # tf.keras.layers.Conv2D dilation_rate = 4
    'Dsep_conv',       # DepthwiseSeparableConv2D
    'invert_Bot_Conv_E2',  # InvertedBottleneckBlock
    'conv_transpose',  # tf.keras.layers.Conv2DTranspose
    'identity'         # tf.keras.layers.Identity
]

CHANNELS = [16, 32, 48, 64, 16, 32, 48, 64]
REPEAT = [1, 2, 3, 4, 1, 2, 3, 4]
K = [1, 3, 5, 7, 1, 3, 5, 7]

Genotype = namedtuple('Genotype', 'Branch1 Branch2 Branch3')

def gray_to_int(gray_code):
    gray_bits = [int(bit) for bit in gray_code]
    binary_bits = [gray_bits[0]]
    for i in range(1, len(gray_bits)):
        next_binary_bit = gray_bits[i] ^ binary_bits[i-1]
        binary_bits.append(next_binary_bit)
    binary_str = ''.join(str(bit) for bit in binary_bits)
    return int(binary_str, 2)


def bstr_to_rstr(bstring):
    rstr = []
    for i in range(0, len(bstring), 3):
        r = gray_to_int(bstring[i:i+3])
        rstr.append(r)
    return rstr

def convert_cell(cell_bit_string):
    tmp = [cell_bit_string[i:i + 3] for i in range(0, len(cell_bit_string), 3)]
    return [tmp[i:i + 3] for i in range(0, len(tmp), 3)]

def convert(bit_string):
    b1 = convert_cell(bit_string[:len(bit_string)//3])
    b2 = convert_cell(bit_string[len(bit_string)//3:(len(bit_string)//3)*2])
    b3 = convert_cell(bit_string[(len(bit_string)//3)*2:])
    return [b1, b2, b3]


def decode(genome):
    genotype = copy.deepcopy(genome)
    channels = genome.pop(0)
    genotype = convert(genome)
    # decodes genome to architecture
    b1 = genotype[0]
    b2 = genotype[1]
    b3 = genotype[2]

    branch1, branch1_concat = [('channels', CHANNELS[channels])], list(range(2, len(b1)+2))
    branch2, branch2_concat = [('channels', CHANNELS[channels])], list(range(2, len(b2)+2))
    branch3, branch3_concat = [('channels', CHANNELS[channels])], list(range(2, len(b3)+2))

    for block in b1:
        for unit in block:
            branch1.append((PRIMITIVES[unit[0]], [K[unit[1]],K[unit[1]]], REPEAT[unit[2]]))

    for block in b2:
        for unit in block:
            branch2.append((PRIMITIVES[unit[0]], [K[unit[1]],K[unit[1]]], REPEAT[unit[2]]))
    for block in b3:
        for unit in block:
            branch3.append((PRIMITIVES[unit[0]], [K[unit[1]],K[unit[1]]], REPEAT[unit[2]]))

    #print(Genotype(Branch1=branch1,Branch2=branch2,Branch3=branch3))
    return Genotype(
        Branch1=branch1,
        Branch2=branch2,
        Branch3=branch3
    )


def get_branches(genotype):
  gens = copy.deepcopy(genotype)
  conv_args = {
      "activation": "relu",
      "padding": "same",
      }
  channels = []
  for element in gens:
    channels.append(element.pop(0))
  branches = [[],[],[]]

  for i in range(len(gens)):

    for layer in gens[i]:

      if layer[0] == 'conv':
        for l in range(layer[2]):
          branches[i].extend([layers.Conv2D(channels[i][1], layer[1], **conv_args)])

      elif layer[0] == 'dil_conv_d2':
        for l in range(layer[2]):
          branches[i].extend([layers.Conv2D(channels[i][1], layer[1], dilation_rate=2,**conv_args)])

      elif layer[0] == 'dil_conv_d3':
        for l in range(layer[2]):
          branches[i].extend([layers.Conv2D(channels[i][1], layer[1], dilation_rate=3,**conv_args)])

      elif layer[0] == 'dil_conv_d4':
        for l in range(layer[2]):
          branches[i].extend([layers.Conv2D(channels[i][1], layer[1], dilation_rate=4,**conv_args)])

      elif layer[0] == 'Dsep_conv':
        for l in range(layer[2]):
          branches[i].extend([layers.DepthwiseConv2D(layer[1], **conv_args), layers.Conv2D(channels[i][1], 1, **conv_args)])

      elif layer[0] == 'invert_Bot_Conv_E2':
        expand = int(channels[i][1])*2
        for l in range(layer[2]):
          branches[i].extend([layers.Conv2D(expand, 1,**conv_args), layers.DepthwiseConv2D(layer[1], **conv_args),layers.Conv2D(channels[i][1], layer[1],**conv_args)])

      elif layer[0] == 'conv_transpose':
        for l in range(layer[2]):
          branches[i].extend([layers.Conv2DTranspose(channels[i][1], layer[1], **conv_args)])

      elif layer[0] == 'identity':
          branches[i].extend([layers.Identity()])

      else:
        print("what?", i, layer[0])
      #  branches[i].extend(block)
  bc = []
  bc.extend(branches)
  bc.append(channels[0][1])
  return bc

class DepthToSpaceLayer(layers.Layer):
    def __init__(self, upscale_factor, **kwargs):
        super(DepthToSpaceLayer, self).__init__(**kwargs)
        self.upscale_factor = upscale_factor

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.upscale_factor)


def get_model(genotype, upscale_factor=2, channels = 3):

  branch1, branch2, branch3, channels_mod = get_branches(genotype)


  conv_args = {
        "activation": "relu",
        "padding": "same",
    }

  inputs = layers.Input(shape=(96, 96, channels), dtype='float32')


  inp = layers.Conv2D(channels_mod, 3, **conv_args)(inputs)

  b1 = branch1[0](inp)

  for l in range(1,len(branch1)):
    b1 = branch1[l](b1)

  b2 = branch2[0](inp)

  for l in range(1,len(branch2)):
    b2 = branch2[l](b2)

  b3 = branch3[0](inp)

  for l in range(1,len(branch3)):
    b3 = branch3[l](b3)

  x = layers.Add()([b1,b2,b3])
  #x = layers.Conv2D(12, 3, **conv_args)(b3)
  x = layers.Conv2D(12, 3, **conv_args)(x)
  x = DepthToSpaceLayer(upscale_factor)(x)
  outputs = layers.Conv2D(3, 3, **conv_args, dtype='float32')(x)
  
  return keras.Model(inputs, outputs)

def calculate_model_flops(model):
    # Ensure the model is built with a defined input shape
    if not model.built:
        sample_input = tf.keras.Input(shape=model.input_shape[1:])
        model(sample_input)

    # Define the forward pass for the model
    forward_pass = tf.function(model.call, input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])

    # TensorFlow Profiler gets FLOPs
    graph_info = profile(forward_pass.get_concrete_function().graph, options=ProfileOptionBuilder.float_operation())

    # Divide by 2 as `profile` counts multiply and accumulate as two FLOPs
    flops = graph_info.total_float_ops // 2
    return flops



def calculate_FLOPs_and_Params(data):
    FLOPs = np.empty(len(data[1]))
    params = np.empty(len(data[1]))
    # Decode each row into phenotype (architecture layers)
    for index, row in data.iterrows():
        bit_string = row[0]
        bit_list = ast.literal_eval(bit_string)         
        decoded_genotype = decode(bit_list)
        print(f"Architecture for row {index}:")
        print(decoded_genotype)
        model = get_model(decoded_genotype)
        FLOPs[index] = calculate_model_flops(model)
        params[index] = model.count_params()

    return [FLOPs, params]
        