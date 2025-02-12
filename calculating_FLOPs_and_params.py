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
    
    b1, b2, b3 = genotype[0], genotype[1], genotype[2]

    branch1, branch1_concat = [('channels', CHANNELS[channels])], list(range(2, len(b1)+2))
    branch2, branch2_concat = [('channels', CHANNELS[channels])], list(range(2, len(b2)+2))
    branch3, branch3_concat = [('channels', CHANNELS[channels])], list(range(2, len(b3)+2))

    for block in b1:
        for unit in block:
            branch1.append((PRIMITIVES[unit[0]], [K[unit[1]], K[unit[1]]], REPEAT[unit[2]]))

    for block in b2:
        for unit in block:
            branch2.append((PRIMITIVES[unit[0]], [K[unit[1]], K[unit[1]]], REPEAT[unit[2]]))
    
    for block in b3:
        for unit in block:
            branch3.append((PRIMITIVES[unit[0]], [K[unit[1]], K[unit[1]]], REPEAT[unit[2]]))

    return Genotype(Branch1=branch1, Branch2=branch2, Branch3=branch3)

def get_branches(genotype):
    gens = copy.deepcopy(genotype)
    conv_args = {
        "activation": "relu",
        "padding": "same",
    }
    channels = []
    for element in gens:
        channels.append(element.pop(0))
    
    branches = [[], [], []]

    for i in range(len(gens)):
        for layer in gens[i]:
            if layer[0] == 'conv':
                for _ in range(layer[2]):
                    branches[i].append(layers.Conv2D(channels[i][1], layer[1], **conv_args))
            elif layer[0] == 'dil_conv_d2':
                for _ in range(layer[2]):
                    branches[i].append(layers.Conv2D(channels[i][1], layer[1], dilation_rate=2, **conv_args))
            elif layer[0] == 'dil_conv_d3':
                for _ in range(layer[2]):
                    branches[i].append(layers.Conv2D(channels[i][1], layer[1], dilation_rate=3, **conv_args))
            elif layer[0] == 'dil_conv_d4':
                for _ in range(layer[2]):
                    branches[i].append(layers.Conv2D(channels[i][1], layer[1], dilation_rate=4, **conv_args))
            elif layer[0] == 'Dsep_conv':
                for _ in range(layer[2]):
                    branches[i].extend([layers.DepthwiseConv2D(layer[1], **conv_args), layers.Conv2D(channels[i][1], 1, **conv_args)])
            elif layer[0] == 'invert_Bot_Conv_E2':
                expand = int(channels[i][1]) * 2
                for _ in range(layer[2]):
                    branches[i].extend([layers.Conv2D(expand, 1, **conv_args), layers.DepthwiseConv2D(layer[1], **conv_args), layers.Conv2D(channels[i][1], layer[1], **conv_args)])
            elif layer[0] == 'conv_transpose':
                for _ in range(layer[2]):
                    branches[i].append(layers.Conv2DTranspose(channels[i][1], layer[1], **conv_args))
            elif layer[0] == 'identity':
                branches[i].append(layers.Identity())

    return branches + [channels[0][1]]

def count_model_layers(model):
    return len(model.layers)

def calculate_model_flops(model):
    if not model.built:
        sample_input = tf.keras.Input(shape=model.input_shape[1:])
        model(sample_input)

    forward_pass = tf.function(model.call, input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])

    graph_info = profile(forward_pass.get_concrete_function().graph, options=ProfileOptionBuilder.float_operation())

    flops = graph_info.total_float_ops // 2
    return flops

def calculate_FLOPs_and_Params(data):
    FLOPs = np.empty(len(data))
    params = np.empty(len(data))
    N_layers = np.empty(len(data))

    for index, row in data.iterrows():
        bit_string = row[0]
        bit_list = ast.literal_eval(bit_string)         
        decoded_genotype = decode(bit_list)
        print(f"Architecture for row {index}:")
        print(decoded_genotype)
        
        model = get_model(decoded_genotype)
        FLOPs[index] = calculate_model_flops(model)
        params[index] = model.count_params()
        N_layers[index] = count_model_layers(model)

    return [FLOPs, params, N_layers]
