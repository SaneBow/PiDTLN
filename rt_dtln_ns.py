#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to process realtime audio with a trained DTLN model. 
This script supports ALSA audio devices. The model expects 16kHz single channel audio input/output.

Example call:
    $python rt_dtln_ns.py -i capture -o playback

Author: sanebow (sanebow@gmail.com)
Version: 23.05.2021

This code is licensed under the terms of the MIT-license.
"""


import numpy as np
import sounddevice as sd
import tflite_runtime.interpreter as tflite
import argparse
import collections
import time
import daemon
import threading


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    '-i', '--input-device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-o', '--output-device', type=int_or_str,
    help='output device (numeric ID or substring)')
parser.add_argument(
    '-c', '--channel', type=int, default=None,
    help='use specific channel of input device')
parser.add_argument(
    '-n', '--no-denoise', action='store_true',
    help='turn off denoise, pass-through')
parser.add_argument(
    '-t', '--threads', type=int, default=1,
    help='number of threads for tflite interpreters')
parser.add_argument(
    '--latency', type=float, default=0.2,
    help='suggested input/output latency in seconds')
parser.add_argument(
    '-D', '--daemonize', action='store_true',
    help='run as a daemon')
parser.add_argument(
    '--measure', action='store_true',
    help='measure and report processing time')

args = parser.parse_args(remaining)

# set some parameters
block_len_ms = 32 
block_shift_ms = 8
fs_target = 16000
# create the interpreters
interpreter_1 = tflite.Interpreter(model_path='./models/model_quant_1.tflite', num_threads=args.threads)
interpreter_1.allocate_tensors()
interpreter_2 = tflite.Interpreter(model_path='./models/model_quant_2.tflite', num_threads=args.threads)
interpreter_2.allocate_tensors()
# Get input and output tensors.
input_details_1 = interpreter_1.get_input_details()
output_details_1 = interpreter_1.get_output_details()
input_details_2 = interpreter_2.get_input_details()
output_details_2 = interpreter_2.get_output_details()
# create states for the lstms
states_1 = np.zeros(input_details_1[1]['shape']).astype('float32')
states_2 = np.zeros(input_details_2[1]['shape']).astype('float32')
# calculate shift and length
block_shift = int(np.round(fs_target * (block_shift_ms / 1000)))
block_len = int(np.round(fs_target * (block_len_ms / 1000)))
# create buffer
in_buffer = np.zeros((block_len)).astype('float32')
out_buffer = np.zeros((block_len)).astype('float32')

t_ring = collections.deque(maxlen=100)

def callback(indata, outdata, frames, buf_time, status):
    # buffer and states to global
    global in_buffer, out_buffer, states_1, states_2, t_ring
    if args.measure:
        start_time = time.time()
    if status:
        print(status)
    if args.channel is not None:
        indata = indata[:, [args.channel]] 
    if args.no_denoise:
        outdata[:] = indata
        if args.measure:
            t_ring.append(time.time() - start_time)
        return
    # write to buffer
    in_buffer[:-block_shift] = in_buffer[block_shift:]
    in_buffer[-block_shift:] = np.squeeze(indata)
    # calculate fft of input block
    in_block_fft = np.fft.rfft(in_buffer)
    in_mag = np.abs(in_block_fft)
    in_phase = np.angle(in_block_fft)
    # reshape magnitude to input dimensions
    in_mag = np.reshape(in_mag, (1,1,-1)).astype('float32')
    # set tensors to the first model
    interpreter_1.set_tensor(input_details_1[1]['index'], states_1)
    interpreter_1.set_tensor(input_details_1[0]['index'], in_mag)
    # run calculation 
    interpreter_1.invoke()
    # get the output of the first block
    out_mask = interpreter_1.get_tensor(output_details_1[0]['index']) 
    states_1 = interpreter_1.get_tensor(output_details_1[1]['index'])   
    # calculate the ifft
    estimated_complex = in_mag * out_mask * np.exp(1j * in_phase)
    estimated_block = np.fft.irfft(estimated_complex)
    # reshape the time domain block
    estimated_block = np.reshape(estimated_block, (1,1,-1)).astype('float32')
    # set tensors to the second block
    interpreter_2.set_tensor(input_details_2[1]['index'], states_2)
    interpreter_2.set_tensor(input_details_2[0]['index'], estimated_block)
    # run calculation
    interpreter_2.invoke()
    # get output tensors
    out_block = interpreter_2.get_tensor(output_details_2[0]['index']) 
    states_2 = interpreter_2.get_tensor(output_details_2[1]['index']) 
    # write to buffer
    out_buffer[:-block_shift] = out_buffer[block_shift:]
    out_buffer[-block_shift:] = np.zeros((block_shift))
    out_buffer  += np.squeeze(out_block)
    # output to soundcard
    outdata[:] = np.expand_dims(out_buffer[:block_shift], axis=-1)
    if args.measure:
        t_ring.append(time.time() - start_time)
    

def open_stream():
    with sd.Stream(device=(args.input_device, args.output_device),
                samplerate=fs_target, blocksize=block_shift,
                dtype=np.float32, latency=args.latency,
                channels=(1 if args.channel is None else None, 1), callback=callback):
        print('#' * 80)
        print('Ctrl-C to exit')
        print('#' * 80)
        if args.measure:
            while True:
                time.sleep(1)
                print('Processing time: {:.2f} ms'.format( 1000 * np.average(t_ring) ), end='\r')
        else:
            threading.Event().wait()


try:
    if args.daemonize:
        with daemon.DaemonContext():
            open_stream()
    else:
        open_stream()
except KeyboardInterrupt:
    parser.exit('')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
    
