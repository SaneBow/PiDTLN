# -*- coding: utf-8 -*-
"""
Script to process realtime audio with a trained DTLN-aec model. 
This script directly interacts with audio devices. It expects 16kHz audio input/output. 
Input device should contain a loopback channel as its last channel, and it assume raw mic input is in the first channel.

Example call:
    $python rt_dtln_aec_mp.py -i capture  -o playback -m /name/of/the/model

Author: sanebow (sanebow@gmail.com)
Version: 23.05.2021

This code is licensed under the terms of the MIT-license.
"""

import soundfile as sf
import sounddevice as sd
import numpy as np
import os
import time
import argparse
import tflite_runtime.interpreter as tflite
from multiprocessing import Process, Queue
import threading
import collections
import daemon


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

# set block len and block shift
block_len = 512
block_shift = 128

in_buffer = np.zeros((block_len)).astype("float32")
in_buffer_lpb = np.zeros((block_len)).astype("float32")

out_buffer = np.zeros((block_len)).astype('float32')# create buffer

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument("--input_device", "-i", type=int_or_str, help="input device (numeric ID or substring)")
parser.add_argument("--output_device", "-o", type=int_or_str, help="output device (numeric ID or substring)")
parser.add_argument("--model", "-m", choices=['128', '256', '512'], default='256', help="number of LSTM units for tf-lite model (one of 128, 256, 512)")
parser.add_argument("--channels", "-c", type=int, default=2, help="number of input channels")
parser.add_argument('--no-aec', '-n',  action='store_true', help='turn off AEC, pass-through')
parser.add_argument("--latency", type=float, default=0.2, help="latency of sound device")
parser.add_argument("--threads", type=int, default=1, help="set thread number for interpreters")
parser.add_argument('--measure', action='store_true', help='measure and report processing time')
parser.add_argument('-D', '--daemonize', action='store_true',help='run as a daemon')
args = parser.parse_args(remaining)

interpreter_1 = tflite.Interpreter(
    model_path="models/dtln_aec_{}_quant_1.tflite".format(args.model), num_threads=args.threads)
interpreter_1.allocate_tensors()
input_details_1 = interpreter_1.get_input_details()
in_idx = next(i for i in input_details_1 if i["name"] == "input_3")["index"]
lpb_idx = next(i for i in input_details_1 if i["name"] == "input_4")["index"]
states_idx = next(i for i in input_details_1 if i["name"] == "input_5")["index"]
output_details_1 = interpreter_1.get_output_details()
states_1 = np.zeros(input_details_1[states_idx]["shape"]).astype("float32")

interpreter_2 = tflite.Interpreter(
    model_path="models/dtln_aec_{}_quant_2.tflite".format(args.model), num_threads=args.threads)
interpreter_2.allocate_tensors()

q1 = Queue(maxsize=1)
q2 = Queue(maxsize=1)

t_ring = collections.deque(maxlen=100)

def callback(indata, outdata, frames, buftime, status):
    global in_buffer, out_buffer, in_buffer_lpb, states_1
    if args.measure:
        start_time = time.time()

    if status:
        print(status)

    if args.no_aec:
        outdata[:, 0] = indata[:, 0]
        if args.measure:
            t_ring.append(time.time() - start_time)
        return

    # write mic stream to buffer
    in_buffer[:-block_shift] = in_buffer[block_shift:]
    in_buffer[-block_shift:] = np.squeeze(indata[:, 0])
    # write playback stream to buffer
    in_buffer_lpb[:-block_shift] = in_buffer_lpb[block_shift:]
    in_buffer_lpb[-block_shift:] = np.squeeze(indata[:, -1])

    ## stage 1
    # calculate fft of input block
    in_block_fft = np.fft.rfft(np.squeeze(in_buffer)).astype("complex64")
    # create magnitude
    in_mag = np.abs(in_block_fft)
    in_mag = np.reshape(in_mag, (1, 1, -1)).astype("float32")
    # calculate log pow of lpb
    lpb_block_fft = np.fft.rfft(np.squeeze(in_buffer_lpb)).astype("complex64")
    lpb_mag = np.abs(lpb_block_fft)
    lpb_mag = np.reshape(lpb_mag, (1, 1, -1)).astype("float32")
    # set tensors to the first model
    interpreter_1.set_tensor(input_details_1[in_idx]["index"], in_mag)
    interpreter_1.set_tensor(input_details_1[lpb_idx]["index"], lpb_mag)
    interpreter_1.set_tensor(input_details_1[states_idx]["index"], states_1)
    # run calculation
    interpreter_1.invoke()
    # # get the output of the first block
    out_mask = interpreter_1.get_tensor(output_details_1[0]["index"])
    states_1 = interpreter_1.get_tensor(output_details_1[1]["index"])

    # apply mask and calculate the ifft
    estimated_block = np.fft.irfft(in_block_fft * out_mask)
    # reshape the time domain frames
    estimated_block = np.reshape(estimated_block, (1, 1, -1)).astype("float32")
    in_lpb = np.reshape(in_buffer_lpb, (1, 1, -1)).astype("float32")

    q1.put((estimated_block, in_lpb))

    out_block = q2.get()
    if out_block is None:
        return
    # shift values and write to buffer
    # write to buffer
    out_buffer[:-block_shift] = out_buffer[block_shift:]
    out_buffer[-block_shift:] = np.zeros((block_shift))
    out_buffer  += np.squeeze(out_block)
    # output to soundcard
    outdata[:] = np.expand_dims(out_buffer[:block_shift], axis=-1)

    if args.measure:
        dt = time.time() - start_time
        t_ring.append(dt)
        if dt > 7e-3:
            print("[warning] process time: {:.2f} ms".format(dt * 1000))

def stage2(interpreter_2, _qi, _qo):
    input_details_2 = interpreter_2.get_input_details()
    est_idx = next(i for i in input_details_2 if i["name"] == "input_6")["index"]
    lpb_idx = next(i for i in input_details_2 if i["name"] == "input_7")["index"]
    states_idx = next(i for i in input_details_2 if i["name"] == "input_8")["index"]
    output_details_2 = interpreter_2.get_output_details()
    states_2 = np.zeros(input_details_2[states_idx]["shape"]).astype("float32")
    while True:
        estimated_block, in_lpb = _qi.get()
        if estimated_block is None:
            _qo.put(None)
            return

        # set tensors to the second block
        interpreter_2.set_tensor(input_details_2[states_idx]["index"], states_2)
        interpreter_2.set_tensor(input_details_2[est_idx]["index"], estimated_block)
        interpreter_2.set_tensor(input_details_2[lpb_idx]["index"], in_lpb)
        # run calculation
        interpreter_2.invoke()
        # get output tensors
        out_block = interpreter_2.get_tensor(output_details_2[0]["index"])
        states_2 = interpreter_2.get_tensor(output_details_2[1]["index"])

        _qo.put(out_block)


p2 = Process(target=stage2, args=(interpreter_2, q1, q2))
q2.put(None)    # bootstrap stage 1
p2.start()

def open_stream():
    with sd.Stream(device=(args.input_device, args.output_device),
                samplerate=16000, blocksize=block_shift,
                dtype=np.float32, latency=args.latency,
                channels=(args.channels, 1), callback=callback):
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
    print("Keyboard interrupt, terminating ...")
except Exception as e:
    raise
finally:
    p2.terminate()
    p2.join()