# -*- coding: utf-8 -*-
"""
Script to process realtime audio with a trained DTLN-aec model (multiprocessing version). 
This script directly interacts with audio devices. It expects 16kHz audio input/output. 
Input device should contain a loopback channel as its last channel, and it assume raw mic input is in the first channel.

Example call:
    $python rt_dtln_aec.py -i capture  -o playback -m /name/of/the/model

Author: sanebow (sanebow@gmail.com)
Version: 23.05.2021

This code is licensed under the terms of the MIT-license.
"""

import sounddevice as sd
import numpy as np
import os
import time
import threading
import argparse
import tflite_runtime.interpreter as tflite
import collections
import daemon

g_use_fftw = True
try:
    import pyfftw
except ImportError:
    print("[WARNING] pyfftw is not installed, use np.fft")
    g_use_fftw = False


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
parser.add_argument( '-l', '--list-devices', action='store_true',
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
parser.add_argument("--model", "-m", choices=['128', '256', '512'], default='128', help="number of LSTM units for tf-lite model (one of 128, 256, 512)")
parser.add_argument("--channels", "-c", type=int, default=2, help="number of input channels")
parser.add_argument('--no-aec', '-n',  action='store_true', help='turn off AEC, pass-through')
parser.add_argument("--latency", type=float, default=0.2, help="latency of sound device")
parser.add_argument("--threads", type=int, default=1, help="set thread number for interpreters")
parser.add_argument('--measure', action='store_true', help='measure and report processing time')
parser.add_argument('--no-fftw', action='store_true', help='use np.fft instead of fftw')
parser.add_argument('-D', '--daemonize', action='store_true',help='run as a daemon')
args = parser.parse_args(remaining)

interpreter_1 = tflite.Interpreter(
    model_path="models/dtln_aec_{}_quant_1.tflite".format(args.model), num_threads=args.threads)
interpreter_1.allocate_tensors()
interpreter_2 = tflite.Interpreter(
    model_path="models/dtln_aec_{}_quant_2.tflite".format(args.model), num_threads=args.threads)
interpreter_2.allocate_tensors()
input_details_1 = interpreter_1.get_input_details()
input_details_2 = interpreter_2.get_input_details()
in_idx = next(i for i in input_details_1 if i["name"] == "input_3")["index"]
lpb1_idx = next(i for i in input_details_1 if i["name"] == "input_4")["index"]
states1_idx = next(i for i in input_details_1 if i["name"] == "input_5")["index"]
est_idx = next(i for i in input_details_2 if i["name"] == "input_6")["index"]
lpb2_idx = next(i for i in input_details_2 if i["name"] == "input_7")["index"]
states2_idx = next(i for i in input_details_2 if i["name"] == "input_8")["index"]
output_details_1 = interpreter_1.get_output_details()
output_details_2 = interpreter_2.get_output_details()
states_1 = np.zeros(input_details_1[states1_idx]["shape"]).astype("float32")
states_2 = np.zeros(input_details_2[states2_idx]["shape"]).astype("float32")

if args.no_fftw:
    g_use_fftw = False
if g_use_fftw:
    fft_buf = pyfftw.empty_aligned(512, dtype='float32')
    rfft = pyfftw.builders.rfft(fft_buf)
    ifft_buf = pyfftw.empty_aligned(257, dtype='complex64')
    irfft = pyfftw.builders.irfft(ifft_buf)

t_ring = collections.deque(maxlen=100)

def callback(indata, outdata, frames, buftime, status):
    global in_buffer, out_buffer, in_buffer_lpb, states_1, states_2, t_ring , g_use_fftw
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

    # calculate fft of input block
    if g_use_fftw:
        fft_buf[:] = in_buffer
        in_block_fft = rfft()
    else:
        in_block_fft = np.fft.rfft(in_buffer).astype("complex64")

    # create magnitude
    in_mag = np.abs(in_block_fft)
    in_mag = np.reshape(in_mag, (1, 1, -1)).astype("float32")
    # calculate log pow of lpb
    if g_use_fftw:
        fft_buf[:] = in_buffer_lpb
        lpb_block_fft = rfft()
    else:
        lpb_block_fft = np.fft.rfft(in_buffer_lpb).astype("complex64")
    lpb_mag = np.abs(lpb_block_fft)
    lpb_mag = np.reshape(lpb_mag, (1, 1, -1)).astype("float32")
    # set tensors to the first model
    interpreter_1.set_tensor(input_details_1[in_idx]["index"], in_mag)
    interpreter_1.set_tensor(input_details_1[lpb1_idx]["index"], lpb_mag)
    interpreter_1.set_tensor(input_details_1[states1_idx]["index"], states_1)
    # run calculation
    interpreter_1.invoke()
    # # get the output of the first block
    out_mask = interpreter_1.get_tensor(output_details_1[0]["index"])
    states_1 = interpreter_1.get_tensor(output_details_1[1]["index"])

    # apply mask and calculate the ifft
    if g_use_fftw:
        ifft_buf[:] = in_block_fft * out_mask
        estimated_block = irfft()
    else:
        estimated_block = np.fft.irfft(in_block_fft * out_mask)
    # reshape the time domain frames
    estimated_block = np.reshape(estimated_block, (1, 1, -1)).astype("float32")
    in_lpb = np.reshape(in_buffer_lpb, (1, 1, -1)).astype("float32")

    # set tensors to the second block
    interpreter_2.set_tensor(input_details_2[lpb2_idx]["index"], in_lpb)
    interpreter_2.set_tensor(input_details_2[est_idx]["index"], estimated_block)
    interpreter_2.set_tensor(input_details_2[states2_idx]["index"], states_2)
    # run calculation
    interpreter_2.invoke()
    # get output tensors
    out_block = interpreter_2.get_tensor(output_details_2[0]["index"])
    states_2 = interpreter_2.get_tensor(output_details_2[1]["index"])

    # shift values and write to buffer
    # write to buffer
    out_buffer[:-block_shift] = out_buffer[block_shift:]
    out_buffer[-block_shift:] = np.zeros((block_shift))
    out_buffer += np.squeeze(out_block)
    # output to soundcard
    outdata[:] = np.expand_dims(out_buffer[:block_shift], axis=-1)

    if args.measure:
        dt = time.time() - start_time
        t_ring.append(dt)
        if dt > 7e-3:
            print("[!] process time: {:.2f} ms".format(dt * 1000))


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
    parser.exit('')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))