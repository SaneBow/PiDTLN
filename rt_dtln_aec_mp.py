# -*- coding: utf-8 -*-
"""
Script to process a folder of .wav files with a trained DTLN-aec model. 
This script supports subfolders and names the processed files the same as the 
original. The model expects 16kHz single channel audio .wav files.
The idea of this script is to use it for baseline or comparison purpose.

Example call:
    $python run_aec.py -i /name/of/input/folder  \
                              -o /name/of/output/folder \
                              -m /name/of/the/model

Author: Nils L. Westhausen (nils.westhausen@uol.de)
Version: 27.10.2020

This code is licensed under the terms of the MIT-license.
"""

import soundfile as sf
import sounddevice as sd
import numpy as np
import os
import time as otime
import argparse
# import tensorflow.lite as tflite
import tflite_runtime.interpreter as tflite
import multiprocessing
import signal

# make GPUs invisible
os.environ["CUDA_VISIBLE_DEVICES"] = ""



# set block len and block shift
block_len = 512
block_shift = 128

in_buffer = np.zeros((block_len)).astype("float32")
in_buffer_lpb = np.zeros((block_len)).astype("float32")

out_buffer = np.zeros((block_len)).astype('float32')# create buffer

# arguement parser for running directly from the command line
parser = argparse.ArgumentParser(description="data evaluation")
parser.add_argument("--input_device", "-i", help="input device (mic+playback)")
parser.add_argument("--output_device", "-o", help="output device (speaker)")
parser.add_argument("--model", "-m", help="name of tf-lite model")
parser.add_argument("--latency", "-l", type=float, default=0, help="latency of sound device")
parser.add_argument("--threads", "-t", type=int, default=1, help="set thread number for interpreters")
parser.add_argument("--channels", "-c", type=int, default=2, help="number of input channels")
parser.add_argument('-n', '--no-aec', action='store_true', help='turn off AEC, pass-through')
args = parser.parse_args()

interpreter_1 = tflite.Interpreter(model_path=args.model + "_1.tflite", num_threads=args.threads)
interpreter_1.allocate_tensors()
input_details_1 = interpreter_1.get_input_details()
in_idx = next(i for i in input_details_1 if i["name"] == "input_3")["index"]
lpb_idx = next(i for i in input_details_1 if i["name"] == "input_4")["index"]
states_idx = next(i for i in input_details_1 if i["name"] == "input_5")["index"]
output_details_1 = interpreter_1.get_output_details()
states_1 = np.zeros(input_details_1[states_idx]["shape"]).astype("float32")

q1 = multiprocessing.JoinableQueue(maxsize=1)
q2 = multiprocessing.Queue(maxsize=1)


def callback(indata, outdata, frames, time, status):
    global in_buffer, out_buffer, in_buffer_lpb, states_1
    if status:
        print(status)
    if args.no_aec:
        outdata[:, 0] = indata[:, 0]
        return
    # start_time = otime.time()

    # write mic stream to buffer
    in_buffer[:-block_shift] = in_buffer[block_shift:]
    in_buffer[-block_shift:] = np.squeeze(indata[:, 0])
    # write playback stream to buffer
    in_buffer_lpb[:-block_shift] = in_buffer_lpb[block_shift:]
    in_buffer_lpb[-block_shift:] = np.squeeze(indata[:, -1])

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

    # print((otime.time() - start_time) * 1000)

def stage2(model, _qi, _qo, threads):
    interpreter_2 = tflite.Interpreter(model_path=model + "_2.tflite", num_threads=threads)
    interpreter_2.allocate_tensors()
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
        # _qi.task_done()

p2 = multiprocessing.Process(target=stage2, args=(args.model, q1, q2, args.threads))
p2.daemon = True
q2.put(None)

try:
    p2.start()
    with sd.Stream(device=(args.input_device, args.output_device),
                samplerate=16000, blocksize=block_shift,
                dtype=np.float32, latency=args.latency,
                channels=(args.channels, 1), callback=callback):
        print('#' * 80)
        print('press Return to quit')
        print('#' * 80)
        input()
except KeyboardInterrupt:
    print("Keyboard interrupt, terminating ...")
except Exception as e:
    raise
finally:
    p2.terminate()
    p2.join()