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

import sounddevice as sd
import numpy as np
import time
import argparse
import tflite_runtime.interpreter as tflite
from multiprocessing import Process, Queue
import threading
import collections
import daemon
import sys
try:
    from multiprocessing import shared_memory
except ImportError:
    try:
        import shared_memory
    except ImportError:
        print("[ERROR] please install shared-memory38 on Python < 3.8")
        exit(1)

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

def fetch_shm_ndarray(name, shape, dtype='float32', init=False):
    buf = np.zeros(shape).astype(dtype)
    if init:
        shm = shared_memory.SharedMemory(name=name, create=True, size=buf.nbytes)
    else:
        shm = shared_memory.SharedMemory(name=name)
    arr = np.ndarray(buf.shape, dtype=buf.dtype, buffer=shm.buf)
    if init:
        arr[:] = buf[:]
    return arr, shm

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
parser.add_argument('--no-fftw', action='store_true', help='use np.fft instead of fftw')
parser.add_argument('-D', '--daemonize', action='store_true',help='run as a daemon')
args = parser.parse_args(remaining)

# set block len and block shift
block_len = 512
block_shift = 128

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

in_buffer = np.zeros((block_len)).astype("float32")
in_buffer_lpb = np.zeros((block_len)).astype("float32")
out_buffer = np.zeros((block_len)).astype('float32')

in_lpb, _shm1 = fetch_shm_ndarray('in_lpb', (1, 1, block_len), init=True)
est_block, _shm2 = fetch_shm_ndarray('est_block', (1, 1, block_len), init=True)
out_block, _shm3 = fetch_shm_ndarray('out_block', (block_len), init=True)
shms = [_shm1, _shm2, _shm3]

q1 = Queue(maxsize=1)
q2 = Queue(maxsize=1)

if g_use_fftw:
    fft_buf = pyfftw.empty_aligned(512, dtype='float32')
    rfft = pyfftw.builders.rfft(fft_buf)
    ifft_buf = pyfftw.empty_aligned(257, dtype='complex64')
    irfft = pyfftw.builders.irfft(ifft_buf)


t_ring = collections.deque(maxlen=512)

# =========== stage 1 =============
def callback(indata, outdata, frames, buftime, status):
    global in_buffer, out_buffer, in_buffer_lpb, states_1, g_use_fftw
    if args.measure:
        start_time = time.time()

    if status:
        print(status)

    if args.no_aec:
        # time.sleep(max(0, np.random.normal(loc=6.0, scale=1.0)*1e-3))
        outdata[:, 0] = indata[:, 0]
        if args.measure:
            t_ring.append(time.time() - start_time)
        return

    q1.put(1)   # wait stage2 read in_buffer

    # write mic stream to buffer
    in_buffer[:-block_shift] = in_buffer[block_shift:]
    in_buffer[-block_shift:] = np.squeeze(indata[:, 0])
    # write playback stream to buffer
    in_buffer_lpb[:-block_shift] = in_buffer_lpb[block_shift:]
    in_buffer_lpb[-block_shift:] = np.squeeze(indata[:, -1])

    # calculate fft of input block
    if g_use_fftw:
        fft_buf[:] = in_buffer
        in_block_fft = rfft().astype("complex64") 
    else:
        in_block_fft = np.fft.rfft(in_buffer).astype("complex64")    
    # create magnitude
    in_mag = np.abs(in_block_fft)
    in_mag = np.reshape(in_mag, (1, 1, -1)).astype("float32")

    # calculate log pow of lpb
    if g_use_fftw:
        fft_buf[:] = in_buffer_lpb
        lpb_block_fft = rfft().astype("complex64")
    else:
        lpb_block_fft = np.fft.rfft(in_buffer_lpb).astype("complex64")
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
    if g_use_fftw:
        ifft_buf[:] = in_block_fft * out_mask
        estimated_block = irfft()
    else:
        estimated_block = np.fft.irfft(in_block_fft * out_mask)

    # reshape the time domain frames
    in_lpb[:] = np.reshape(in_buffer_lpb, (1, 1, -1)).astype("float32")
    est_block[:] = np.reshape(estimated_block, (1, 1, -1)).astype("float32")

    q2.get()    # stage2 can continue

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
        if dt > 8e-3:
            print("[warning] process time: {:.2f} ms".format(dt * 1000))


def stage2(interpreter_2, q1, q2):
    input_details_2 = interpreter_2.get_input_details()
    est_idx = next(i for i in input_details_2 if i["name"] == "input_6")["index"]
    lpb_idx = next(i for i in input_details_2 if i["name"] == "input_7")["index"]
    states_idx = next(i for i in input_details_2 if i["name"] == "input_8")["index"]
    output_details_2 = interpreter_2.get_output_details()
    states_2 = np.zeros(input_details_2[states_idx]["shape"]).astype("float32")

    block_len = 512
    in_lpb, _shm1 = fetch_shm_ndarray('in_lpb', (1, 1, block_len))
    est_block, _shm2 = fetch_shm_ndarray('est_block', (1, 1, block_len))
    out_block, _shm3 = fetch_shm_ndarray('out_block', (block_len))

    q2.put(0)   # ready, block for q1 to run first

    while True:
        q2.put(2)   # wait stage1 calculate est_block

        # set tensors to the second block
        interpreter_2.set_tensor(input_details_2[lpb_idx]["index"], in_lpb)
        interpreter_2.set_tensor(input_details_2[est_idx]["index"], est_block)
        interpreter_2.set_tensor(input_details_2[states_idx]["index"], states_2)

        q1.get()    # stage1 can continue

        # run calculation
        interpreter_2.invoke()
        # get output tensors
        out_block[:] = interpreter_2.get_tensor(output_details_2[0]["index"])
        states_2 = interpreter_2.get_tensor(output_details_2[1]["index"])


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
                print('Processing time: {:.2f} ms, std={:.2f}'.format( 
                    1000 * np.average(t_ring), 1000 * np.std(t_ring) 
                ), end='\r')
        else:
            threading.Event().wait()


try:
    p2 = Process(target=stage2, args=(interpreter_2, q1, q2))
    p2.start()
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
    for shm in shms:
        shm.close()
        shm.unlink()
    p2.terminate()
    p2.join()