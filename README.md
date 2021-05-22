# PiDTLN
Apply machine learning model DTLN for noise suppression and acoustic echo cancellation on Raspberry Pi

The target of this project is to integrate and use two amazing pretrained models [DTLN](https://github.com/breizhn/DTLN) and [DTLN-aec](https://github.com/breizhn/DTLN-aec) on Raspberry Pi for realtime noise suppression (NS) and/or acoustic echo cancellation (AEC) tasks.


# Noise Suppression with DTLN

This is simple as the [DTLN](https://github.com/breizhn/DTLN) project already provides a realtime script for handling data from/to audio devices.
I add a few useful options in the `rt_dtln_ns.py` based on the orginal script. See `--help` for details.

## Setup

1. Configure Loopback and Test DTLN
  * Enable `snd-aloop` with `sudo modprobe snd_aloop`. You may want to add a line `snd-aloop` in `/etc/modules` to automatically enable it on boot.
  * Now check `arecord -l`, you should able to see two new Loopback devices.
  * Run DTLN with `python3 rt_dtln_ns.py -o 'Loopback 0' --measure`, you should see processing times < 6ms. If your processing time is longer you may need a more powerful device. If you see a lot of "input underflow" try to adjust the latency for a higher value, e.g., `--latency 0.5`.
  * Run `arecord -D hw:Loopback,1 rec.wav` in a separate shell to record denoised audio. Then listen to it or open with Audacity. You should noice obvious noise removal and clear voice.

2. Setup DTLN as a Service
  * Add the below entry to `/etc/asound.conf`. Which adds a virtual mic with DTLN output and set it as the default capturing device system wide.
  * Add the `dtln_ns.service` to `/etc/systemd/user/` and enable it with `systemctl --global enable dtln_ns`
  * Reboot and record some audio to see if DTLN NS is taking effect.


# Acoustic Echo Cancellation

This is based on the [DTLN-aec](https://github.com/breizhn/DTLN-aec) project. It currently only has a file-based demo script with tflite (not quantized) models. To make it realtime, I converted models to quantized models and created two realtime scripts:
* `models/dtln_aec_???_quant*` are quantized models. `???` is the number of LSTM units, larger means slower but supposed to be better.
* `rt_dtln_aec.py` is similar to `rt_dtln_ns.py`, which takes a pair of devices as input and output. It assumes the input device contains channel for loopback.
* `rt_dtln_aec_parallel.py` is a multiprocessing version, it runs ~2x faster on slower models.

## Setup with Hardware Loopback

1. Assume you have a sound card which supports hardware loopback, and the loopback is on the last channel of captured audio. In my case is the [Respeaker USB Mic Array V2.0](https://wiki.seeedstudio.com/ReSpeaker_Mic_Array_v2.0/).
2. List devices with `python3 rt_dtln_ns.py -l`. Note down a unique substring of your soundcard's name. In my case it can be "UAC1.0".
3. Test with `python3 rt_dtln_ns.py -i UAC1.0 -o UAC1.0 -c 6 -m models/dtln_aec_128_quant`. Speak to your mic, you should hear no feedback echo.

