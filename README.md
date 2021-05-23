# PiDTLN

The target of this project is to integrate and use two amazing pretrained models [DTLN](https://github.com/breizhn/DTLN) and [DTLN-aec](https://github.com/breizhn/DTLN-aec) on Raspberry Pi for realtime noise suppression (NS) and/or acoustic echo cancellation (AEC) tasks.


# Noise Suppression with DTLN

This is simple as the [DTLN](https://github.com/breizhn/DTLN) project already provides a realtime script for handling data from/to audio devices.
I add a few useful options in the `rt_dtln_ns.py` based on the orginal script. See `--help` for details.

## Setup

### Configure Loopback and Test DTLN
  * Enable `snd-aloop` with `sudo modprobe snd_aloop`. You may want to add a line `snd-aloop` in `/etc/modules` to automatically enable it on boot.
  * Now check `arecord -l`, you should able to see two new Loopback devices.
  * Run DTLN with `python3 rt_dtln_ns.py -o 'Loopback 0' --measure`, you should see processing times < 6ms. If your processing time is longer you may need a more powerful device. If you see a lot of "input underflow" try to adjust the latency for a higher value, e.g., `--latency 0.5`.
  * Run `arecord -D hw:Loopback,1 rec.wav` in a separate shell to record denoised audio. Then listen to it or open with Audacity. You should noice obvious noise removal and clear voice.

### Setup DTLN as a Service
  * Add the below entry to `/etc/asound.conf`. Which adds a virtual mic with DTLN output and set it as the default capturing device system wide.
  * Add the `dtln_ns.service` to `/etc/systemd/user/` and enable it with `systemctl --global enable dtln_ns`
  * Reboot and record some audio to see if DTLN NS is taking effect.


# Acoustic Echo Cancellation

This is based on the [DTLN-aec](https://github.com/breizhn/DTLN-aec) project. It currently only has a file-based demo script with tflite (not quantized) models. To make it realtime, I converted models to quantized models and created two realtime scripts:
* `models/dtln_aec_???_quant*` are quantized models. `???` is the number of LSTM units, larger means slower but supposed to be better.
* `rt_dtln_aec.py` is similar to `rt_dtln_ns.py`, which takes a pair of devices as input and output. It assumes the input device contains channel for loopback.
* `rt_dtln_aec_mp.py` is a multiprocessing version, it runs close to 2x faster on slower models.

## Setup with Hardware Loopback

You need to have a sound card which supports hardware loopback, and the loopback is on the last channel of captured audio. In my case is the [Respeaker USB Mic Array V2.0](https://wiki.seeedstudio.com/ReSpeaker_Mic_Array_v2.0/), which has 6 input channels and last one is playback.

1. List devices with `python3 rt_dtln_ns.py -l`. Note down a unique substring of your soundcard's name. In my case it can be "UAC1.0".
2. Test with `python3 rt_dtln_ns.py -i UAC1.0 -o UAC1.0 -c 6 -m models/dtln_aec_128_quant`. Speak to your mic, you should hear no feedback echo.
3. Follow the similar procedure in DTLN NS setup to put AEC output to a virtual capturing device. So you can use it in other programs.

## Setup without Hardware Loopback

When you don't have a soundcard that supports hardware loopback, you need to create a virtual input device whose last channel stores playback loopback. I made a [ALSA AEC plugin](configs/aec_asound.conf) that can achieve this. Copy the file to `/etc/alsa/alsa.conf/50-aec.conf`. Then you will have two additional alsa interfaces: `aec` and `aec_internal`. To use them, simply do:
1. Play some music to AEC virtual device: `aplay -D aec:cardname music.wav`
2. Run AEC script with: `python3 rt_dtln_aec.py -m 128 -i aec_internal:cardname -o aec_internal:cardname`.
3. Record from AEC virtual device: `arecord -D aec:cardname -f S16_LE -r 16000 -c 1 -V mono rec.wav`

Now look at recorded audio file, music should be removed. The effect is not always good in my tests, possibly due to small model unit (128). You may try the same with `rt_dtln_aec_mp.py -m 256`. 
