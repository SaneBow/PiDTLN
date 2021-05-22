# PiDTLN
Apply machine learning model DTLN for noise suppression and acoustic echo cancellation on Raspberry Pi

The target of this project is to integrate and use two amazing pretrained models [DTLN](https://github.com/breizhn/DTLN) and [DTLN-aec](https://github.com/breizhn/DTLN-aec) on Raspberry Pi for realtime noise suppression (NS) and/or acoustic echo cancellation (AEC) tasks.

# Noise Suppression with DTLN
This is simple as the [DTLN](https://github.com/breizhn/DTLN) project already provides a realtime script for handling data from/to audio devices.
I add a few useful options in the `rt_dtln_ns.py` based on the orginal script. See `--help` for details.

## Basic Usage

1. Configure Loopback and Test DTLN
  * Enable `snd-aloop` with `sudo modprobe snd_aloop`. You may want to add a line `snd-aloop` in `/etc/modules` to automatically enable it on boot.
  * Now check `arecord -l`, you should able to see two new Loopback devices.
  * Run DTLN with `python3 rt_dtln_ns.py -o 'Loopback 0' --measure`, you should see processing times < 6ms. If your processing time is longer you may need a more powerful device. If you see a lot of "input underflow" try to adjust the latency for a higher value, e.g., `--latency 0.5`.
  * Run `arecord -D hw:Loopback,1 rec.wav` in a separate shell to record denoised audio. Then listen to it or open with Audacity. You should noice obvious noise removal and clear voice.

