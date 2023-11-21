#!/usr/bin/env python3

import argparse
import numpy
import simpleaudio


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("input", type=str,
		help="Raman spectrum file in LabSpec6's text dump format",
	)
	ap.add_argument("--duration", "-t", type=float, default=1.0,
		metavar="0.1-5.0",
		help="duration to play (in seconds); will be clipped if exceeding the "
			"range 0.1-5.0 [1.0]",
	)
	ap.add_argument("--sample-rate", "-r", type=int, default=44100,
		help="sample rate [44100]",
	)
	ap.add_argument("--volume", "-v", type=int, default=50,
		metavar="0-100",
		help="sound playback volume [50]",
	)

	# parse and refine args
	args = ap.parse_args()

	args.duration = min(args.duration, 5.0)
	args.duration = max(args.duration, 0.1)
	if args.sample_rate <= 0:
		raise ValueError("sample rate must be positive")
	args.volume = min(args.volume, 100)
	args.volume = max(args.volume, 0)

	return args


class RamanSpectrum(object):
	def __init__(self, wavenum, intens):
		wavenum = numpy.asarray(wavenum)
		intens = numpy.asarray(intens)
		if len(wavenum) != len(intens):
			raise ValueError("wavenum and intens must have matching lengths")
		self.wavenum = wavenum
		self.intens = intens
		return

	@classmethod
	def from_labspec_txt_dump(cls, f: str):
		raw = numpy.loadtxt(f, dtype=float, delimiter="\t", unpack=True)
		if len(raw) != 2:
			raise ValueError("the input file must be 2-column tab-delimited")
		wavenum, intens = raw.reshape(2, -1)
		new = cls(wavenum, intens)
		return new


def play_spectrum(spec: RamanSpectrum, *, sample_rate=44100, duration=1.0,
		volume=50,
	):

	n_samples = int(duration * sample_rate)
	audio = numpy.zeros(n_samples, dtype=float)
	time = numpy.linspace(0, duration, n_samples, False)

	# well inverse fft is better here but it requires scipy, so...
	for freq, intens in zip(spec.wavenum, spec.intens):
		audio += numpy.sin(freq * time * 2 * numpy.pi) * intens

	audio *= (volume / 100 * 32767 / numpy.abs(audio).max())
	audio = audio.astype(numpy.int16)

	playback = simpleaudio.play_buffer(audio, 1, 2, sample_rate)
	playback.wait_done()
	return


def main():
	args = get_args()
	spec = RamanSpectrum.from_labspec_txt_dump(args.input)
	play_spectrum(spec, sample_rate=args.sample_rate, duration=args.duration,
		volume=args.volume)
	return


if __name__ == "__main__":
	main()
