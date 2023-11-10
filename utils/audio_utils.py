import os
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np


class AudioIOUtils():
    """The ``AudioIOUtils`` class helps user to prepare audio file for
    inference or learning.
    """

    def __init__(self):
        pass

    def prepare_audio_for_kws(self, wav_file, rate_in):
        """Prepare audio sample for KWS:
            * Resample file to 16kHz
            * Trimmed file to get audio data in a one second sample
                (cf. https://github.com/petewarden/extract_loudest_section)
            * Encode file to WAV format

        :args:
            wav_file(Numpy.array): A Numpy array with raw audio data
            rate_in(int): Input audio rate

        :return:
            A WAV audio file.
        """
        # Convert Numpy.array to Tensor
        data_tf = tf.convert_to_tensor(wav_file, np.float32)

        # Get Mono stream and resample it to 16KHz
        resampled_audio = tfio.audio.resample(data_tf[:, :-1],
                                              rate_in=rate_in,
                                              rate_out=16000)

        start = 0
        stop = 0
        desired_samples = 16000
        step = 250
        current_volume_sum = 0.0

        current_volume_sum = np.sum(np.square(resampled_audio[:16000]))

        loudest_end_index = desired_samples
        loudest_volume = current_volume_sum

        # Going accross the audio file with a custom step
        # A higher value decrease precision on loudest section extraction but
        # the loop will run faster. In our case, audio file are not complex,
        # it's preferable to save processing time to get a responsive app.
        for i in range(16000, len(resampled_audio), step):
            # Start point
            trailing_value = resampled_audio[i - desired_samples]
            current_volume_sum -= abs(trailing_value)

            # End point
            leading_value = resampled_audio[i]
            current_volume_sum += abs(leading_value)

            if current_volume_sum > loudest_volume:
                loudest_volume = current_volume_sum
                loudest_end_index = i

        # Get sound markers
        loudest_start_index = loudest_end_index - desired_samples
        start = loudest_start_index
        stop = loudest_end_index

        # Encode audio file to WAV format
        encoded_wav = tf.compat.v1.audio.encode_wav(resampled_audio[start:stop],
                                                    sample_rate=16000)

        return encoded_wav

    def save_audio(self, wav_data, filename):
        """Save WAV audio data to a WAV file.

        :args:
            wav_data(EagerTensor): WAV audio data
            filename(str): output filename
        """
        tf.io.write_file(filename, wav_data)
