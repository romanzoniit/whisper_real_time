import io
import os
from time import sleep
from datetime import datetime, timedelta

import torch
import whisper
import speech_recognition as sr
from tempfile import NamedTemporaryFile


from src.utils.microphone import Microphone
from src.utils.parser_args import args


class Recognition(Microphone):
    def __init__(self):
        super().__init__()
        self.model = args.model
        self.audio_model = whisper.load_model(self.model)
        print("Model loaded.\n")
        self.record_timeout = args.record_timeout
        self.phrase_timeout = args.phrase_timeout
        self.temp_file = NamedTemporaryFile().name
        self.transcription = ['']
        self.calibrate_energy_threshold()
        self.recorder.listen_in_background(self.source, self.record_callback, phrase_time_limit=self.record_timeout)

    def calibrate_energy_threshold(self):
        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)

    def record_callback(self, _, audio: sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        self.data_queue.put(data)

    def recognize(self, language=args.language):
        if language == "en":
            args.langauge = "en"
        # Cue the user that we're ready to go.

        while True:
            try:
                now = datetime.utcnow()
                # Pull raw recorded audio from the queue.
                if not self.data_queue.empty():
                    phrase_complete = False
                    # If enough time has passed between recordings, consider the phrase complete.
                    # Clear the current working audio buffer to start over with the new data.
                    if self.phrase_time and now - self.phrase_time > timedelta(seconds=self.phrase_timeout):
                        self.last_sample = bytes()
                        phrase_complete = True
                    # This is the last time we received new audio data from the queue.
                    self.phrase_time = now

                    # Concatenate our current audio data with the latest audio data.
                    while not self.data_queue.empty():
                        data = self.data_queue.get()
                        self.last_sample += data

                    # Use AudioData to convert the raw data to wav data.
                    audio_data = sr.AudioData(self.last_sample, self.source.SAMPLE_RATE, self.source.SAMPLE_WIDTH)
                    wav_data = io.BytesIO(audio_data.get_wav_data())

                    # Write wav data to the temporary file as bytes.
                    with open(self.temp_file, 'w+b') as f:
                        f.write(wav_data.read())

                    # Read the transcription.
                    result = self.audio_model.transcribe(self.temp_file, language=args.language, fp16=torch.cuda.is_available())
                    text = result['text'].strip()

                    # If we detected a pause between recordings, add a new item to our transcripion.
                    # 'Otherwise' edit the existing one.
                    if phrase_complete:
                        self.transcription.append(text)
                    else:
                        self.transcription[-1] = text

                    # Clear the console to reprint the updated transcription.
                    os.system('cls' if os.name == 'nt' else 'clear')
                    for line in self.transcription:
                        print(line)
                    # Flush stdout.
                    print('', end='', flush=True)

                    # Infinite loops are bad for processors, must sleep.
                    sleep(0.25)
            except KeyboardInterrupt:
                break

        print("\n\nTranscription:")
        for line in self.transcription:
            print(line)
        return self.transcription


if __name__ == "__main__":
    rec = Recognition()
    rec.recognize()