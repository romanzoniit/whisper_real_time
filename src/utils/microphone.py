import speech_recognition as sr
from queue import Queue
from sys import platform
from src.utils.parser_args import args


class Microphone:
    def __init__(self):
        # The last time a recording was retreived from the queue.
        self.phrase_time = None
        # Current raw audio bytes.
        self.last_sample = bytes()
        # Thread safe Queue for passing data from the threaded recording callback.
        self.data_queue = Queue()
        # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = args.energy_threshold
        # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
        self.recorder.dynamic_energy_threshold = False
        self.source = None
        # Important for linux users.
        # Prevents permanent application hang and crash by using the wrong Microphone
        self.check_platform()

    def linux_platform(self):
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    self.source = sr.Microphone(sample_rate=16000, device_index=index)
                    break

    def check_platform(self):
        if 'linux' in platform:
            self.linux_platform()
        else:
            self.source = sr.Microphone(sample_rate=16000)