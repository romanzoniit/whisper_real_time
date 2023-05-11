import argparse
from sys import platform


def parser_kwargs():
    parser = argparse.ArgumentParser(description="Transcribe in realtime with whisper model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                     )
    parser.add_argument("--model",
                        default="large",
                        help="Whisper model to use",
                        choices=["tiny", "base", "small", "medium", "large"],
                        type=str)
    parser.add_argument("--language",
                        default="ru",
                        choices=["en", "ru"],
                        help="Language to use",
                        type=str)
    parser.add_argument("--energy_threshold",
                        default=1000,
                        help="Energy level for mic to detect.",
                        type=int)
    parser.add_argument("--record_timeout",
                        default=2,
                        help="How real time the recording is in seconds.",
                        type=float)
    parser.add_argument("--phrase_timeout",
                        default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.",
                        type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.",
                            type=str)
    return parser.parse_args()


args = parser_kwargs()