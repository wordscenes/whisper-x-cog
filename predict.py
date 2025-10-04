import json
import os
import typing
from importlib.metadata import version

from cog import BasePredictor, Path, Input
import whisperx


# all model files will be downloaded to this directory
MODEL_CACHE = Path(__file__).parent / 'model_cache'
temperature_increment_on_fallback = 0.2

# Not v3! See "rejected experiments" in readme
WHISPER_MODEL = 'large-v2'

TARGET_LANGUAGES = [
    'ar',
    'zh',
    'en',
    'fr',
    'de',
    'hi',
    'id',
    'it',
    'ja',
    'ko',
    'pt',
    'ru',
    'es',
    'tr',
]

OUR_ALIGN_MODELS_HF = {
    'id': 'indonesian-nlp/wav2vec2-large-xlsr-indonesian',
}

def report_versions():
    print(f'Using whisperx version {version("whisperx")}')
    print(f'Using Whisper model {WHISPER_MODEL}')

def load_align_model(language: str):
    return whisperx.load_align_model(
        language_code=language,
        device='cuda',
        model_name = OUR_ALIGN_MODELS_HF.get(language, None),
        model_dir=str(MODEL_CACHE),
    )

class Predictor(BasePredictor):
    def setup(self, weights = None, download_all_align_models = False):
        """Load the model into memory to make running multiple predictions efficient"""

        # Ensure the model cache directory exists
        os.makedirs(MODEL_CACHE, exist_ok=True)

        self.model = whisperx.load_model(
            WHISPER_MODEL,
            device='cuda',
            compute_type='float16',
            download_root=str(MODEL_CACHE),
            asr_options={'suppress_numerals': True},
        )
        self.align_models = {}
        if download_all_align_models:
            for lang in TARGET_LANGUAGES:
                self.align_models[lang] = load_align_model(lang)

    def predict(  # pyright: ignore[reportIncompatibleMethodOverride] (Pyright complains about **kwargs missing, but cog doesn't support untyped parameters)
        self,
        audio_path: Path = Input(description='Audio to transcribe or align'),
        mode: str = Input(default="transcribe", choices=["transcribe", "align"], description="Mode: 'transcribe' to generate transcript, 'align' to align provided segments"),
        segments: str = Input(default="", description="Segments (JSON array with text, start, and end keys) to align with audio (required when mode='align')"),
        language: str = Input(default="en", description="Language to transcribe"),
    ) -> str:
        report_versions()
        
        if language not in TARGET_LANGUAGES:
            raise ValueError(f'Language {language} is not supported by the current align model cache ({TARGET_LANGUAGES})')

        audio = whisperx.load_audio(audio_path)
        
        if mode == 'transcribe':
            result = self.model.transcribe(audio, language=language)
        else:
            if segments == '':
                raise ValueError('Segments are required when mode is align')
            result = {'segments': json.loads(segments)}

        if language not in self.align_models:
            self.align_models[language] = load_align_model(language)

        align_model = self.align_models[language]
        result = whisperx.align(
            result['segments'],
            align_model[0],
            align_model[1],
            audio,
            'cuda',
            return_char_alignments=False,
        )

        if not isinstance(result, dict) and callable(getattr(result, 'to_dict')):
            result = result.to_dict()
        output = json.dumps(result, allow_nan=True, ensure_ascii=False)

        return output
