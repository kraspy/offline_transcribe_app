import gc
import logging
import subprocess
import tempfile
import warnings
from datetime import datetime as dt
from pathlib import Path

import click
import orjson
import torch
import whisperx
from rich.logging import RichHandler

# ========================================
# Логирование
# ========================================
warnings.filterwarnings(
    'ignore',
    message='.*TensorFloat-32 \\(TF32\\) has been disabled.*',
    module='pyannote\\.audio.*',
)

warnings.filterwarnings(
    'ignore',
    category=DeprecationWarning,
)

rich_handler = RichHandler(
    show_level=False,
    show_path=False,
    markup=True,
    rich_tracebacks=True,
    log_time_format='%Y.%m.%d %H:%M:%S',
)

logging.basicConfig(
    level=logging.INFO,
    handlers=[rich_handler],
    format='%(message)s',
)

# ========================================
# Константы
# ========================================
OUTPUT_DIR = Path('output')

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)

OUTPUT_FILE_TPL = OUTPUT_DIR / '%Y-%m-%d_%H-%M-%S'

OUTPUT_MD_FILE = OUTPUT_FILE_TPL.with_suffix('.md')


# ========================================
# dict -> TXT
# ========================================
def dialog_to_txt(
    dialog: list[dict],
    timestamp: bool = False,
    speaker: bool = False,
    newline: bool = False,
) -> str:
    text_result = ''

    for seg in dialog:
        start = seg.get('start', 0)
        end = seg.get('end', 0)
        txt = seg.get('text', '').strip()
        spk = seg.get('speaker', '')

        if timestamp:
            text_result += f'[{start:.2f}-{end:.2f}] '
        if speaker:
            text_result += f'{spk}: '

        text_result += f'{txt}'

        if newline:
            text_result += '\n'

    return text_result


# ========================================
# Функции обработки/транскрибации аудио
# ========================================
def _split_audio(
    file_path: Path,
    highpass: int = 120,
    lowpass: int = 2500,
    loudnorm_I: float = -23,
    loudnorm_TP: float = -3,
    loudnorm_LRA: float = 6,
    sample_rate: int = 16000,
    sample_fmt: str = 's16',
) -> tuple[Path, Path]:
    """
    Делит стереофайл на левый и правый каналыВозвращает пути к двум моно-файлам.
    """
    if not file_path.exists():
        logging.error(f'Файл для обработки не найден: {file_path}')
        return file_path, file_path

    try:
        with tempfile.NamedTemporaryFile(
            suffix='_left.wav',
            delete=False,
        ) as tmp_left:
            left_path = Path(tmp_left.name)

        with tempfile.NamedTemporaryFile(
            suffix='_right.wav', delete=False
        ) as tmp_right:
            right_path = Path(tmp_right.name)

        cmd_left = [
            'ffmpeg',
            '-y',
            '-i',
            str(file_path),
            '-filter_complex',
            (
                f'[0:a]channelsplit=channel_layout=stereo:channels=FL[FL];'
                f'[FL]loudnorm=I={loudnorm_I}:TP={loudnorm_TP}:LRA={loudnorm_LRA},'
                f'highpass=f={highpass},lowpass=f={lowpass}[FLout]'
            ),
            '-map',
            '[FLout]',
            '-ar',
            str(sample_rate),
            '-ac',
            '1',
            '-sample_fmt',
            sample_fmt,
            '-c:a',
            'pcm_s16le',
            str(left_path),
        ]

        cmd_right = [
            'ffmpeg',
            '-y',
            '-i',
            str(file_path),
            '-filter_complex',
            (
                f'[0:a]channelsplit=channel_layout=stereo:channels=FR[FR];'
                f'[FR]loudnorm=I={loudnorm_I}:TP={loudnorm_TP}:LRA={loudnorm_LRA},'
                f'highpass=f={highpass},lowpass=f={lowpass}[FRout]'
            ),
            '-map',
            '[FRout]',
            '-ar',
            str(sample_rate),
            '-ac',
            '1',
            '-sample_fmt',
            sample_fmt,
            '-c:a',
            'pcm_s16le',
            str(right_path),
        ]

        logging.info(f'⚙️  Запуск:\n{" ".join(cmd_left)}')
        subprocess.run(
            cmd_left, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )

        logging.info(f'⚙️  Запуск:\n{" ".join(cmd_right)}')
        subprocess.run(
            cmd_right, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )

        logging.info('✅ Разделение прошло успешно!')
        logging.info(f'🎙️  Левый канал сохранен: {left_path}')
        logging.info(f'🎙️  Правый канал сохранен: {right_path}')

        return left_path, right_path

    except Exception as e:
        logging.error(f'Ошибка при разделении аудио ({file_path.name}): {e}')
        return file_path, file_path


def _merge_segments(
    left_segments: list[dict],
    right_segments: list[dict],
) -> list[dict]:
    """
    Объединяет тексты левого и правого каналов.
    """
    merged = []
    for seg in left_segments:
        seg['speaker'] = 'speaker1'
        merged.append(seg)

    for seg in right_segments:
        seg['speaker'] = 'speaker2'
        merged.append(seg)

    return sorted(merged, key=lambda x: x['start'])


def _transcribe_with_whisperx(
    audio_path: Path,
    model_name: str = 'large-v2',
    diarize: bool = False,
) -> list:
    """
    Транскрибирует моноканальный аудиофайл с помощью WhisperX.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    compute_type = 'float16' if device == 'cuda' else 'int8'

    model_wx = whisperx.load_model(model_name, device=device, compute_type=compute_type)
    audio_data = whisperx.load_audio(str(audio_path))

    result_transcription = model_wx.transcribe(
        audio_data,
        language='ru',
        batch_size=5,
    )

    align_model, align_metadata = whisperx.load_align_model(
        language_code=result_transcription['language'], device=device
    )
    aligned_result = whisperx.align(
        result_transcription['segments'],
        align_model,
        align_metadata,
        audio_data,
        device=device,
        return_char_alignments=False,
    )

    del align_model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    result = []

    for seg in aligned_result.get('segments', []):
        seg_text = (seg.get('text') or '').strip()
        seg_start = (
            float(seg.get('start', 0.0)) if seg.get('start') is not None else 0.0
        )
        seg_end = (
            float(seg.get('end', seg_start))
            if seg.get('end') is not None
            else seg_start
        )

        words = seg.get('words') or []
        scores = []
        for w in words:
            s = w.get('score', None)
            if isinstance(s, (int, float)):
                scores.append(float(s))

        avg_score: float = sum(scores) / len(scores)

        result.append(
            {
                'start': seg_start,
                'end': seg_end,
                'text': seg_text,
                'avg_score': avg_score,
            }
        )
    print(result)
    return result


def transcribe_audio(
    file_path: str,
) -> str:
    """
    Старт транскрибации с помощью WhisperX.
    На вход подается путь к стерео-аудиофайлу.
    Разбивает аудио на каналы (left/right), транскрибирует каждый канал и склеивает их в итоговый текст.
    """

    original = Path(file_path)
    if not original.exists():
        logging.error(f'Файл для транскрибации не найден: {file_path}')
        return ''

    left_path, right_path = _split_audio(file_path=original)

    logging.info('🚀 Запуск WhisperX для левого канала...')
    left_segments = _transcribe_with_whisperx(left_path)

    logging.info('🚀 Запуск WhisperX для правого канала...')
    right_segments = _transcribe_with_whisperx(right_path)

    final_text = _merge_segments(left_segments, right_segments)

    symbol_count = len(' '.join([seg['text'] for seg in final_text]))

    logging.info(
        f'✅ Транскрибация завершена, итоговая длина текста: {symbol_count} символов.'
    )
    return final_text


# ========================================
# CLI
# ========================================
@click.command()
@click.option(
    '-p',
    '--path',
    'audio_path',
    required=True,
    type=click.Path(path_type=Path, exists=True),
    help='Путь к файлу или директории с аудио',
)
@click.option(
    '-v',
    '--verbose',
    'is_verbose',
    is_flag=True,
    help='Печать результатов транскрибации в консоль',
)
@click.option(
    '-tt',
    '--text-timestamp',
    'timestamp',
    is_flag=True,
    help='Включить отметку времени в тексте',
)
@click.option(
    '-ts',
    '--test-speaker',
    'speaker',
    is_flag=True,
    help='Включить отметку говорящего в тексте',
)
@click.option(
    '-tn',
    '--text-newline',
    'newline',
    is_flag=True,
    help='Включить перенос строки в тексте',
)
def main(
    audio_path: Path,
    is_verbose: bool,
    timestamp: bool,
    speaker: bool,
    newline: bool,
) -> None:
    """
    Транскрибирует файл или все файлы в директории и пишет результат в Markdown:
    ## Файл: <путь>
    ```
    <Текст диалога>
    ```
    ```
    <JSON диалога>
    ```
    """
    timestamp = dt.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_md_path = OUTPUT_DIR / f'{timestamp}.md'

    def write_result_md(md_path: Path, src_path: Path, dialog: list[dict]) -> None:
        txt = dialog_to_txt(
            dialog=dialog,
            timestamp=timestamp,
            speaker=speaker,
            newline=newline,
        )
        json_bytes = orjson.dumps(dialog, option=orjson.OPT_INDENT_2)

        with open(md_path, 'a', encoding='utf-8') as f:
            f.write(f'## Файл: {src_path}\n')
            f.write('```text\n')
            f.write(txt)
            if not txt.endswith('\n'):
                f.write('\n')
            f.write('```\n')
            f.write('```json\n')
            f.write(json_bytes.decode('utf-8'))
            if not json_bytes.endswith(b'\n'):
                f.write('\n')
            f.write('```\n\n---\n\n')

    if audio_path.is_file():
        result = transcribe_audio(str(audio_path))
        if result:
            write_result_md(output_md_path, audio_path, result)
            if is_verbose:
                logging.info(
                    '📝 Итоговый текст:\n%s', dialog_to_txt(result, newline=True)
                )
    else:
        # Обработка всех файлов верхнего уровня в директории
        for file in sorted(audio_path.iterdir()):
            if file.is_file():
                result = transcribe_audio(str(file))
                if result:
                    write_result_md(output_md_path, file, result)
                    if is_verbose:
                        logging.info(
                            'Итоговый текст (%s):\n%s',
                            file,
                            dialog_to_txt(result, newline=True),
                        )

    logging.info('✅ Результаты сохранены: %s', output_md_path)


if __name__ == '__main__':
    main()
