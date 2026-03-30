from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import wave
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import huggingface_hub
import imageio_ffmpeg
import orjson
from huggingface_hub.errors import LocalEntryNotFoundError
from tqdm.auto import tqdm


@dataclass(slots=True)
class PipelineConfig:
    input_audio: Path
    output_dir: Path
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str | None = "en"
    chunk_seconds: int = 20 * 60
    beam_size: int = 5
    vad_filter: bool = True
    word_timestamps: bool = True
    condition_on_previous_text: bool = False
    overwrite_chunks: bool = False
    overwrite_transcripts: bool = False
    split_only: bool = False
    ffmpeg_path: str | None = None
    model_cache_dir: Path | None = None


@dataclass(slots=True)
class ChunkInfo:
    index: int
    path: Path
    offset_seconds: float
    duration_seconds: float


MODEL_ALLOW_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]

MODEL_REQUIRED_FILES = (
    "config.json",
    "model.bin",
    "tokenizer.json",
)


class ProgressReporter:
    BAR_WIDTH = 28

    def __init__(self, chunks: list[ChunkInfo]) -> None:
        self.total_chunks = len(chunks)
        self.total_seconds = sum(chunk.duration_seconds for chunk in chunks)
        self.completed_seconds = 0.0
        self.current_chunk: ChunkInfo | None = None
        self.current_position = 0
        self.current_progress = 0.0
        self._last_line_length = 0

    def start_chunk(self, position: int, chunk: ChunkInfo) -> None:
        self.current_position = position
        self.current_chunk = chunk
        self.current_progress = 0.0
        self.render(0.0, note="准备中")

    def update(self, chunk_seconds: float) -> None:
        if self.current_chunk is None:
            return

        duration = max(self.current_chunk.duration_seconds, 0.0)
        bounded = max(chunk_seconds, 0.0)
        if duration:
            bounded = min(bounded, duration)
        if bounded < self.current_progress:
            return

        self.current_progress = bounded
        self.render(bounded)

    def finish_chunk(self) -> None:
        if self.current_chunk is None:
            return

        self.current_progress = max(
            self.current_progress,
            self.current_chunk.duration_seconds,
        )
        self.render(self.current_progress)
        self.completed_seconds += self.current_chunk.duration_seconds
        sys.stdout.write("\n")
        sys.stdout.flush()
        self.current_chunk = None
        self.current_progress = 0.0
        self._last_line_length = 0

    def render(self, chunk_seconds: float, note: str | None = None) -> None:
        if self.current_chunk is None:
            return

        overall_seconds = min(
            self.completed_seconds + chunk_seconds,
            self.total_seconds or self.completed_seconds + chunk_seconds,
        )
        ratio = 1.0 if self.total_seconds <= 0 else overall_seconds / self.total_seconds
        ratio = min(max(ratio, 0.0), 1.0)
        filled = min(self.BAR_WIDTH, int(ratio * self.BAR_WIDTH))
        bar = "#" * filled + "-" * (self.BAR_WIDTH - filled)
        note_text = f" | {note}" if note else ""
        line = (
            f"\r[{bar}] {ratio * 100:5.1f}% "
            f"{format_duration(overall_seconds)}/{format_duration(self.total_seconds)} "
            f"| 分段 {self.current_position}/{self.total_chunks} "
            f"{self.current_chunk.path.name} "
            f"{format_duration(chunk_seconds)}/{format_duration(self.current_chunk.duration_seconds)}"
            f"{note_text}"
        )
        padding = max(self._last_line_length - len(line), 0)
        sys.stdout.write(line + (" " * padding))
        sys.stdout.flush()
        self._last_line_length = len(line)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Type is not JSON serializable: {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        orjson.dumps(
            payload,
            option=orjson.OPT_INDENT_2,
            default=json_default,
        )
    )


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def resolve_ffmpeg_path(explicit_path: str | None) -> str:
    if explicit_path:
        return explicit_path
    return imageio_ffmpeg.get_ffmpeg_exe()


def ensure_dirs(output_dir: Path) -> tuple[Path, Path]:
    chunks_dir = output_dir / "chunks"
    transcripts_dir = output_dir / "transcripts"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    return chunks_dir, transcripts_dir


def get_wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frame_rate = handle.getframerate()
        if frame_rate <= 0:
            return 0.0
        return handle.getnframes() / float(frame_rate)


def build_chunk_info(index: int, path: Path, chunk_seconds: int) -> ChunkInfo:
    return ChunkInfo(
        index=index,
        path=path,
        offset_seconds=index * chunk_seconds,
        duration_seconds=get_wav_duration_seconds(path),
    )


def format_duration(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def normalize_language(language: str | None) -> str | None:
    if language is None:
        return None

    normalized = language.strip()
    if not normalized:
        return None
    if normalized.lower() in {"auto", "detect"}:
        return None
    return normalized


def resolve_model_cache_dir(config: PipelineConfig) -> str | None:
    if config.model_cache_dir is None:
        return None
    return str(config.model_cache_dir)


def resolve_model_repo_id(model_name: str) -> str:
    from faster_whisper.utils import _MODELS

    if "/" in model_name:
        return model_name

    repo_id = _MODELS.get(model_name)
    if repo_id is None:
        raise ValueError(
            f"Invalid model size '{model_name}', expected one of: {', '.join(_MODELS.keys())}"
        )
    return repo_id


def snapshot_download_with_progress(
    *,
    repo_id: str,
    cache_dir: str | None,
    local_files_only: bool,
    force_download: bool = False,
) -> str:
    return huggingface_hub.snapshot_download(
        repo_id,
        local_files_only=local_files_only,
        allow_patterns=MODEL_ALLOW_PATTERNS,
        cache_dir=cache_dir,
        force_download=force_download,
        tqdm_class=tqdm,
    )


def model_artifacts_are_complete(model_path: Path) -> bool:
    if not model_path.is_dir():
        return False

    for filename in MODEL_REQUIRED_FILES:
        if not (model_path / filename).is_file():
            return False

    return any(model_path.glob("vocabulary.*"))


def purge_invalid_model_snapshot(model_path: Path) -> None:
    if not model_path.exists():
        return

    snapshot_root = model_path.parent
    repo_root = snapshot_root.parent

    if snapshot_root.name == "snapshots" and repo_root.name.startswith("models--"):
        shutil.rmtree(model_path, ignore_errors=True)
        blobs_dir = repo_root / "blobs"
        if blobs_dir.is_dir():
            for incomplete_blob in blobs_dir.glob("*.incomplete"):
                incomplete_blob.unlink(missing_ok=True)


def ensure_model_available(config: PipelineConfig) -> str:
    model_path = Path(config.model)
    if model_path.is_dir():
        if not model_artifacts_are_complete(model_path):
            raise RuntimeError(
                f"Model directory is incomplete: {model_path}. "
                "Expected config.json, model.bin, tokenizer.json, and vocabulary.*"
            )
        return str(model_path.resolve())

    repo_id = resolve_model_repo_id(config.model)
    cache_dir = resolve_model_cache_dir(config)

    try:
        cached_path = Path(
            snapshot_download_with_progress(
                repo_id=repo_id,
                cache_dir=cache_dir,
                local_files_only=True,
            )
        )
        if model_artifacts_are_complete(cached_path):
            return str(cached_path)

        print(f"检测到模型缓存不完整，准备重新下载: {config.model}")
        purge_invalid_model_snapshot(cached_path)
        repaired_path = Path(
            snapshot_download_with_progress(
                repo_id=repo_id,
                cache_dir=cache_dir,
                local_files_only=False,
                force_download=True,
            )
        )
        if not model_artifacts_are_complete(repaired_path):
            raise RuntimeError(
                f"Model download finished but required files are still missing: {repaired_path}"
            )
        return str(repaired_path)
    except LocalEntryNotFoundError:
        print(f"首次下载模型: {config.model}")
        print("正在显示模型下载进度，请稍候...")
        downloaded_path = Path(
            snapshot_download_with_progress(
                repo_id=repo_id,
                cache_dir=cache_dir,
                local_files_only=False,
            )
        )
        if not model_artifacts_are_complete(downloaded_path):
            raise RuntimeError(
                f"Model download finished but required files are still missing: {downloaded_path}"
            )
        return str(downloaded_path)


def split_audio(config: PipelineConfig) -> list[ChunkInfo]:
    chunks_dir, _ = ensure_dirs(config.output_dir)
    existing_chunks = sorted(chunks_dir.glob("chunk_*.wav"))
    if existing_chunks and not config.overwrite_chunks:
        return [
            build_chunk_info(index=index, path=path, chunk_seconds=config.chunk_seconds)
            for index, path in enumerate(existing_chunks)
        ]

    for path in existing_chunks:
        path.unlink()

    ffmpeg_path = resolve_ffmpeg_path(config.ffmpeg_path)
    chunk_pattern = chunks_dir / "chunk_%04d.wav"

    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-nostdin",
        "-i",
        str(config.input_audio),
        "-vn",
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-f",
        "segment",
        "-segment_time",
        str(config.chunk_seconds),
        "-reset_timestamps",
        "1",
        str(chunk_pattern),
    ]
    run_command(command)

    chunk_files = sorted(chunks_dir.glob("chunk_*.wav"))
    if not chunk_files:
        raise RuntimeError("No audio chunks were produced.")

    chunks = [
        build_chunk_info(index=index, path=path, chunk_seconds=config.chunk_seconds)
        for index, path in enumerate(chunk_files)
    ]
    write_json(
        config.output_dir / "chunk_manifest.json",
        {
            "source_audio": str(config.input_audio),
            "chunk_seconds": config.chunk_seconds,
            "chunks": [
                {
                    "index": chunk.index,
                    "path": str(chunk.path),
                    "offset_seconds": chunk.offset_seconds,
                    "duration_seconds": round(chunk.duration_seconds, 3),
                }
                for chunk in chunks
            ],
        },
    )
    return chunks


def load_whisper_model(config: PipelineConfig):
    from faster_whisper import WhisperModel

    model_path = ensure_model_available(config)
    return WhisperModel(
        model_path,
        device=config.device,
        compute_type=config.compute_type,
    )


def transcribe_chunk(
    model: Any,
    config: PipelineConfig,
    chunk: ChunkInfo,
    progress_callback: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    _, transcripts_dir = ensure_dirs(config.output_dir)
    chunk_output = transcripts_dir / f"chunk_{chunk.index:04d}.json"
    if chunk_output.exists() and not config.overwrite_transcripts:
        if progress_callback is not None:
            progress_callback(chunk.duration_seconds)
        return orjson.loads(chunk_output.read_bytes())

    segment_iter, info = model.transcribe(
        str(chunk.path),
        language=config.language,
        task="transcribe",
        beam_size=config.beam_size,
        vad_filter=config.vad_filter,
        word_timestamps=config.word_timestamps,
        condition_on_previous_text=config.condition_on_previous_text,
    )

    segments_payload: list[dict[str, Any]] = []
    text_parts: list[str] = []

    for segment in segment_iter:
        segment_payload = {
            "id": segment.id,
            "start": round(segment.start + chunk.offset_seconds, 3),
            "end": round(segment.end + chunk.offset_seconds, 3),
            "text": segment.text.strip(),
        }
        if config.word_timestamps:
            words = []
            for word in segment.words or []:
                if word.start is None or word.end is None:
                    continue
                words.append(
                    {
                        "word": word.word,
                        "start": round(word.start + chunk.offset_seconds, 3),
                        "end": round(word.end + chunk.offset_seconds, 3),
                        "probability": word.probability,
                    }
                )
            segment_payload["words"] = words

        segments_payload.append(segment_payload)
        if segment_payload["text"]:
            text_parts.append(segment_payload["text"])

        if progress_callback is not None:
            progress_callback(segment.end)

    if progress_callback is not None:
        progress_callback(chunk.duration_seconds)

    payload = {
        "chunk_index": chunk.index,
        "chunk_path": str(chunk.path),
        "offset_seconds": chunk.offset_seconds,
        "duration_seconds": round(chunk.duration_seconds, 3),
        "detected_language": getattr(info, "language", config.language),
        "language_probability": getattr(info, "language_probability", None),
        "duration_after_vad": getattr(info, "duration_after_vad", None),
        "segments": segments_payload,
        "text": " ".join(text_parts).strip(),
    }
    write_json(chunk_output, payload)
    return payload


def merge_results(config: PipelineConfig, chunk_results: list[dict[str, Any]]) -> dict[str, Any]:
    merged_segments: list[dict[str, Any]] = []
    merged_text_parts: list[str] = []

    for chunk in chunk_results:
        merged_segments.extend(chunk["segments"])
        if chunk["text"]:
            merged_text_parts.append(chunk["text"])

    merged_text = "\n".join(merged_text_parts).strip()
    payload = {
        "source_audio": str(config.input_audio),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "segments": merged_segments,
        "text": merged_text,
    }
    write_json(config.output_dir / "transcript.json", payload)
    (config.output_dir / "transcript.txt").write_text(merged_text, encoding="utf-8")
    return payload


def run_pipeline(config: PipelineConfig, model: Any | None = None) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    chunks = split_audio(config)
    if config.split_only:
        payload = {
            "source_audio": str(config.input_audio),
            "chunk_count": len(chunks),
            "chunk_seconds": config.chunk_seconds,
            "chunks": [
                {
                    "index": chunk.index,
                    "path": str(chunk.path),
                    "offset_seconds": chunk.offset_seconds,
                    "duration_seconds": round(chunk.duration_seconds, 3),
                }
                for chunk in chunks
            ],
        }
        write_json(config.output_dir / "split_only.json", payload)
        print(f"Wrote chunks to {config.output_dir / 'chunks'}")
        return payload

    model = model or load_whisper_model(config)
    results = []
    progress = ProgressReporter(chunks)

    print(f"开始转写，共 {len(chunks)} 个分段。")
    for position, chunk in enumerate(chunks, start=1):
        progress.start_chunk(position, chunk)
        results.append(
            transcribe_chunk(
                model,
                config,
                chunk,
                progress_callback=progress.update,
            )
        )
        progress.finish_chunk()

    merged = merge_results(config, results)
    print(f"转写完成，结果已写入: {config.output_dir}")
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chunk large audio files and transcribe them with faster-whisper."
    )
    parser.add_argument("input_audio", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for chunks and transcripts. Defaults to outputs/<input stem>.",
    )
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--language", default="en")
    parser.add_argument("--chunk-minutes", type=int, default=20)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--ffmpeg-path", default=None)
    parser.add_argument("--model-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--overwrite-chunks",
        action="store_true",
        help="Rebuild chunk WAV files even if they already exist.",
    )
    parser.add_argument(
        "--overwrite-transcripts",
        action="store_true",
        help="Re-run chunk transcription even if JSON files already exist.",
    )
    parser.add_argument(
        "--split-only",
        action="store_true",
        help="Only split the source audio into chunks and stop before transcription.",
    )
    parser.add_argument(
        "--disable-vad",
        action="store_true",
        help="Turn off VAD filtering in faster-whisper.",
    )
    parser.add_argument(
        "--disable-word-timestamps",
        action="store_true",
        help="Turn off word-level timestamps.",
    )
    parser.add_argument(
        "--condition-on-previous-text",
        action="store_true",
        help="Pass previous text into decoding. Default is off for safer chunk independence.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    output_dir = args.output_dir or Path("outputs") / args.input_audio.stem
    return PipelineConfig(
        input_audio=args.input_audio.resolve(),
        output_dir=output_dir.resolve(),
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=normalize_language(args.language),
        chunk_seconds=args.chunk_minutes * 60,
        beam_size=args.beam_size,
        vad_filter=not args.disable_vad,
        word_timestamps=not args.disable_word_timestamps,
        condition_on_previous_text=args.condition_on_previous_text,
        overwrite_chunks=args.overwrite_chunks,
        overwrite_transcripts=args.overwrite_transcripts,
        split_only=args.split_only,
        ffmpeg_path=args.ffmpeg_path,
        model_cache_dir=args.model_cache_dir.resolve() if args.model_cache_dir else None,
    )
