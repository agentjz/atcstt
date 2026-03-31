from __future__ import annotations

import sys
import wave
from argparse import Namespace
from pathlib import Path

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import atc_asr.pipeline as pipeline_module
from atc_asr.pipeline import (
    ChunkInfo,
    PipelineConfig,
    cleanup_spawned_processes,
    config_from_args,
    ensure_model_available,
    model_artifacts_are_complete,
    purge_invalid_model_snapshot,
    run_command,
    run_pipeline,
    transcribe_chunk,
)


class DummyWord:
    def __init__(self, word: str, start: float, end: float, probability: float) -> None:
        self.word = word
        self.start = start
        self.end = end
        self.probability = probability


class DummySegment:
    def __init__(
        self,
        segment_id: int,
        start: float,
        end: float,
        text: str,
        words: list[DummyWord],
    ) -> None:
        self.id = segment_id
        self.start = start
        self.end = end
        self.text = text
        self.words = words


class DummyInfo:
    language = "en"
    language_probability = 0.99
    duration_after_vad = 1.0


class DummyModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def transcribe(self, audio_path: str, **kwargs: object):
        self.calls.append({"audio_path": audio_path, **kwargs})
        segments = [
            DummySegment(
                segment_id=0,
                start=0.0,
                end=0.4,
                text="hello",
                words=[DummyWord("hello", 0.0, 0.4, 0.95)],
            ),
            DummySegment(
                segment_id=1,
                start=0.4,
                end=1.0,
                text="world",
                words=[DummyWord("world", 0.4, 1.0, 0.93)],
            ),
        ]
        return iter(segments), DummyInfo()


def write_wav(path: Path, duration_seconds: float = 1.0, sample_rate: int = 16000) -> None:
    total_frames = int(duration_seconds * sample_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * total_frames)


def test_transcribe_chunk_reports_progress_and_writes_json(tmp_path) -> None:
    chunk_path = tmp_path / "chunks" / "chunk_0000.wav"
    write_wav(chunk_path)
    config = PipelineConfig(
        input_audio=chunk_path,
        output_dir=tmp_path / "output",
    )
    chunk = ChunkInfo(
        index=0,
        path=chunk_path,
        offset_seconds=0.0,
        duration_seconds=1.0,
    )
    updates: list[float] = []
    model = DummyModel()

    payload = transcribe_chunk(
        model,
        config,
        chunk,
        progress_callback=updates.append,
    )

    assert updates == [0.4, 1.0, 1.0]
    assert payload["text"] == "hello world"
    transcript_json = config.output_dir / "transcripts" / "chunk_0000.json"
    assert transcript_json.exists()
    assert model.calls[0]["language"] == "en"


def test_run_pipeline_prints_realtime_progress(monkeypatch, tmp_path, capsys) -> None:
    chunk_path = tmp_path / "chunks" / "chunk_0000.wav"
    write_wav(chunk_path)
    config = PipelineConfig(
        input_audio=chunk_path,
        output_dir=tmp_path / "output",
    )
    chunk = ChunkInfo(
        index=0,
        path=chunk_path,
        offset_seconds=0.0,
        duration_seconds=1.0,
    )

    monkeypatch.setattr("atc_asr.pipeline.split_audio", lambda _: [chunk])
    monkeypatch.setattr("atc_asr.pipeline.load_whisper_model", lambda _: DummyModel())

    payload = run_pipeline(config)

    assert payload["text"] == "hello world"
    output = capsys.readouterr().out
    assert "开始转写，共 1 个分段。" in output
    assert "chunk_0000.wav" in output
    assert "100.0%" in output
    assert (config.output_dir / "transcript.txt").read_text(encoding="utf-8") == "hello world"


def test_config_from_args_accepts_auto_language(tmp_path) -> None:
    input_audio = tmp_path / "sample.wav"
    input_audio.write_bytes(b"fake")
    args = Namespace(
        input_audio=input_audio,
        output_dir=None,
        model="large-v3",
        device="cuda",
        compute_type="float16",
        language="auto",
        chunk_minutes=20,
        beam_size=5,
        ffmpeg_path=None,
        model_cache_dir=None,
        overwrite_chunks=False,
        overwrite_transcripts=False,
        split_only=False,
        disable_vad=False,
        disable_word_timestamps=False,
        condition_on_previous_text=False,
    )

    config = config_from_args(args)

    assert config.language is None


def test_ensure_model_available_uses_cache_when_present(monkeypatch, tmp_path) -> None:
    config = PipelineConfig(
        input_audio=tmp_path / "audio.wav",
        output_dir=tmp_path / "output",
        model="large-v3",
    )
    calls: list[tuple[str, str | None, bool]] = []

    def fake_snapshot_download_with_progress(
        *,
        repo_id: str,
        cache_dir: str | None,
        local_files_only: bool,
        force_download: bool = False,
    ) -> str:
        calls.append((repo_id, cache_dir, local_files_only))
        cached_model = tmp_path / "cached-model"
        cached_model.mkdir(parents=True, exist_ok=True)
        (cached_model / "config.json").write_text("{}", encoding="utf-8")
        (cached_model / "tokenizer.json").write_text("{}", encoding="utf-8")
        (cached_model / "vocabulary.txt").write_text("token", encoding="utf-8")
        (cached_model / "model.bin").write_bytes(b"ok")
        return str(cached_model)

    monkeypatch.setattr("atc_asr.pipeline.snapshot_download_with_progress", fake_snapshot_download_with_progress)

    model_path = ensure_model_available(config)

    assert model_path == str(tmp_path / "cached-model")
    assert calls == [("Systran/faster-whisper-large-v3", None, True)]


def test_model_artifacts_are_complete_requires_model_bin(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "vocabulary.txt").write_text("token", encoding="utf-8")

    assert not model_artifacts_are_complete(model_dir)


def test_purge_invalid_model_snapshot_removes_snapshot_and_incomplete_blob(tmp_path) -> None:
    repo_root = tmp_path / "models--Systran--faster-whisper-tiny.en"
    snapshot_dir = repo_root / "snapshots" / "abc123"
    blob_dir = repo_root / "blobs"
    snapshot_dir.mkdir(parents=True)
    blob_dir.mkdir(parents=True)
    (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")
    incomplete_blob = blob_dir / "broken.incomplete"
    incomplete_blob.write_bytes(b"")

    purge_invalid_model_snapshot(snapshot_dir)

    assert not snapshot_dir.exists()
    assert not incomplete_blob.exists()


def test_ensure_model_available_redownloads_when_cached_snapshot_is_incomplete(
    monkeypatch, tmp_path, capsys
) -> None:
    config = PipelineConfig(
        input_audio=tmp_path / "audio.wav",
        output_dir=tmp_path / "output",
        model="tiny.en",
    )
    incomplete_path = tmp_path / "models--Systran--faster-whisper-tiny.en" / "snapshots" / "bad"
    incomplete_path.mkdir(parents=True)
    (incomplete_path / "config.json").write_text("{}", encoding="utf-8")
    (incomplete_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (incomplete_path / "vocabulary.txt").write_text("token", encoding="utf-8")
    repaired_path = tmp_path / "models--Systran--faster-whisper-tiny.en" / "snapshots" / "good"
    repaired_path.mkdir(parents=True)
    (repaired_path / "config.json").write_text("{}", encoding="utf-8")
    (repaired_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (repaired_path / "vocabulary.txt").write_text("token", encoding="utf-8")
    (repaired_path / "model.bin").write_bytes(b"ok")
    calls: list[tuple[str, str | None, bool, bool]] = []

    def fake_snapshot_download_with_progress(
        *,
        repo_id: str,
        cache_dir: str | None,
        local_files_only: bool,
        force_download: bool = False,
    ) -> str:
        calls.append((repo_id, cache_dir, local_files_only, force_download))
        if local_files_only:
            return str(incomplete_path)
        return str(repaired_path)

    monkeypatch.setattr("atc_asr.pipeline.snapshot_download_with_progress", fake_snapshot_download_with_progress)

    model_path = ensure_model_available(config)

    assert model_path == str(repaired_path)
    assert calls == [
        ("Systran/faster-whisper-tiny.en", None, True, False),
        ("Systran/faster-whisper-tiny.en", None, False, True),
    ]
    output = capsys.readouterr().out
    assert "检测到模型缓存不完整，准备重新下载: tiny.en" in output
    assert not incomplete_path.exists()


def test_ensure_model_available_shows_download_message_on_first_download(monkeypatch, tmp_path, capsys) -> None:
    config = PipelineConfig(
        input_audio=tmp_path / "audio.wav",
        output_dir=tmp_path / "output",
        model="large-v3",
    )
    calls: list[tuple[str, str | None, bool]] = []

    def fake_snapshot_download_with_progress(
        *,
        repo_id: str,
        cache_dir: str | None,
        local_files_only: bool,
        force_download: bool = False,
    ) -> str:
        calls.append((repo_id, cache_dir, local_files_only))
        if local_files_only:
            raise LocalEntryNotFoundError("missing")
        downloaded_model = tmp_path / "downloaded-model"
        downloaded_model.mkdir(parents=True, exist_ok=True)
        (downloaded_model / "config.json").write_text("{}", encoding="utf-8")
        (downloaded_model / "tokenizer.json").write_text("{}", encoding="utf-8")
        (downloaded_model / "vocabulary.txt").write_text("token", encoding="utf-8")
        (downloaded_model / "model.bin").write_bytes(b"ok")
        return str(downloaded_model)

    monkeypatch.setattr("atc_asr.pipeline.snapshot_download_with_progress", fake_snapshot_download_with_progress)

    model_path = ensure_model_available(config)

    assert model_path == str(tmp_path / "downloaded-model")
    assert calls == [
        ("Systran/faster-whisper-large-v3", None, True),
        ("Systran/faster-whisper-large-v3", None, False),
    ]
    output = capsys.readouterr().out
    assert "首次下载模型: large-v3" in output
    assert "正在显示模型下载进度" in output


def test_run_command_terminates_child_process_when_wait_is_interrupted(monkeypatch) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.pid = 4321
            self.returncode = None
            self.terminated = 0

        def wait(self, timeout=None):
            if timeout is not None:
                return self.returncode
            raise KeyboardInterrupt

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated += 1
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    process = FakeProcess()
    monkeypatch.setattr("atc_asr.pipeline.subprocess.Popen", lambda command: process)

    with pytest.raises(KeyboardInterrupt):
        run_command(["ffmpeg", "-version"])

    assert process.terminated == 1
    assert not pipeline_module.ACTIVE_CHILD_PROCESSES


def test_cleanup_spawned_processes_terminates_only_tracked_children() -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.pid = 2468
            self.returncode = None
            self.terminated = 0

        def wait(self, timeout=None):
            return self.returncode

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated += 1
            self.returncode = 0

        def kill(self):
            self.returncode = 0

    process = FakeProcess()
    pipeline_module.track_child_process(process)

    cleanup_spawned_processes()

    assert process.terminated == 1
    assert not pipeline_module.ACTIVE_CHILD_PROCESSES
