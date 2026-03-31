from __future__ import annotations

import sys
import wave
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run
from atc_asr import launcher


def write_wav(path: Path, duration_seconds: float = 1.0, sample_rate: int = 16000) -> None:
    total_frames = int(duration_seconds * sample_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * total_frames)


def test_main_prompts_for_input_then_device_model_and_language(
    monkeypatch, tmp_path, capsys
) -> None:
    input_audio = tmp_path / "sample.wav"
    write_wav(input_audio)
    captured: list[object] = []
    answers = iter([str(input_audio), "", "2", "3", "2", "q"])

    monkeypatch.setattr(sys, "argv", ["run.py"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr(launcher, "probe_media_duration_seconds", lambda _: 1.0)
    monkeypatch.setattr(launcher, "run_pipeline", lambda config: captured.append(config))

    run.main()

    config = captured[0]
    assert config.device == "cpu"
    assert config.model == "small.en"
    assert config.language == "zh"
    assert config.compute_type == "int8"
    assert config.input_audio == input_audio.resolve()
    assert config.output_dir == (ROOT / "outputs" / input_audio.stem).resolve()

    output = capsys.readouterr().out
    assert "atc-stt" in output
    assert "██╔══██╗" in output
    assert "会话:" in output
    assert "目录:" in output
    assert "用途    : 本地 ATC 音频转写，支持单文件和文件夹递归批处理" in output
    assert "输入    : 可直接拖入单个音频/视频文件，或拖入整个文件夹后回车" in output
    assert "筛选    : 文件夹模式支持 all | 1,3,5-8 | only new | failed only | ext:mp3,wav | name:keyword" in output
    assert "重跑    : 已有结果默认跳过；需要强制重跑请使用 --overwrite" in output
    assert "脚本化  : 需要固定参数或自动化运行时，请使用 atc-asr-cli" in output
    assert "扫描摘要" in output
    assert "执行计划" in output
    assert "请选择运行设备" in output
    assert "请选择转写模型" in output
    assert "请选择识别语言" in output
    assert "输入路径" in output
    assert "返回初始界面" in output


def test_main_uses_gpu_default_model_and_default_language(monkeypatch, tmp_path) -> None:
    input_audio = tmp_path / "sample.wav"
    write_wav(input_audio)
    captured: list[object] = []
    answers = iter([str(input_audio), "", "", "", "", "q"])

    monkeypatch.setattr(sys, "argv", ["run.py"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr(launcher, "probe_media_duration_seconds", lambda _: 1.0)
    monkeypatch.setattr(launcher, "run_pipeline", lambda config: captured.append(config))

    run.main()

    config = captured[0]
    assert config.device == "cuda"
    assert config.model == "large-v3"
    assert config.language == "en"
    assert config.compute_type == "float16"


def test_interactive_launcher_loops_back_for_the_next_job(
    monkeypatch, tmp_path, capsys
) -> None:
    first_audio = tmp_path / "first.wav"
    second_audio = tmp_path / "second.wav"
    write_wav(first_audio)
    write_wav(second_audio)
    captured: list[object] = []
    answers = iter(
        [
            str(first_audio),
            "",
            "2",
            "1",
            "1",
            str(second_audio),
            "",
            "2",
            "1",
            "1",
            "q",
        ]
    )

    monkeypatch.setattr(launcher, "probe_media_duration_seconds", lambda _: 1.0)
    monkeypatch.setattr(launcher, "run_pipeline", lambda config: captured.append(config))

    launcher.run_interactive_launcher(
        launcher.build_parser().parse_args([]),
        input_fn=lambda prompt="": next(answers),
        pipeline_runner=launcher.run_pipeline,
    )

    assert [config.input_audio for config in captured] == [
        first_audio.resolve(),
        second_audio.resolve(),
    ]
    output = capsys.readouterr().out
    assert output.count("atc-stt") >= 2
    assert "本轮任务已完成，返回初始界面。输入 q 可退出。" in output
    assert "已退出 ATC 转写工具。" in output


def test_main_accepts_custom_output_dir_in_interactive_mode(monkeypatch, tmp_path) -> None:
    input_audio = tmp_path / "sample.wav"
    custom_output = tmp_path / "custom output"
    write_wav(input_audio)
    captured: list[object] = []
    answers = iter([str(input_audio), f'\ufeff"{custom_output}"', "", "", "", "q"])

    monkeypatch.setattr(sys, "argv", ["run.py"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr(launcher, "probe_media_duration_seconds", lambda _: 1.0)
    monkeypatch.setattr(launcher, "run_pipeline", lambda config: captured.append(config))

    run.main()

    config = captured[0]
    assert config.output_dir == custom_output.resolve()


def test_resolve_input_path_handles_bom_and_quotes(tmp_path) -> None:
    input_audio = tmp_path / "sample.wav"
    write_wav(input_audio)
    ui = launcher.LauncherUI(
        no_color=True,
        input_fn=lambda prompt="": f'\ufeff"{input_audio}"',
    )

    resolved = launcher.resolve_input_path(None, ui=ui)

    assert resolved == input_audio.resolve()


def test_interactive_launcher_uses_defaults_and_exits_cleanly_when_input_is_exhausted(
    monkeypatch, tmp_path, capsys
) -> None:
    input_audio = tmp_path / "sample.wav"
    write_wav(input_audio)
    captured: list[object] = []
    answers = iter([str(input_audio)])

    monkeypatch.setattr(launcher, "probe_media_duration_seconds", lambda _: 1.0)
    monkeypatch.setattr(launcher, "run_pipeline", lambda config: captured.append(config))

    launcher.run_interactive_launcher(
        launcher.build_parser().parse_args([]),
        input_fn=lambda prompt="": next(answers),
        pipeline_runner=launcher.run_pipeline,
    )

    assert len(captured) == 1
    config = captured[0]
    assert config.input_audio == input_audio.resolve()
    assert config.output_dir == (ROOT / "outputs" / input_audio.stem).resolve()
    assert config.device == "cuda"
    assert config.model == "large-v3"
    assert config.language == "en"
    assert config.compute_type == "float16"
    output = capsys.readouterr().out
    assert "本轮任务已完成，返回初始界面。输入 q 可退出。" in output
    assert "已退出 ATC 转写工具。" in output


def test_main_processes_directory_recursively(monkeypatch, tmp_path, capsys) -> None:
    input_dir = tmp_path / "input_root"
    first_audio = input_dir / "day1" / "a.wav"
    second_audio = input_dir / "day2" / "sub" / "b.mp3"
    ignored_file = input_dir / "day2" / "notes.txt"
    write_wav(first_audio)
    second_audio.parent.mkdir(parents=True, exist_ok=True)
    second_audio.write_bytes(b"fake mp3")
    ignored_file.parent.mkdir(parents=True, exist_ok=True)
    ignored_file.write_text("ignore me", encoding="utf-8")

    captured: list[object] = []
    answers = iter(["all", "y"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            str(input_dir),
            "--device",
            "cpu",
            "--model",
            "tiny.en",
            "--language",
            "auto",
        ],
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr(
        launcher,
        "probe_media_duration_seconds",
        lambda path: 60.0 if path.suffix.lower() == ".wav" else None,
    )
    monkeypatch.setattr(launcher, "run_pipeline", lambda config: captured.append(config))

    run.main()

    assert len(captured) == 2
    assert [config.input_audio for config in captured] == [
        first_audio.resolve(),
        second_audio.resolve(),
    ]
    assert [config.output_dir for config in captured] == [
        (ROOT / "outputs" / input_dir.name / "day1" / "a.wav").resolve(),
        (ROOT / "outputs" / input_dir.name / "day2" / "sub" / "b.mp3").resolve(),
    ]
    assert all(config.language is None for config in captured)

    output = capsys.readouterr().out
    assert "扫描摘要" in output
    assert "可处理音频数" in output
    assert "忽略文件数" in output
    assert "选择结果" in output
    assert "文件进度: 1/2" in output
    assert "文件进度: 2/2" in output
    assert "批处理总结" in output


def test_run_launcher_dry_run_only_scans_and_reports_skip(monkeypatch, tmp_path, capsys) -> None:
    input_dir = tmp_path / "input_root"
    first_audio = input_dir / "a.wav"
    second_audio = input_dir / "b.wav"
    write_wav(first_audio)
    write_wav(second_audio)
    output_root = tmp_path / "planned_outputs"
    completed_output = output_root / "a.wav"
    completed_output.mkdir(parents=True, exist_ok=True)
    (completed_output / "transcript.json").write_text("{}", encoding="utf-8")

    args = launcher.build_parser().parse_args(
        [
            str(input_dir),
            "--dry-run",
            "--yes",
            "--select",
            "all",
            "--output-dir",
            str(output_root),
        ]
    )

    monkeypatch.setattr(launcher, "probe_media_duration_seconds", lambda _: 30.0)
    monkeypatch.setattr(
        launcher,
        "run_pipeline",
        lambda config: (_ for _ in ()).throw(AssertionError("dry run should not execute")),
    )

    launcher.run_launcher(args)

    output = capsys.readouterr().out
    assert "Dry run 完成" in output
    assert "Dry run 将执行" in output
    assert "跳过文件" in output
    assert "a.wav" in output
    assert "b.wav" in output


def test_select_entries_supports_filters() -> None:
    entries = (
        launcher.AudioScanEntry(
            index=1,
            audio_path=Path("day1/a.wav"),
            display_path=Path("day1/a.wav"),
            output_dir=Path("out/a.wav"),
            duration_seconds=1.0,
            state="new",
            state_reason="待处理",
        ),
        launcher.AudioScanEntry(
            index=2,
            audio_path=Path("day1/b.mp3"),
            display_path=Path("day1/b.mp3"),
            output_dir=Path("out/b.mp3"),
            duration_seconds=1.0,
            state="completed",
            state_reason="已有结果",
        ),
        launcher.AudioScanEntry(
            index=3,
            audio_path=Path("retry/c.wav"),
            display_path=Path("retry/c.wav"),
            output_dir=Path("out/c.wav"),
            duration_seconds=1.0,
            state="failed",
            state_reason="失败待重试",
        ),
    )

    assert [entry.index for entry in launcher.select_entries(entries, "1,3").entries] == [1, 3]
    assert [entry.index for entry in launcher.select_entries(entries, "only new").entries] == [1]
    assert [entry.index for entry in launcher.select_entries(entries, "failed only").entries] == [3]
    assert [entry.index for entry in launcher.select_entries(entries, "ext:mp3,wav").entries] == [1, 2, 3]
    assert [entry.index for entry in launcher.select_entries(entries, "name:retry").entries] == [3]


def test_process_audio_files_reuses_loaded_model_for_batch(monkeypatch, tmp_path, capsys) -> None:
    first_audio = tmp_path / "a.wav"
    second_audio = tmp_path / "b.wav"
    write_wav(first_audio)
    write_wav(second_audio)
    output_root = tmp_path / "outputs"
    shared_model = object()
    loaded = []
    seen = []

    entries = (
        launcher.AudioScanEntry(
            index=1,
            audio_path=first_audio.resolve(),
            display_path=Path("a.wav"),
            output_dir=(output_root / "a.wav").resolve(),
            duration_seconds=1.0,
            state="new",
            state_reason="待处理",
        ),
        launcher.AudioScanEntry(
            index=2,
            audio_path=second_audio.resolve(),
            display_path=Path("b.wav"),
            output_dir=(output_root / "b.wav").resolve(),
            duration_seconds=1.0,
            state="new",
            state_reason="待处理",
        ),
    )

    def fake_load_whisper_model(config):
        loaded.append(config.input_audio)
        return shared_model

    def fake_run_pipeline(config, model=None):
        seen.append((config.input_audio, model))
        print(f"processed: {config.input_audio.name}")
        return {"text": config.input_audio.name}

    monkeypatch.setattr("atc_asr.launcher.load_whisper_model", fake_load_whisper_model)
    monkeypatch.setattr("atc_asr.launcher.run_pipeline", fake_run_pipeline)

    launcher.process_audio_files(
        selected_entries=entries,
        model="large-v3",
        device="cuda",
        compute_type="float16",
        language="en",
        args=Namespace(chunk_minutes=20, beam_size=5, overwrite=False, split_only=False),
        ui=launcher.LauncherUI(no_color=True, input_fn=lambda prompt="": ""),
        pipeline_runner=launcher.run_pipeline,
    )

    assert loaded == [first_audio.resolve()]
    assert seen == [
        (first_audio.resolve(), shared_model),
        (second_audio.resolve(), shared_model),
    ]
    output = capsys.readouterr().out
    assert "预加载模型" in output


def test_run_launcher_skips_completed_files_without_overwrite(
    monkeypatch, tmp_path, capsys
) -> None:
    input_dir = tmp_path / "input_root"
    first_audio = input_dir / "done.wav"
    second_audio = input_dir / "new.wav"
    write_wav(first_audio)
    write_wav(second_audio)
    output_root = tmp_path / "batch_outputs"
    done_output = output_root / "done.wav"
    done_output.mkdir(parents=True, exist_ok=True)
    (done_output / "transcript.json").write_text("{}", encoding="utf-8")

    args = launcher.build_parser().parse_args(
        [
            str(input_dir),
            "--yes",
            "--output-dir",
            str(output_root),
            "--device",
            "cpu",
            "--model",
            "tiny.en",
            "--language",
            "auto",
        ]
    )

    captured: list[object] = []
    monkeypatch.setattr(launcher, "probe_media_duration_seconds", lambda _: 10.0)
    monkeypatch.setattr(launcher, "run_pipeline", lambda config: captured.append(config))

    launcher.run_launcher(args)

    assert [config.input_audio for config in captured] == [second_audio.resolve()]
    output = capsys.readouterr().out
    assert "将跳过 1 个已有结果的文件" in output
    assert "跳过文件" in output
    assert "done.wav" in output
