from __future__ import annotations

import argparse
import inspect
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import imageio_ffmpeg
from rich import box
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from atc_asr.console import configure_console_for_utf8
from atc_asr.pipeline import (
    PipelineConfig,
    cleanup_spawned_processes,
    format_duration,
    load_whisper_model,
    run_pipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
BATCH_FAILURE_MARKER = ".atc_asr_failed.txt"
SCAN_PREVIEW_LIMIT = 12
SELECTION_HELP = "all | 1,3,5-8 | only new | failed only | ext:mp3,wav | name:keyword"
EXIT_COMMANDS = {"q", "quit", "exit"}
LEADING_INPUT_ARTIFACTS = ("\ufeff", "\ufffe", "\u200b", "\u200e", "\u200f", "\u2060")
MOJIBAKE_BOM_PREFIXES = ("ï»¿", "ÿþ", "þÿ")
STARTUP_BANNER_LINES = (
    " █████╗ ████████╗ ██████╗     ███████╗████████╗████████╗",
    "██╔══██╗╚══██╔══╝██╔════╝     ██╔════╝╚══██╔══╝╚══██╔══╝",
    "███████║   ██║   ██║          ███████╗   ██║      ██║   ",
    "██╔══██║   ██║   ██║          ╚════██║   ██║      ██║   ",
    "██║  ██║   ██║   ╚██████╗     ███████║   ██║      ██║   ",
    "╚═╝  ╚═╝   ╚═╝    ╚═════╝     ╚══════╝   ╚═╝      ╚═╝   ",
)


def format_startup_path(path: Path) -> str:
    return path.resolve().as_posix()


def format_prompt_path(path: Path) -> str:
    resolved = path.resolve()
    home = Path.home().resolve()
    try:
        relative = resolved.relative_to(home)
    except ValueError:
        return resolved.name or resolved.drive or "/"
    if not relative.parts:
        return "~"
    return f"~/{relative.as_posix()}"


def make_startup_session_id(now: datetime | None = None) -> str:
    current = now or datetime.now()
    return f"{current.strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"


def build_startup_usage_lines() -> tuple[str, ...]:
    return (
        "用途    : 本地 ATC 音频转写，支持单文件和文件夹递归批处理",
        "输入    : 可直接拖入单个音频/视频文件，或拖入整个文件夹后回车",
        f"筛选    : 文件夹模式支持 {SELECTION_HELP}",
        "默认    : 直接回车可使用默认值，适合常规 GPU 转写流程",
        "重跑    : 已有结果默认跳过；需要强制重跑请使用 --overwrite",
        "模式    : --split-only 只切片，--dry-run 只扫描和预览",
        "脚本化  : 需要固定参数或自动化运行时，请使用 atc-asr-cli",
    )


@dataclass(frozen=True, slots=True)
class MenuOption:
    key: str
    value: str
    label: str
    description: str


@dataclass(frozen=True, slots=True)
class AudioScanEntry:
    index: int
    audio_path: Path
    display_path: Path
    output_dir: Path
    duration_seconds: float | None
    state: str
    state_reason: str


@dataclass(frozen=True, slots=True)
class ScanSummary:
    input_path: Path
    output_root: Path
    total_files: int
    audio_entries: tuple[AudioScanEntry, ...]
    ignored_files: tuple[Path, ...]

    @property
    def audio_count(self) -> int:
        return len(self.audio_entries)

    @property
    def ignored_count(self) -> int:
        return len(self.ignored_files)

    @property
    def total_duration_seconds(self) -> float:
        return sum(entry.duration_seconds or 0.0 for entry in self.audio_entries)

    def count_by_state(self, state: str) -> int:
        return sum(1 for entry in self.audio_entries if entry.state == state)


@dataclass(frozen=True, slots=True)
class SelectionResult:
    expression: str
    entries: tuple[AudioScanEntry, ...]


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    selected_entries: tuple[AudioScanEntry, ...]
    runnable_entries: tuple[AudioScanEntry, ...]
    skipped_entries: tuple[tuple[AudioScanEntry, str], ...]


@dataclass(slots=True)
class BatchExecutionSummary:
    succeeded: list[AudioScanEntry]
    failed: list[tuple[AudioScanEntry, str]]
    skipped: list[tuple[AudioScanEntry, str]]


class ExitInteractiveLauncher(Exception):
    """Raised when the interactive launcher should terminate."""


class PromptInputClosed(Exception):
    """Raised when prompt input is no longer available."""


DEVICE_OPTIONS = (
    MenuOption(
        key="1",
        value="cuda",
        label="GPU (CUDA)",
        description="推荐正式转写使用，速度更快，默认使用 float16。",
    ),
    MenuOption(
        key="2",
        value="cpu",
        label="CPU",
        description="适合没有 CUDA 的机器，默认使用 int8。",
    ),
)

MODEL_OPTIONS = (
    MenuOption(
        key="1",
        value="tiny.en",
        label="tiny.en",
        description="速度最快，适合 CPU 快速试跑。",
    ),
    MenuOption(
        key="2",
        value="base.en",
        label="base.en",
        description="比 tiny 更稳一些，速度仍然较快。",
    ),
    MenuOption(
        key="3",
        value="small.en",
        label="small.en",
        description="准确率和速度较平衡。",
    ),
    MenuOption(
        key="4",
        value="medium.en",
        label="medium.en",
        description="准确率更高，建议在 GPU 上使用。",
    ),
    MenuOption(
        key="5",
        value="large-v3",
        label="large-v3",
        description="精度最高，推荐 GPU 正式转写使用。",
    ),
)

LANGUAGE_OPTIONS = (
    MenuOption(
        key="1",
        value="en",
        label="英语 (en)",
        description="默认选项，适合 ATC 英语通话。",
    ),
    MenuOption(
        key="2",
        value="zh",
        label="中文 (zh)",
        description="适合中文语音。",
    ),
    MenuOption(
        key="3",
        value="auto",
        label="自动检测",
        description="让模型先自动判断语言。",
    ),
)

SUPPORTED_AUDIO_EXTENSIONS = {
    ".aac",
    ".avi",
    ".flac",
    ".m4a",
    ".m4b",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
    ".wma",
}

STATE_LABELS = {
    "new": "待处理",
    "completed": "已有结果",
    "failed": "失败待重试",
}

STATE_STYLES = {
    "new": "green",
    "completed": "yellow",
    "failed": "red",
}


class LauncherUI:
    def __init__(
        self,
        *,
        no_color: bool = False,
        input_fn: Callable[[str], str] | None = None,
    ) -> None:
        redirected = not getattr(sys.stdout, "isatty", lambda: False)()
        self.plain_text = no_color or redirected
        self.console = Console(
            no_color=self.plain_text,
            force_terminal=not self.plain_text,
            highlight=False,
            soft_wrap=True,
        )
        self.input_fn = input_fn
        self.table_box = box.ASCII if self.plain_text else box.ROUNDED

    def print(self, message: object = "", *, style: str | None = None) -> None:
        self.console.print(message, style=style)

    def blank(self) -> None:
        self.console.print()

    def info(self, message: str) -> None:
        self.print(message, style="blue")

    def success(self, message: str) -> None:
        self.print(message, style="green")

    def warning(self, message: str) -> None:
        self.print(message, style="yellow")

    def error(self, message: str) -> None:
        self.print(message, style="red")

    def make_table(self, title: str) -> Table:
        return Table(
            title=title,
            box=self.table_box,
            header_style="bold blue",
            show_lines=False,
            expand=False,
            pad_edge=False,
        )

    def ask(self, prompt: str, *, default: str | None = None) -> str:
        if self.input_fn is None:
            try:
                return Prompt.ask(
                    prompt,
                    console=self.console,
                    default=default,
                    show_default=default is not None,
                )
            except (EOFError, KeyboardInterrupt) as exc:
                if default is not None:
                    return default
                raise PromptInputClosed from exc

        default_hint = f" [{default}]" if default is not None else ""
        try:
            raw_value = self.input_fn(f"{prompt}{default_hint}: ")
        except (EOFError, KeyboardInterrupt, StopIteration) as exc:
            if default is not None:
                return default
            raise PromptInputClosed from exc
        raw = normalize_prompt_text(raw_value)
        if raw:
            return raw
        return default or ""

    def print_startup_banner(self) -> None:
        cwd = Path.cwd()
        username = os.environ.get("USERNAME") or os.environ.get("USER") or "user"
        hostname = os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "host"
        session_id = make_startup_session_id()

        self.blank()
        self.print(f"{username}@{hostname}:{format_prompt_path(cwd)}$ atc-stt", style="dim")
        for line in STARTUP_BANNER_LINES:
            self.print(f"  {line}", style="bold cyan")
        self.print(f"会话: {session_id}", style="dim")
        self.print(f"目录: {format_startup_path(cwd)}", style="dim")
        for line in build_startup_usage_lines():
            self.print(line, style="dim")

    def confirm(self, prompt: str, *, default: bool = True) -> bool:
        if self.input_fn is None:
            try:
                return Confirm.ask(
                    prompt,
                    console=self.console,
                    default=default,
                    show_default=True,
                )
            except (EOFError, KeyboardInterrupt):
                return default

        default_hint = "Y/n" if default else "y/N"
        while True:
            try:
                raw_value = self.input_fn(f"{prompt} [{default_hint}]: ")
            except (EOFError, KeyboardInterrupt, StopIteration):
                return default
            raw = normalize_prompt_text(raw_value).lower()
            if not raw:
                return default
            if raw in {"y", "yes", "1"}:
                return True
            if raw in {"n", "no", "0"}:
                return False
            self.warning("请输入 yes 或 no。")

    def print_scan_summary(
        self,
        summary: ScanSummary,
        *,
        split_only: bool,
        overwrite: bool,
    ) -> None:
        mode = "文件夹递归" if summary.input_path.is_dir() else "单文件"
        unknown_durations = sum(
            1 for entry in summary.audio_entries if entry.duration_seconds is None
        )
        duration_text = format_duration(summary.total_duration_seconds)
        if unknown_durations:
            duration_text = f"{duration_text}（另有 {unknown_durations} 个文件时长未探测到）"

        table = self.make_table("扫描摘要")
        table.add_column("项目", style="blue")
        table.add_column("值")
        table.add_row("输入路径", str(summary.input_path))
        table.add_row("处理模式", mode)
        table.add_row("总文件数", str(summary.total_files))
        table.add_row("可处理音频数", str(summary.audio_count))
        table.add_row("忽略文件数", str(summary.ignored_count))
        table.add_row("总时长", duration_text)
        table.add_row("输出目录", str(summary.output_root))
        table.add_row("新文件", str(summary.count_by_state("new")))
        table.add_row("已有结果", str(summary.count_by_state("completed")))
        table.add_row("失败待重试", str(summary.count_by_state("failed")))
        self.blank()
        self.print(table)

        preview = self.make_table("文件预览")
        preview.add_column("#", justify="right", style="blue")
        preview.add_column("状态")
        preview.add_column("时长", justify="right")
        preview.add_column("文件")
        for entry in summary.audio_entries[:SCAN_PREVIEW_LIMIT]:
            preview.add_row(
                str(entry.index),
                Text(STATE_LABELS[entry.state], style=STATE_STYLES[entry.state]),
                format_entry_duration(entry),
                str(entry.display_path),
            )
        self.print(preview)

        remaining = summary.audio_count - min(summary.audio_count, SCAN_PREVIEW_LIMIT)
        if remaining > 0:
            self.info(f"还有 {remaining} 个文件未在预览中展开。")

        if summary.count_by_state("completed"):
            if overwrite:
                self.warning(
                    "检测到已有结果文件，但已启用 --overwrite，本次会重新生成这些结果。"
                )
            else:
                target_name = "切片结果" if split_only else "转写结果"
                self.warning(
                    f"检测到已有{target_name}，默认会跳过这些文件；如需重跑请使用 --overwrite。"
                )

        if summary.count_by_state("failed"):
            self.warning(
                "检测到失败或未完成的旧结果，可用 failed only 只重跑这些文件。"
            )

    def print_selection_help(self) -> None:
        self.info(f"选择语法: {SELECTION_HELP}")

    def print_selection_summary(self, selection: SelectionResult, plan: ExecutionPlan) -> None:
        table = self.make_table("选择结果")
        table.add_column("项目", style="blue")
        table.add_column("值")
        table.add_row("选择表达式", selection.expression)
        table.add_row("已选择文件", str(len(selection.entries)))
        table.add_row("将执行", str(len(plan.runnable_entries)))
        table.add_row("将跳过", str(len(plan.skipped_entries)))
        self.blank()
        self.print(table)

        preview = self.make_table("已选择文件预览")
        preview.add_column("#", justify="right", style="blue")
        preview.add_column("状态")
        preview.add_column("文件")
        for entry in selection.entries[:SCAN_PREVIEW_LIMIT]:
            preview.add_row(
                str(entry.index),
                Text(STATE_LABELS[entry.state], style=STATE_STYLES[entry.state]),
                str(entry.display_path),
            )
        self.print(preview)

        if len(selection.entries) > SCAN_PREVIEW_LIMIT:
            self.info(f"已选择文件过多，仅预览前 {SCAN_PREVIEW_LIMIT} 项。")

    def print_execution_plan(
        self,
        *,
        input_path: Path,
        output_root: Path,
        device: str,
        model: str,
        language: str | None,
        compute_type: str,
        plan: ExecutionPlan,
        args: argparse.Namespace,
    ) -> None:
        mode = "文件夹递归" if input_path.is_dir() else "单文件"
        table = self.make_table("执行计划")
        table.add_column("项目", style="blue")
        table.add_column("值")
        table.add_row("输入路径", str(input_path))
        table.add_row("处理模式", mode)
        table.add_row("输出目录", str(output_root))
        table.add_row("已选择文件", str(len(plan.selected_entries)))
        table.add_row("即将执行", str(len(plan.runnable_entries)))
        table.add_row("将跳过", str(len(plan.skipped_entries)))
        table.add_row("运行设备", device)
        table.add_row("转写模型", model)
        table.add_row("识别语言", format_language(language))
        table.add_row("计算精度", compute_type)
        table.add_row("仅切片", "是" if args.split_only else "否")
        table.add_row("覆盖已有结果", "是" if args.overwrite else "否")
        self.blank()
        self.print(table)

    def print_dry_run_summary(self, plan: ExecutionPlan) -> None:
        self.success("Dry run 完成：未执行切片或转写，仅输出扫描和选择结果。")
        self.print_execution_summary(
            BatchExecutionSummary(
                succeeded=[],
                failed=[],
                skipped=list(plan.skipped_entries),
            ),
            would_run=list(plan.runnable_entries),
        )

    def print_processing_header(
        self,
        *,
        position: int,
        total: int,
        entry: AudioScanEntry,
    ) -> None:
        self.blank()
        self.info(f"文件进度: {position}/{total}")
        self.print(Text.assemble(("当前文件: ", "blue"), (str(entry.audio_path), "cyan")))
        self.print(Text.assemble(("输出目录: ", "blue"), (str(entry.output_dir), "cyan")))

    def print_execution_summary(
        self,
        summary: BatchExecutionSummary,
        *,
        would_run: list[AudioScanEntry] | None = None,
    ) -> None:
        table = self.make_table("批处理总结")
        table.add_column("项目", style="blue")
        table.add_column("数量", justify="right")
        table.add_row("成功", str(len(summary.succeeded)))
        table.add_row("失败", str(len(summary.failed)))
        table.add_row("跳过", str(len(summary.skipped)))
        if would_run is not None:
            table.add_row("Dry run 将执行", str(len(would_run)))
        self.blank()
        self.print(table)

        if would_run:
            self.success("Dry run 将执行以下文件：")
            for entry in would_run:
                self.print(f"- {entry.display_path}")

        if summary.succeeded:
            self.success("成功文件：")
            for entry in summary.succeeded:
                self.print(f"- {entry.display_path}")

        if summary.failed:
            self.error("失败文件：")
            for entry, message in summary.failed:
                self.print(f"- {entry.display_path}")
                self.print(f"  原因: {message}")

        if summary.skipped:
            self.warning("跳过文件：")
            for entry, reason in summary.skipped:
                self.print(f"- {entry.display_path}")
                self.print(f"  原因: {reason}")


def default_output_dir(input_audio: Path) -> Path:
    return OUTPUTS_ROOT / input_audio.stem


def default_batch_output_root(input_dir: Path) -> Path:
    return OUTPUTS_ROOT / input_dir.name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="交互式启动 ATC 语音转写流程。")
    parser.add_argument("input_path", nargs="?", type=Path, help="音频文件或文件夹路径")
    parser.add_argument("--model", default=None, help="模型名称；不传时会交互选择")
    parser.add_argument("--device", default=None, help="运行设备：cuda/gpu 或 cpu")
    parser.add_argument(
        "--compute-type",
        default=None,
        help="计算精度；不传时会根据设备自动选择",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="识别语言代码，默认 en，也支持 auto 自动检测",
    )
    parser.add_argument(
        "--chunk-minutes",
        type=int,
        default=20,
        help="切段时长（分钟），默认 20",
    )
    parser.add_argument("--beam-size", type=int, default=5, help="beam size，默认 5")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录；单文件模式下是输出目录，文件夹模式下是输出根目录",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有切段和转写结果",
    )
    parser.add_argument(
        "--split-only",
        action="store_true",
        help="只切段，不转写",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只扫描和预览，不真正执行切片或转写",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="跳过目录确认，按默认或 --select 结果直接执行",
    )
    parser.add_argument(
        "--select",
        default=None,
        help="目录模式的选择表达式，例如 all、1,3,5-8、only new、failed only、ext:mp3,wav、name:keyword",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="禁用彩色输出；输出被重定向时也会自动降级为纯文本",
    )
    return parser


def normalize_prompt_text(raw: object | None) -> str:
    text = "" if raw is None else str(raw)
    text = text.replace("\x00", "").strip()

    changed = True
    while changed and text:
        changed = False
        for prefix in (*LEADING_INPUT_ARTIFACTS, *MOJIBAKE_BOM_PREFIXES):
            if text.startswith(prefix):
                text = text[len(prefix) :].lstrip()
                changed = True
    return text


def strip_wrapping_quotes(raw: str) -> str:
    text = raw.strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    for quote in ('"', "'"):
        if text.startswith(quote):
            text = text[1:].lstrip()
        if text.endswith(quote):
            text = text[:-1].rstrip()
    return text


def normalize_path_text(raw: object | None) -> str:
    return strip_wrapping_quotes(normalize_prompt_text(raw))


def resolve_path_value(raw: Path | str, *, field_name: str) -> Path:
    normalized = normalize_path_text(raw)
    if not normalized:
        raise SystemExit(f"{field_name}不能为空。")
    return Path(normalized).expanduser().resolve()


def should_exit_prompt(raw: str) -> bool:
    return normalize_path_text(raw).lower() in EXIT_COMMANDS


def normalize_device(raw: str) -> str:
    normalized = raw.strip().lower()
    if normalized in {"1", "cuda", "gpu"}:
        return "cuda"
    if normalized in {"2", "cpu"}:
        return "cpu"
    raise ValueError(f"不支持的设备选项：{raw}")


def normalize_language(raw: str | None) -> str | None:
    if raw is None:
        return None

    normalized = raw.strip().lower()
    if not normalized:
        return None
    if normalized in {"1", "en", "english"}:
        return "en"
    if normalized in {"2", "zh", "zh-cn", "cn", "chinese"}:
        return "zh"
    if normalized in {"3", "auto", "detect"}:
        return None
    return raw.strip()


def format_language(language: str | None) -> str:
    return language or "自动检测"


def print_menu(title: str, options: tuple[MenuOption, ...], *, ui: LauncherUI) -> None:
    ui.print(title, style="blue")
    for option in options:
        ui.print(f"{option.key}. {option.label} - {option.description}")


def choose_menu_option(
    options: tuple[MenuOption, ...],
    prompt: str,
    default_key: str,
    *,
    ui: LauncherUI,
    allow_custom_value: bool = False,
) -> str:
    by_key = {option.key.lower(): option.value for option in options}
    by_value = {option.value.lower(): option.value for option in options}

    while True:
        raw = ui.ask(prompt, default=default_key).strip()
        if not raw:
            raw = default_key

        normalized = raw.lower()
        if normalized in by_key:
            return by_key[normalized]
        if normalized in by_value:
            return by_value[normalized]
        if allow_custom_value and raw:
            return raw

        if allow_custom_value:
            ui.warning("请输入序号，或直接输入对应值。")
            continue
        ui.warning("请输入有效的选项。")


def resolve_device(
    device: str | None,
    *,
    ui: LauncherUI,
) -> str:
    if device is not None:
        return normalize_device(device)

    print_menu("请选择运行设备：", DEVICE_OPTIONS, ui=ui)
    return choose_menu_option(
        DEVICE_OPTIONS,
        prompt="输入序号",
        default_key="1",
        ui=ui,
    )


def resolve_model(
    model: str | None,
    device: str,
    *,
    ui: LauncherUI,
) -> str:
    if model is not None:
        return model.strip()

    default_key = "5" if device == "cuda" else "1"
    ui.blank()
    print_menu("请选择转写模型：", MODEL_OPTIONS, ui=ui)
    return choose_menu_option(
        MODEL_OPTIONS,
        prompt="输入序号或模型名",
        default_key=default_key,
        allow_custom_value=True,
        ui=ui,
    )


def resolve_language(
    language: str | None,
    *,
    ui: LauncherUI,
) -> str | None:
    normalized_language = normalize_language(language)
    if language is not None:
        return normalized_language

    ui.blank()
    print_menu("请选择识别语言：", LANGUAGE_OPTIONS, ui=ui)
    chosen = choose_menu_option(
        LANGUAGE_OPTIONS,
        prompt="输入序号或语言代码",
        default_key="1",
        allow_custom_value=True,
        ui=ui,
    )
    return normalize_language(chosen)


def resolve_compute_type(compute_type: str | None, device: str) -> str:
    if compute_type:
        return compute_type
    return "float16" if device == "cuda" else "int8"


def resolve_input_path(
    input_path: Path | None,
    *,
    ui: LauncherUI,
) -> Path:
    if input_path is not None:
        return resolve_path_value(input_path, field_name="输入路径")

    ui.blank()
    try:
        raw = ui.ask("请输入音频文件或文件夹路径，也可以把路径拖到这个窗口后回车；输入 q 退出")
    except PromptInputClosed:
        raise ExitInteractiveLauncher
    if should_exit_prompt(raw):
        raise ExitInteractiveLauncher

    normalized = normalize_path_text(raw)
    if not normalized:
        raise SystemExit("未提供输入路径。")
    return Path(normalized).expanduser().resolve()


def collect_audio_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path.resolve()]

    if not input_path.is_dir():
        raise SystemExit(f"输入路径不存在，或不是文件/文件夹：{input_path}")

    audio_files = sorted(
        path.resolve()
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )
    if not audio_files:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise SystemExit(
            f"在文件夹中没有找到可处理的音频文件：{input_path}\n支持的扩展名：{supported}"
        )
    return audio_files


def resolve_output_root(input_path: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return resolve_path_value(output_dir, field_name="输出目录")
    if input_path.is_dir():
        return default_batch_output_root(input_path).resolve()
    return default_output_dir(input_path).resolve()


def prompt_output_root(input_path: Path, *, ui: LauncherUI) -> Path:
    default_output_root = resolve_output_root(input_path, None)
    ui.blank()
    raw = ui.ask(
        "请输入输出目录；直接回车使用默认值",
        default=str(default_output_root),
    )
    normalized = normalize_path_text(raw)
    if not normalized:
        return default_output_root
    return Path(normalized).expanduser().resolve()


def output_dir_for_audio(audio_path: Path, input_path: Path, output_root: Path) -> Path:
    if input_path.is_file():
        return output_root.resolve()

    relative_parent = audio_path.parent.relative_to(input_path)
    return (output_root / relative_parent / audio_path.name).resolve()


def result_artifact_path(output_dir: Path, split_only: bool) -> Path:
    return output_dir / ("split_only.json" if split_only else "transcript.json")


def failure_marker_path(output_dir: Path) -> Path:
    return output_dir / BATCH_FAILURE_MARKER


def has_partial_output(output_dir: Path) -> bool:
    if not output_dir.exists():
        return False

    checks = (
        output_dir / "chunks",
        output_dir / "transcripts",
        output_dir / "chunk_manifest.json",
        output_dir / "split_only.json",
    )
    for path in checks:
        if path.is_file():
            return True
        if path.is_dir() and any(path.iterdir()):
            return True
    return False


def detect_entry_state(output_dir: Path, *, split_only: bool) -> tuple[str, str]:
    result_path = result_artifact_path(output_dir, split_only)
    if result_path.exists():
        return "completed", f"已存在结果文件：{result_path.name}"

    failed_path = failure_marker_path(output_dir)
    if failed_path.exists():
        return "failed", f"检测到失败标记：{failed_path.name}"

    if has_partial_output(output_dir):
        return "failed", "检测到未完成的中间结果"

    return "new", "待处理"


def probe_media_duration_seconds(path: Path) -> float | None:
    try:
        _frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(path))
    except Exception:
        return None

    try:
        return max(float(seconds), 0.0)
    except (TypeError, ValueError):
        return None


def format_entry_duration(entry: AudioScanEntry) -> str:
    if entry.duration_seconds is None:
        return "未知"
    return format_duration(entry.duration_seconds)


def scan_input_path(
    input_path: Path,
    *,
    output_root: Path,
    split_only: bool,
) -> ScanSummary:
    if input_path.is_file():
        output_dir = output_dir_for_audio(input_path, input_path, output_root)
        state, reason = detect_entry_state(output_dir, split_only=split_only)
        entry = AudioScanEntry(
            index=1,
            audio_path=input_path.resolve(),
            display_path=Path(input_path.name),
            output_dir=output_dir,
            duration_seconds=probe_media_duration_seconds(input_path),
            state=state,
            state_reason=reason,
        )
        return ScanSummary(
            input_path=input_path.resolve(),
            output_root=output_root,
            total_files=1,
            audio_entries=(entry,),
            ignored_files=(),
        )

    if not input_path.is_dir():
        raise SystemExit(f"输入路径不存在，或不是文件/文件夹：{input_path}")

    all_files = sorted(path.resolve() for path in input_path.rglob("*") if path.is_file())
    audio_files = [
        path for path in all_files if path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ]
    if not audio_files:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise SystemExit(
            f"在文件夹中没有找到可处理的音频文件：{input_path}\n支持的扩展名：{supported}"
        )

    ignored_files = tuple(path for path in all_files if path not in audio_files)
    entries: list[AudioScanEntry] = []
    for index, audio_path in enumerate(audio_files, start=1):
        output_dir = output_dir_for_audio(audio_path, input_path, output_root)
        state, reason = detect_entry_state(output_dir, split_only=split_only)
        entries.append(
            AudioScanEntry(
                index=index,
                audio_path=audio_path,
                display_path=audio_path.relative_to(input_path),
                output_dir=output_dir,
                duration_seconds=probe_media_duration_seconds(audio_path),
                state=state,
                state_reason=reason,
            )
        )

    return ScanSummary(
        input_path=input_path.resolve(),
        output_root=output_root,
        total_files=len(all_files),
        audio_entries=tuple(entries),
        ignored_files=ignored_files,
    )


def build_config(
    *,
    audio_path: Path,
    output_dir: Path,
    model: str,
    device: str,
    compute_type: str,
    language: str | None,
    args: argparse.Namespace,
) -> PipelineConfig:
    return PipelineConfig(
        input_audio=audio_path,
        output_dir=output_dir,
        model=model,
        device=device,
        compute_type=compute_type,
        language=language,
        chunk_seconds=args.chunk_minutes * 60,
        beam_size=args.beam_size,
        overwrite_chunks=args.overwrite,
        overwrite_transcripts=args.overwrite,
        split_only=args.split_only,
    )


def parse_index_selection(raw: str, entries: tuple[AudioScanEntry, ...]) -> tuple[AudioScanEntry, ...]:
    allowed_chars = set("0123456789,- ")
    if any(char not in allowed_chars for char in raw):
        raise ValueError("序号选择只支持数字、逗号和范围，例如 1,3,5-8。")

    chosen_indexes: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = [item.strip() for item in token.split("-", maxsplit=1)]
            if not start_text or not end_text:
                raise ValueError("范围选择格式不正确，例如 5-8。")
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError("范围选择的起点不能大于终点。")
            chosen_indexes.extend(range(start, end + 1))
            continue
        chosen_indexes.append(int(token))

    if not chosen_indexes:
        raise ValueError("请选择至少一个文件。")

    unique_indexes: list[int] = []
    for index in chosen_indexes:
        if index < 1 or index > len(entries):
            raise ValueError(f"文件序号超出范围：{index}")
        if index not in unique_indexes:
            unique_indexes.append(index)

    by_index = {entry.index: entry for entry in entries}
    return tuple(by_index[index] for index in unique_indexes)


def select_entries(
    entries: tuple[AudioScanEntry, ...],
    expression: str,
) -> SelectionResult:
    normalized = " ".join(expression.strip().lower().split())
    if not normalized or normalized == "all":
        return SelectionResult(expression="all", entries=entries)

    if normalized == "only new":
        selected = tuple(entry for entry in entries if entry.state == "new")
        return SelectionResult(expression="only new", entries=selected)

    if normalized == "failed only":
        selected = tuple(entry for entry in entries if entry.state == "failed")
        return SelectionResult(expression="failed only", entries=selected)

    if normalized.startswith("ext:"):
        raw_extensions = normalized.removeprefix("ext:")
        wanted = {
            (item if item.startswith(".") else f".{item}")
            for item in (part.strip() for part in raw_extensions.split(","))
            if item
        }
        if not wanted:
            raise ValueError("ext: 后面至少要提供一个扩展名。")
        selected = tuple(
            entry for entry in entries if entry.audio_path.suffix.lower() in wanted
        )
        return SelectionResult(expression=expression.strip(), entries=selected)

    if normalized.startswith("name:"):
        keyword = expression.strip()[5:].strip()
        if not keyword:
            raise ValueError("name: 后面至少要提供一个关键词。")
        keyword_lower = keyword.lower()
        selected = tuple(
            entry
            for entry in entries
            if keyword_lower in str(entry.display_path).lower()
            or keyword_lower in entry.audio_path.name.lower()
        )
        return SelectionResult(expression=expression.strip(), entries=selected)

    return SelectionResult(
        expression=expression.strip(),
        entries=parse_index_selection(expression, entries),
    )


def resolve_selection(
    summary: ScanSummary,
    *,
    args: argparse.Namespace,
    ui: LauncherUI,
) -> SelectionResult:
    if summary.input_path.is_file():
        return SelectionResult(expression="single file", entries=summary.audio_entries)

    expression = args.select
    if expression is None and args.yes:
        expression = "all"

    while True:
        if expression is None:
            ui.blank()
            ui.print_selection_help()
            expression = ui.ask("请选择要处理的文件", default="all")

        try:
            selection = select_entries(summary.audio_entries, expression)
        except ValueError as exc:
            if args.select is not None or args.yes:
                raise SystemExit(str(exc)) from exc
            ui.warning(str(exc))
            expression = None
            continue

        if not selection.entries:
            message = "当前选择没有匹配到任何文件，请重新输入。"
            if args.select is not None or args.yes:
                raise SystemExit(message)
            ui.warning(message)
            expression = None
            continue

        return selection


def skip_reason_for_entry(entry: AudioScanEntry, *, args: argparse.Namespace) -> str | None:
    if not args.overwrite and entry.state == "completed":
        result_name = result_artifact_path(entry.output_dir, args.split_only).name
        return f"已存在结果，默认跳过（{result_name}）"
    return None


def build_execution_plan(
    selection: SelectionResult,
    *,
    args: argparse.Namespace,
) -> ExecutionPlan:
    runnable_entries: list[AudioScanEntry] = []
    skipped_entries: list[tuple[AudioScanEntry, str]] = []

    for entry in selection.entries:
        reason = skip_reason_for_entry(entry, args=args)
        if reason is None:
            runnable_entries.append(entry)
            continue
        skipped_entries.append((entry, reason))

    return ExecutionPlan(
        selected_entries=selection.entries,
        runnable_entries=tuple(runnable_entries),
        skipped_entries=tuple(skipped_entries),
    )


def should_continue(
    *,
    input_path: Path,
    plan: ExecutionPlan,
    args: argparse.Namespace,
    ui: LauncherUI,
) -> bool:
    if args.yes or input_path.is_file():
        return True

    if not plan.runnable_entries:
        return True

    if plan.skipped_entries:
        prompt = (
            f"检测到 {len(plan.selected_entries)} 个已选文件，"
            f"其中 {len(plan.skipped_entries)} 个已有结果将跳过，"
            f"是否继续执行 {len(plan.runnable_entries)} 个文件？"
        )
    else:
        prompt = f"检测到 {len(plan.selected_entries)} 个可转写文件，是否继续？"
    return ui.confirm(prompt, default=True)


def write_failure_marker(output_dir: Path, audio_path: Path, message: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    failure_marker_path(output_dir).write_text(
        "\n".join(
            (
                f"source_audio={audio_path}",
                f"failed_at={datetime.now().isoformat(timespec='seconds')}",
                f"error={message}",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def clear_failure_marker(output_dir: Path) -> None:
    marker = failure_marker_path(output_dir)
    if marker.exists():
        marker.unlink()


def supports_shared_model(pipeline_runner: Callable[..., object]) -> bool:
    try:
        parameters = inspect.signature(pipeline_runner).parameters.values()
    except (TypeError, ValueError):
        return False

    for parameter in parameters:
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == "model":
            return True
    return False


def process_audio_files(
    *,
    selected_entries: tuple[AudioScanEntry, ...],
    model: str,
    device: str,
    compute_type: str,
    language: str | None,
    args: argparse.Namespace,
    ui: LauncherUI,
    pipeline_runner=None,
) -> BatchExecutionSummary:
    if pipeline_runner is None:
        pipeline_runner = run_pipeline

    plan = build_execution_plan(
        SelectionResult(expression="runtime", entries=selected_entries),
        args=args,
    )
    summary = BatchExecutionSummary(
        succeeded=[],
        failed=[],
        skipped=list(plan.skipped_entries),
    )
    runnable_entries = list(plan.runnable_entries)
    executor = pipeline_runner

    if selected_entries and plan.skipped_entries:
        ui.warning(
            f"将跳过 {len(plan.skipped_entries)} 个已有结果的文件；如需重跑请使用 --overwrite。"
        )

    if not runnable_entries:
        ui.warning("当前没有需要执行的文件。")
        ui.print_execution_summary(summary)
        return summary

    if len(runnable_entries) > 1 and not args.split_only and supports_shared_model(pipeline_runner):
        first_entry = runnable_entries[0]
        first_config = build_config(
            audio_path=first_entry.audio_path,
            output_dir=first_entry.output_dir,
            model=model,
            device=device,
            compute_type=compute_type,
            language=language,
            args=args,
        )
        ui.info("批量模式：正在预加载模型，后续文件会复用同一个模型。")
        shared_model = load_whisper_model(first_config)

        def executor(config: PipelineConfig):
            return pipeline_runner(config, model=shared_model)

    for position, entry in enumerate(runnable_entries, start=1):
        config = build_config(
            audio_path=entry.audio_path,
            output_dir=entry.output_dir,
            model=model,
            device=device,
            compute_type=compute_type,
            language=language,
            args=args,
        )

        if len(runnable_entries) > 1:
            ui.print_processing_header(
                position=position,
                total=len(runnable_entries),
                entry=entry,
            )

        try:
            executor(config)
        except Exception as exc:
            write_failure_marker(entry.output_dir, entry.audio_path, str(exc))
            summary.failed.append((entry, str(exc)))
            ui.error(f"处理失败: {entry.audio_path}")
            ui.error(f"失败原因: {exc}")
        else:
            clear_failure_marker(entry.output_dir)
            summary.succeeded.append(entry)

    ui.print_execution_summary(summary)
    return summary


def run_launcher(
    args: argparse.Namespace,
    *,
    input_fn: Callable[[str], str] | None = None,
    pipeline_runner=None,
) -> None:
    try:
        if pipeline_runner is None:
            pipeline_runner = run_pipeline

        ui = LauncherUI(no_color=args.no_color, input_fn=input_fn)
        interactive_prompt = args.input_path is None
        if interactive_prompt:
            ui.print_startup_banner()
        input_path = resolve_input_path(args.input_path, ui=ui)
        if not input_path.exists():
            raise SystemExit(f"找不到输入路径：{input_path}")

        output_root = resolve_output_root(input_path, args.output_dir)
        if interactive_prompt and args.output_dir is None:
            output_root = prompt_output_root(input_path, ui=ui)

        summary = scan_input_path(
            input_path,
            output_root=output_root,
            split_only=args.split_only,
        )
        ui.print_scan_summary(
            summary,
            split_only=args.split_only,
            overwrite=args.overwrite,
        )

        selection = resolve_selection(summary, args=args, ui=ui)
        plan = build_execution_plan(selection, args=args)
        ui.print_selection_summary(selection, plan)

        if args.dry_run:
            ui.print_dry_run_summary(plan)
            return

        if not plan.runnable_entries:
            ui.warning("没有需要执行的文件，流程结束。")
            ui.print_execution_summary(
                BatchExecutionSummary(
                    succeeded=[],
                    failed=[],
                    skipped=list(plan.skipped_entries),
                )
            )
            return

        device = resolve_device(args.device, ui=ui)
        model = resolve_model(args.model, device, ui=ui)
        language = resolve_language(args.language, ui=ui)
        compute_type = resolve_compute_type(args.compute_type, device)

        ui.print_execution_plan(
            input_path=input_path,
            output_root=output_root,
            device=device,
            model=model,
            language=language,
            compute_type=compute_type,
            plan=plan,
            args=args,
        )

        if not should_continue(input_path=input_path, plan=plan, args=args, ui=ui):
            raise SystemExit("用户取消执行。")

        execution_summary = process_audio_files(
            selected_entries=plan.selected_entries,
            model=model,
            device=device,
            compute_type=compute_type,
            language=language,
            args=args,
            ui=ui,
            pipeline_runner=pipeline_runner,
        )

        if execution_summary.failed:
            raise SystemExit(f"共有 {len(execution_summary.failed)} 个文件处理失败。")
    finally:
        cleanup_spawned_processes()


def format_system_exit_message(exc: SystemExit) -> str | None:
    code = exc.code
    if code in {None, 0}:
        return None
    if isinstance(code, str):
        return code
    return f"程序异常退出，状态码：{code}"


def should_loop_interactively(args: argparse.Namespace) -> bool:
    return args.input_path is None


def run_interactive_launcher(
    args: argparse.Namespace,
    *,
    input_fn: Callable[[str], str] | None = None,
    pipeline_runner=None,
) -> None:
    loop_ui = LauncherUI(no_color=args.no_color, input_fn=input_fn)

    try:
        while True:
            try:
                run_launcher(args, input_fn=input_fn, pipeline_runner=pipeline_runner)
            except ExitInteractiveLauncher:
                loop_ui.info("已退出 ATC 转写工具。")
                return
            except SystemExit as exc:
                message = format_system_exit_message(exc)
                if message:
                    loop_ui.error(message)
                loop_ui.info("返回初始界面，继续拖入新的文件或文件夹；输入 q 可退出。")
            else:
                loop_ui.success("本轮任务已完成，返回初始界面。输入 q 可退出。")
    finally:
        cleanup_spawned_processes()


def main(argv: Sequence[str] | None = None) -> None:
    configure_console_for_utf8()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if should_loop_interactively(args):
            run_interactive_launcher(args)
            return
        run_launcher(args)
    finally:
        cleanup_spawned_processes()
