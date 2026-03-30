# ATC ASR CLI

这是一个面向本地部署的 ATC 音频转写项目，核心基于 `faster-whisper`，当前只保留 CLI 形态，不再包含 web 端。

这样调整后的目标很明确：

- 更适合后续打包成 Windows `exe`
- 更容易维护和扩展
- 保留现有切段、转写、批处理和断点复用逻辑
- 避免 UI 层和核心转写逻辑耦合

## 当前架构

项目现在按 CLI-only 方式组织：

- `run.py`
  本地启动入口，适合直接双击、命令行运行或后续打包成 `exe`
- `src/atc_asr/launcher.py`
  交互式 CLI 编排层，负责设备/模型/语言选择、输入路径处理、批量任务调度
- `src/atc_asr/cli.py`
  参数式 CLI，适合脚本化调用、自动化任务和集成到外部系统
- `src/atc_asr/pipeline.py`
  核心转写流水线，负责切段、模型下载、转写、结果合并
- `src/atc_asr/console.py`
  Windows 控制台 UTF-8 兼容处理
- `tests/`
  CLI 和流水线测试
- `docs/windows-gpu-notes.md`
  Windows GPU 环境说明
- `ref/`
  参考项目，仅用于对照和研究，不参与当前运行

## 功能特性

- 支持单个音频文件转写
- 支持整个文件夹递归批量处理
- 自动切段，适合长音频
- 输出 `txt/json` 结果
- 支持断点复用已有切段和分段转写结果
- 批量模式下支持模型预加载复用
- 首次下载模型时会显示下载进度
- 同时保留交互式和参数式两种 CLI 工作方式

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .[dev]
```

如果你打算使用 GPU，请先准备好系统层面的 CUDA 12 与 cuDNN 9。  
可参考 [docs/windows-gpu-notes.md](/c:/Users/Administrator/Desktop/atcstt/docs/windows-gpu-notes.md)。

## 推荐使用方式

### 1. 交互式 CLI

这是最适合日常人工操作的方式：

```powershell
.\.venv\Scripts\python.exe .\run.py
```

安装为包之后，也可以直接运行：

```powershell
atc-asr
```

启动后会依次提示：

1. 选择 `GPU` 或 `CPU`
2. 选择转写模型
3. 选择识别语言
4. 输入音频文件或文件夹路径

默认规则：

- 默认设备是 `GPU`
- GPU 默认模型是 `large-v3`
- CPU 默认模型是 `tiny.en`
- 默认语言是 `en`
- GPU 默认 `compute_type=float16`
- CPU 默认 `compute_type=int8`

### 2. 参数式 CLI

适合脚本、自动化流程和外部调用：

```powershell
atc-asr-cli .\sample.wav --output-dir .\outputs\sample --model small --device cpu --compute-type int8 --language en
```

或者：

```powershell
.\.venv\Scripts\python.exe -m atc_asr.cli .\sample.wav --output-dir .\outputs\sample --model small --device cpu --compute-type int8 --language en
```

## 常用示例

只切段，不转写：

```powershell
.\.venv\Scripts\python.exe -m atc_asr.cli .\sample.wav --output-dir .\outputs\sample --split-only
```

CPU 试跑：

```powershell
.\.venv\Scripts\python.exe -m atc_asr.cli .\sample.wav --output-dir .\outputs\sample_cpu --model tiny.en --device cpu --compute-type int8 --language en
```

使用 `small` 模型：

```powershell
.\.venv\Scripts\python.exe -m atc_asr.cli .\sample.wav --output-dir .\outputs\sample_small --model small --device cpu --compute-type int8 --language en
```

自动检测语言：

```powershell
.\.venv\Scripts\python.exe -m atc_asr.cli .\sample.wav --language auto
```

## 文件夹批处理

如果输入的是文件夹，程序会：

- 递归遍历所有子目录
- 自动筛选常见音视频文件
- 逐个文件执行切段和转写
- 在 `outputs` 下保留相对目录结构

示例输出结构：

```text
outputs\<输入文件夹名>\<相对子目录>\<原文件名>\
```

## 输出结果

单文件模式默认输出到：

```text
outputs\<输入音频文件名>\
```

主要产物包括：

- `chunks/`
- `transcripts/`
- `chunk_manifest.json`
- `transcript.json`
- `transcript.txt`

## 未来打包为 EXE 的建议

如果后续要做商业化桌面交付，建议把交互式 CLI 作为主入口：

- 直接使用 [run.py](/c:/Users/Administrator/Desktop/atcstt/run.py)
- 或者使用包入口 `atc_asr.launcher:main`

这样做的好处是：

- 入口稳定，不依赖浏览器
- 不需要维护前后端通信
- 更适合 PyInstaller、Nuitka 等方式打包
- 后续如果要接入授权、日志、配置文件、任务队列，也更容易在 CLI 编排层扩展

## 测试

运行测试：

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## 说明

- 当前仓库已经移除了 web 端代码、入口、依赖和测试
- `ref/` 中的内容仅作为参考，不属于当前运行时架构
- 如果你后面要继续拆分配置管理、日志模块或打包脚本，可以优先从 `launcher.py` 这一层继续演进
