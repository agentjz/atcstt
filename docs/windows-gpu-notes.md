# Windows GPU 说明

## 当前状态

- `faster-whisper` 已能安装和导入
- CPU 转写已跑通
- Windows GPU 是否可用，取决于本机 CUDA 运行时是否完整

## 需要准备什么

如果要在 Windows 上使用 GPU，系统里通常需要这些运行时库：

- CUDA 12 运行时
- cuDNN 9 运行时

这些 DLL 需要能被 Python 进程找到，`faster-whisper` 才能正常调用 GPU。

## 现在的做法

项目代码里不再写死任何外部 CUDA 路径。

如果你的机器已经正确安装 CUDA 和 cuDNN，并且系统 `PATH` 已配置好，程序就会直接使用 GPU。

如果还没有配置好，建议优先从系统环境层面处理，而不是在项目代码里追加本地磁盘路径。

## 常见报错

如果看到类似下面的错误：

```text
Library cublas64_12.dll is not found or cannot be loaded
```

一般说明不是项目代码问题，而是 Windows 上的 CUDA 运行库还没准备好。

## 建议

- 先用 `CPU + tiny.en` 跑通整条流程
- 再切到 `GPU + large-v3`
- 如果 GPU 启动失败，优先检查 CUDA 和 cuDNN，而不是先怀疑转写逻辑
