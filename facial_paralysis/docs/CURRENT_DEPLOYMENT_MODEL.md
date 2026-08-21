# 当前部署模型：Shared V9 public OCI v1

当前锁定部署版本是 `Shared V9 / BLV9-009` 的三种子集成。公共镜像内置模型代码、
三个冻结权重和校验清单，不需要私有模型挂载或 Docker registry 登录。Compose 使用
不可变镜像摘要：
`ghcr.io/williamqiuzy/facial-attention-lab-shared-v9@sha256:ec0e2b34e2233e159d555ab3761fe113f5b768562ba9d9d7bf7c2d7a27d42c95`。

该镜像已经在 H200 上分别完成 CPU 和 CUDA 启动、readiness 与真实推理验收；同一输入
的 CPU/GPU 概率最大差为 `5.52e-6`。它以 UID/GID 1001 非 root 运行，根文件系统只读，
删除全部 Linux capabilities，只将 API 绑定到宿主机 `127.0.0.1:18090`。

这仍然是研究推理服务，不是临床产品。当前 API 只接受已验证、预处理后的 MediaPipe
临床动作张量；它不直接读取原始视频，也没有 Mayo HB 标签训练，因此不能把启动验收
解释为 Mayo 正确率、HB 分级效果或临床诊断能力。跨服务器操作说明位于
`deploy/shared-v9/README.md`，镜像证据位于
`releases/shared-v9-research-v1/oci_manifest.json`。Shared V8 仅作为历史复现记录保留。
