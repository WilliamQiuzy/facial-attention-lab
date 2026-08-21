# 当前部署模型：Shared V8 deployment v1

当前锁定部署版本是 `ResidualSharedRouterV8 / RSR8-001`。它把 110D 临床几何、
可用时的 478 点动作轨迹和动作类型编码送入同一个共享 64 维患者表示；数据协议只在
共享表示之后通过小型残差适配器与二分类头处理。模型以 38 名 PalsyNet、36 名
NeuroFace 和 56 名 MEEI 暴露开发参与者进行一次确定性全数据拟合，训练 seed 为 0，
20 个 epoch，冻结权重 SHA-256 为
`72e40ea7b127b6768e931665df622550f06cc5a1bbad20070a42614c5b9901ab`。

本版本已经通过 H200 GPU 服务验收：重启前后各执行 1000 次串行与 200 次并发请求，
三种协议输出完全一致；重启后串行 P95 为 28.07 ms，GPU 显存约 744 MiB，CPU/GPU
最大概率差为 `7.03e-6`。容器使用非 root 用户、只读根文件系统、只读模型挂载、删除
全部 Linux capabilities，并只发布到服务器本机回环地址。

这是一项部署验收，不是临床验证。它没有 Mayo HB 标签训练，也不能报告 Mayo 二分类
正确率或 HB 分级效果。`Universal Clinical Router v4` 暂时保留为当前科学比较基准；
Shared V8 是当前实际部署版本。公开 Git 保存代码、摘要和验收证据，拟合权重只保存在
受限模型发布目录中。

详细发布文件位于 `releases/shared-v8-deployment-v1/`。
