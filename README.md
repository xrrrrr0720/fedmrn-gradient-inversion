# FedMRN 梯度反演与客户端上传隐私实验复现包

将**整个文件夹**复制给接收者，保持内部目录结构，然后在此文件夹中运行命令。包内包含实验脚本、所需 FedMRN 模型代码、FashionMNIST 原始文件、1,000 份历史客户端上传、参考模型和已生成结果；运行时不依赖发送者电脑上的其他路径，也不需要下载数据。

## 文件说明

| 路径 | 用途 |
|---|---|
| `audit.py` | 历史上传结构审计、单轮与三轮受控图像反演 |
| `label_sweep.py` | 前 50 张训练图像的单轮与三轮标签推断 |
| `test_audit.py` | 基础逻辑和包内资源检查 |
| `fedmrn_code/model/` | 实验用 FedMRN 模型与随机噪声层源码 |
| `data/FashionMNIST/raw/` | FashionMNIST 训练和测试原始文件 |
| `traces/1.pt` | 历史参考模型检查点 |
| `traces/client_*_round_*/local_update.npz` | 10 个客户端、各 100 轮的上传快照 |
| `results/` | 500 次优化的数值、图像和受控观测结果 |
| `results_100iter/` | 100 次优化的预实验结果 |
| `verify_bundle.py`、`SHA256SUMS.json` | 传输后逐文件校验完整性 |
| `PLAN.md`、`REPORT.md` | 威胁模型、实验过程、结果与局限 |

## 环境

已验证环境：Windows、Python 3.12.3、PyTorch 2.2.0+cu118、Torchvision 0.17.0+cu118、NumPy 1.26.4、Pillow 10.4.0、NVIDIA GeForce RTX 2050。`requirements.txt` 固定 Python 包的版本；有 CUDA GPU 的机器应安装与驱动兼容的 PyTorch 构建。CPU 也可运行脚本，但完整图像反演明显更慢，且本包未在另一台机器上做跨平台数值验证。

在 PowerShell 中进入**收到的这个文件夹**后运行：

```powershell
python verify_bundle.py
python -m pip install -r requirements.txt
python -m unittest test_audit.py -v
python audit.py --iterations 500 --rounds 3 --sample-index 0 --seed 20261023 --lr 0.1
python label_sweep.py --count 50 --rounds 3 --seed 20261023
```

先运行完整性校验，预期显示 `Verified ... files`；该步骤只使用 Python 标准库。`audit.py` 会覆盖 `results/results.json`、`results/controlled_observations.pt` 和四张 PNG；`label_sweep.py` 会覆盖 `results/label_sweep.json`。要保留随包交付的原始结果，请先复制 `results/`。重跑后若需更新校验清单，运行 `python verify_bundle.py --create`。只加载可信来源的 `.pt` 文件。

## 预期检查点

- 单元测试：8 项；没有 CUDA 时 GPU 专项测试跳过。
- 历史审计：1,000 份客户端上传，每份 25 个完整匹配的状态数组。
- 标签推断：在单样本、单步、已知训练前模型的受控条件下，单轮和三轮均为 50/50。
- 图像反演：500 次优化在本次打包复现中得到原始梯度、单轮、三轮 PSNR 约 6.27、6.74、7.08 dB；三种恢复图均无法辨认原图。即使固定随机种子，浮点与优化结果仍可能随运行环境有小幅差异。

详细证据与结论边界见 [REPORT.md](REPORT.md)。历史快照不是原始梯度，`results/` 也不是论文作者发布的攻击结果。本包包含研究者自行生成的攻击代码、数据与产物；FedMRN 模型代码来源为[论文作者公开仓库](https://github.com/Leopold1423/fedmrn-mm24)。
