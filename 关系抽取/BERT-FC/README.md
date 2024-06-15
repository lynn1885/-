# 环境配置
由于我本机没有GPU，此处通过AutoDL线上平台进行测试

这是租用的线上平台的配置，您在复现时，最好保持Cuda版本、python版本和下述版本一致

Python  3.10(ubuntu22.04)

Cuda    12.1

GPU     RTX 4090D(24GB) * 1

CPU     15 vCPU Intel(R) Xeon(R) Platinum 8474C



下面是依赖库的版本，同样地，您在复现时最好保持版本一致，可以尽可能减少复现时出现的问题

PyTorch           2.1.0

transformers      4.41.2

scikit-learn      1.5.0 ⭐ 注意要安装这个库


# 安装Bert-Base-Chinese模型
如果您也像我一样在AutoDL线上平台运行代码，可能会遇到在线上平台无法使用代理，进而导致无法从HuggingFace平台下载Bert-Base-Chinese预训练模型的问题。
对此，您可以在线上平台打开一个命令行，输入`HF_ENDPOINT=https://hf-mirror.com huggingface-cli download bert-base-chinese`，这样就可以从国内镜像下载bert-base-chinese模型了