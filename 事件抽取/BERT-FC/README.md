# 环境配置
由于我本机没有GPU，此处通过AutoDL线上平台进行测试

这是租用的线上平台的配置，您在复现时，最好保持Cuda版本、python版本和下述版本一致：

Python  3.10(ubuntu22.04)

Cuda    12.1

GPU     RTX 4090D(24GB) * 1

CPU     15 vCPU Intel(R) Xeon(R) Platinum 8474C



下面是依赖库的版本，同样地，您在复现时最好保持版本一致，这样可以尽可能减少复现时出现的问题：

PyTorch           2.1.0

transformers      4.41.2

seqeval           1.2.2


# 安装Bert-Base-Chinese模型
如果您也像我一样在AutoDL线上平台运行代码，可能会遇到在线上平台无法使用代理，进而导致无法从HuggingFace平台下载Bert-Base-Chinese预训练模型的问题。
对此，您可以在线上平台打开一个命令行，输入`HF_ENDPOINT=https://hf-mirror.com huggingface-cli download bert-base-chinese`，这样就可以从国内镜像下载bert-base-chinese模型了

# 注意，对于事件抽取，您还要关注一下config.py
其中一定需要修改的是配置项是：self.bert_dir，需要将其指向您电脑上BERT-Base-Chinese模型的存储路径⭐

如果是windows平台，大概存储在这里/C:/Users/ji/.cache/huggingface/hub/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55/，注意最后面是一串随机数，您的可能会和我的不一样，需要您在自己电脑上找到该路径，并配置在这里

如果是linux平台，如AUTODL线上平台，大概会存储在这里/root/.cache/huggingface/hub/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f/，注意最后面是一串随机数，您的可能会和我的不一样，需要您在自己电脑上找到该路径，并配置在这里

配置好该项之后，模型就可以运行起来了。

但是您在训练过程中可能会遇到显存不足，或者训练时间过长的问题，此时可以尝试降低如下两个配置项

self.train_epochs = 20  # Max training epoch

self.train_batch_size = 16  # training batch size