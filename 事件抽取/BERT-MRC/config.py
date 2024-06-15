# 把一些配置项放在了这个类中，仅仅是一些配置对象
class ARGS:
    def __init__(self):
        # ⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️ 需要在这里填入BERT的路径 ⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
        # 如果是AUTODL线上平台，则他的bert-base-chinese默认存储在/root/.cache/huggingface/hub/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f/，注意后面的这串随机值可能会不一样,是要找到这个路径并填在这里⚠️
        # self.bert_dir = 'C:/Users/ji/.cache/huggingface/hub/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55'
        self.bert_dir = '/root/.cache/huggingface/hub/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f/'

        # 一些路径
        self.data_dir = './data/final_data/'  # data dir for uer
        self.log_dir = './logs/'  # log dir for uer
        self.output_dir = './checkpoints/'  # the output dir for model checkpoints

        self.num_tags = 65  # number of tags
        self.seed = 123  # random seed
        self.gpu_ids = '0'  # gpu ids to use, -1 for cpu, "0,1" for multi gpu

        self.max_seq_len = 128  # maximum sequence length，一般bert模型其实可以支持到512

        self.swa_start = 3  # the epoch when swa start
        self.dropout_prob = 0.1  # drop out probability
        self.lr = 3e-5  # learning rate for the bert module
        self.other_lr = 3e-4  # learning rate for the module except bert
        self.max_grad_norm = 1.0  # max grad clip
        self.warmup_proportion = 0.1  # warmup proportion
        self.weight_decay = 0.01  # weight decay
        self.adam_epsilon = 1e-8  # adam epsilon

        # ⚠️如果跑的时间太长，降低下面的参数⚠️
        self.train_epochs = 10  # Max training epoch
        # ⚠️如果内存或显存不够，降低下面的参数⚠️
        self.train_batch_size = 64  # training batch size
        self.eval_batch_size = 64  # evaluation batch size

        self.eval_model = True  # whether to eval model after training


args = ARGS()
args

# 这是原来的一些配置项，暂时放在这里以供参考
# win
# python main.py --bert_dir="C:/Users/ji/.cache/huggingface/hub/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f/" --data_dir="./data/final_data/" --log_dir="./logs/" --output_dir="./checkpoints/" --num_tags=65 --seed=123 --gpu_ids="0" --max_seq_len=128 --lr=3e-5 --other_lr=3e-4 --train_batch_size=32 --train_epochs=5 --eval_batch_size=32


# linux
# python main.py --bert_dir="/root/.cache/huggingface/hub/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f/" --data_dir="./data/final_data/" --log_dir="./logs/" --output_dir="./checkpoints/" --num_tags=65 --seed=123 --gpu_ids="0" --max_seq_len=128 --lr=3e-5 --other_lr=3e-4 --train_batch_size=32 --train_epochs=5 --eval_batch_size=32
