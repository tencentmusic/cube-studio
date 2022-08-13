class TrainingConfig(object):
    epochs = 10
    batch_size = 16  # batch_size != 1
    lr = 0.0005
    device = 'cuda:7'


class BiLSTMConfig(object):
    input_size = 768  # embedding size
    hidden_size = 512


class BiLSTMCRFTrainConfig(object):
    epochs = 10
    batch_size = 16  # batch_size != 1
    lr = 0.0005
    device = 'cuda:7'
input_size = 768  # embedding size
hidden_size = 512

model_name = "HMM"
data_path = "./data/"
data_name = "annotated_data.txt"
data_rate = [0.7, 0.1, 0.2]
device = 'cpu'
lr = 0.00001
epochs = 10
batch_size = 64
