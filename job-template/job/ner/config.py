class TrainingConfig(object):
    epochs = 10
    batch_size = 16  # batch_size != 1
    lr = 0.0005

class BiLSTMConfig(object):
    input_size = 768   # embedding size
    hidden_size = 512

class BiLSTMCRFTrainConfig(object):
    epochs = 10
    batch_size = 16  # batch_size != 1
    lr = 0.0005