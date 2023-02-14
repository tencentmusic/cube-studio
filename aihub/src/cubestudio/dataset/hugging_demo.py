

from datasets import Dataset,Audio
import pysnooper

audio_dataset = Dataset.from_dict({"audio": ["audio_test/audio/BAC009S0764W0121.wav", "audio_test/audio/BAC009S0764W0121.wav", "audio_test/audio/BAC009S0764W0121.wav"]})

audio_dataset = audio_dataset.cast_column("audio", Audio(sampling_rate=16000))
# print(audio_dataset['audio'])
# print(audio_dataset[0])
# # print(audio_dataset[0]["audio"])

def prepare_dataset(batch):
    audio = batch["audio"]
    print(audio)
    return batch

table = audio_dataset.map(prepare_dataset, remove_columns=audio_dataset.column_names)

