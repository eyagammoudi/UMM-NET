import numpy as np
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter


def collate(batch):
    images = [b[0] for b in batch]
    labels = [b[1] for b in batch]

    return {"data": np.array(images), "seg": np.array(labels)}


class DataLoaderMultiProcess:
    def __init__(self, dataset, batch_size=2):
        pass
        self.dataset = dataset
        self.batch_size = batch_size
        self.keys = [i for i in range(len(dataset))]
        self.thread_id = 0

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.keys, self.batch_size, True, None)
        batch_data = []
        for key in selected_keys:
            batch_data.append(self.dataset.__getitem__(key))
        batch_data = collate(batch_data)
        return batch_data

    def __next__(self):
        return self.generate_train_batch()

    def set_thread_id(self, thread_id):
        self.thread_id = thread_id


def get_multiprocess_dataloader(dataset, train_transform, batch_size, num_process=4, num_cache_of_process=3, pin_memory=True):
    dataloader = DataLoaderMultiProcess(dataset, batch_size)
    batchgenerator_train = MultiThreadedAugmenter(dataloader, train_transform, num_process,
                                                  num_cache_of_process,
                                                  pin_memory=pin_memory)
    return batchgenerator_train