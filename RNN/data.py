import torch
from torch.utils.data import DataLoader, Dataset
from data import load_dataset
from transformers import BertTokenizer
from setting import Config  


class YelpDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'label': item['label']
        }

def get_dataloaders(config):
    # 加载数据集
    dataset = load_dataset('yelp_review_full')

    # 映射标签
    def preprocess(example):
        example['label'] = map_label(example['label'])
        return example
    dataset = dataset.map(preprocess)

    # 分词器
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=config.max_seq_len,
            return_tensors=None
        )

    train_data = dataset['train'].map(tokenize_function, batched=True)
    test_data = dataset['test'].map(tokenize_function, batched=True)

    # 设置张量格式
    train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])


    train_dataset = YelpDataset(train_data)
    test_dataset = YelpDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, test_loader