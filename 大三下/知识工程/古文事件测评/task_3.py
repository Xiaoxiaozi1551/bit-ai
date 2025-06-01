import json
import jsonlines
import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import precision_score, recall_score, f1_score

# 文件路径
original_path = 'data-1120213587/datasets/CHED_data/粗粒度事件类型判定任务数据/'
doc2id_path = 'doc2id.jsonl'
fine_grained_labels_path = 'coarse_grained_labels.jsonl'
train_path = 'train_coarse.jsonl'
valid_text_only_path = 'valid text_only_data.jsonl'
valid_path = 'valid_coarse.jsonl'
test_text_only_path = 'task1_test_predict.jsonl'

# 读取 jsonl 文件的函数
def read_jsonl(file_path):
    data = []
    if file_path != 'task1_test_predict.jsonl':
        f_p = os.path.join(original_path, file_path)
    else:
        f_p = file_path
    with jsonlines.open(f_p) as reader:
        for obj in reader:
            data.append(obj)
    return data

# 加载数据
doc2id_data = read_jsonl(doc2id_path)
fine_grained_labels_data = read_jsonl(fine_grained_labels_path)
train_data = read_jsonl(train_path)
valid_text_only_data = read_jsonl(valid_text_only_path)
valid_data = read_jsonl(valid_path)
test_text_only_data = read_jsonl(test_text_only_path)

# 提取事件数据
def extract_event_data(data):
    extracted_data = []
    seen_texts = set()
    for entry in data:
        sen_id = entry['sen_id']
        doc_id = entry['doc_id']
        text = entry['text']
        if text in seen_texts:
            continue
        seen_texts.add(text)
        if entry.get('events') == None:
            event_id = 0
            trigger = 0
            label = 0
            start_offset = 0
            end_offset = 1
            extracted_data.append({
                'sen_id': sen_id,
                'doc_id': doc_id,
                'text': text,
                'event_id': event_id,
                'trigger': trigger,
                'label': label,
                'start_offset': start_offset,
                'end_offset': end_offset
            })
        else:
            events = entry['events']
            for event in events:
                event_id = event['id']
                trigger = event['trigger']
                label = event['label']
                start_offset = event['start_offset']
                end_offset = event['end_offset']
                extracted_data.append({
                    'sen_id': sen_id,
                    'doc_id': doc_id,
                    'text': text,
                    'event_id': event_id,
                    'trigger': trigger,
                    'label': label,
                    'start_offset': start_offset,
                    'end_offset': end_offset
                })
    return extracted_data

# 提取训练集和验证集中的事件数据
train_event_data = extract_event_data(train_data)
valid_event_data = extract_event_data(valid_data)
test_event_data = extract_event_data(test_text_only_data)

# 转换为 DataFrame 便于后续处理
train_df = pd.DataFrame(train_event_data)
valid_df = pd.DataFrame(valid_event_data)
test_df = pd.DataFrame(test_event_data)

# 创建标签映射
label2id = {item['label']: item['label_id']-1 for item in fine_grained_labels_data}
id2label = {item['label_id']-1: item['label'] for item in fine_grained_labels_data}

# 添加细粒度标签ID到数据集中
train_df['label_id'] = train_df['label'].map(label2id)
valid_df['label_id'] = valid_df['label'].map(label2id)
test_df['label_id'] = 0

# 定义用于创建带 prompt 的输入函数
def create_prompt_input(text, trigger):
    prompt_template = f"事件：{text} 触发词：{trigger}。该事件的类型是？"
    return prompt_template

class EventDataset(Dataset):
    def __init__(self, sen_ids, doc_ids, texts, triggers, labels, start_offset, end_offset, tokenizer, max_length):
        self.sen_ids = sen_ids
        self.doc_ids = doc_ids
        self.texts = texts
        self.triggers = triggers
        self.labels = labels
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        prompt_input = create_prompt_input(self.texts[idx], self.triggers[idx])
        encoding = self.tokenizer(
            prompt_input,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        item['sen_id'] = self.sen_ids[idx]  # 添加sen_id
        item['doc_id'] = self.doc_ids[idx]  # 添加doc_id
        item['text'] = self.texts[idx]  # 添加text
        item['trigger'] = self.triggers[idx]  # 添加trigger
        item['start_offset'] = self.start_offset[idx]  # 添加start_offset
        item['end_offset'] = self.end_offset[idx]  # 添加end_offset
        return item

# 准备数据集
tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
max_length = 128

train_texts = train_df['text'].tolist()
train_triggers = train_df['trigger'].tolist()
valid_texts = valid_df['text'].tolist()
valid_triggers = valid_df['trigger'].tolist()
train_labels = train_df['label_id'].tolist()
valid_labels = valid_df['label_id'].tolist()
train_sen_ids = train_df['sen_id'].tolist()
valid_sen_ids = valid_df['sen_id'].tolist()
train_doc_ids = train_df['doc_id'].tolist()
valid_doc_ids = valid_df['doc_id'].tolist()
train_start = train_df['start_offset'].tolist()
valid_start = valid_df['start_offset'].tolist()
train_end = train_df['end_offset'].tolist()
valid_end = valid_df['end_offset'].tolist()
test_sen_ids = test_df['sen_id'].tolist()
test_doc_ids = test_df['doc_id'].tolist()
test_texts = test_df['text'].tolist()
test_triggers = test_df['trigger'].tolist()
test_labels = test_df['label_id'].tolist()
test_start = test_df['start_offset'].tolist()
test_end = test_df['end_offset'].tolist()

train_dataset = EventDataset(train_sen_ids, train_doc_ids, train_texts, train_triggers, train_labels, train_start,
                             train_end, tokenizer, max_length)
valid_dataset = EventDataset(valid_sen_ids, valid_doc_ids, valid_texts, valid_triggers, valid_labels, valid_start,
                             valid_end, tokenizer, max_length)
test_dataset = EventDataset(test_sen_ids, test_doc_ids, test_texts, test_triggers, test_labels, test_start,
                             test_end, tokenizer, max_length)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=16)
test_dataloader = DataLoader(test_dataset, batch_size=16)

class BertClassifierModel(torch.nn.Module):
    def __init__(self, bert_model, num_labels):
        super(BertClassifierModel, self).__init__()
        self.bert = bert_model
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(bert_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state[:, 0, :])  # 使用 [CLS] 分类 token
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss
        else:
            return logits

# 准备模型
num_labels = len(label2id)
bert_model = BertModel.from_pretrained('./bert-base-chinese')
model = BertClassifierModel(bert_model, num_labels)

# 训练参数
epochs = 5
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

train_losses = []  # 添加一个列表来记录每个epoch的训练损失

# 训练模型
for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        loss = model(input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_train_loss = total_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)  # 记录每个epoch的平均损失
    print(f'Epoch {epoch + 1}, Loss: {avg_train_loss}')

# 绘制训练损失图
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, marker='o', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Across Epochs')
plt.legend()
plt.grid()
plt.show()

# 验证模型并计算评价指标
model.eval()
all_predictions = []
all_labels = []
all_texts = []
all_sen_ids = []
all_doc_ids = []
all_triggers = []
all_start_offsets = []
all_end_offsets = []

test_all_predictions = []
test_all_labels = []
test_all_texts = []
test_all_sen_ids = []
test_all_doc_ids = []
test_all_triggers = []
test_all_start_offsets = []
test_all_end_offsets = []

with torch.no_grad():
    progress_bar = tqdm(valid_dataloader, desc='Evaluating', leave=True)
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(logits, dim=-1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_texts.extend(batch['text'])
        all_sen_ids.extend(batch['sen_id'])
        all_doc_ids.extend(batch['doc_id'])
        all_triggers.extend(batch['trigger'])
        all_start_offsets.extend(batch['start_offset'])
        all_end_offsets.extend(batch['end_offset'])

with torch.no_grad():
    progress_bar = tqdm(test_dataloader, desc='Predicting', leave=True)
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids, attention_mask=attention_mask)
        test_predictions = torch.argmax(logits, dim=-1)

        test_all_predictions.extend(test_predictions.cpu().numpy())
        test_all_labels.extend(labels.cpu().numpy())
        test_all_texts.extend(batch['text'])
        test_all_sen_ids.extend(batch['sen_id'])
        test_all_doc_ids.extend(batch['doc_id'])
        test_all_triggers.extend(batch['trigger'])
        test_all_start_offsets.extend(batch['start_offset'])
        test_all_end_offsets.extend(batch['end_offset'])

# 计算宏平均和微平均指标
macro_precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
macro_recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
macro_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)

micro_precision = precision_score(all_labels, all_predictions, average='micro', zero_division=0)
micro_recall = recall_score(all_labels, all_predictions, average='micro', zero_division=0)
micro_f1 = f1_score(all_labels, all_predictions, average='micro', zero_division=0)

total_score = (macro_f1 + micro_f1) / 2

print(f'Macro Precision: {macro_precision}')
print(f'Macro Recall: {macro_recall}')
print(f'Macro F1: {macro_f1}')
print(f'Micro Precision: {micro_precision}')
print(f'Micro Recall: {micro_recall}')
print(f'Micro F1: {micro_f1}')
print(f'Total Score: {total_score}')

# 保存预测结果
output_path = 'task3_valid_predict_coarse.jsonl'
with jsonlines.open(output_path, 'w') as writer:
    for sen_id, doc_id, text, trigger, start_offset, end_offset, pred in zip(all_sen_ids, all_doc_ids, all_texts, all_triggers, all_start_offsets, all_end_offsets, all_predictions):
        event = {
            "id": None,  # 占位id
            "trigger": trigger,
            "label": id2label[pred],
            "start_offset": int(start_offset),
            "end_offset": int(end_offset)
        }
        result = {
            "sen_id": int(sen_id),
            "doc_id": int(doc_id),
            "text": text,
            "events": [event]
        }
        writer.write(result)

output_path = 'task3_test_predict_coarse.jsonl'
with jsonlines.open(output_path, 'w') as writer:
    for sen_id, doc_id, text, trigger, start_offset, end_offset, pred in zip(test_all_sen_ids, test_all_doc_ids, test_all_texts,
                                                                             test_all_triggers, test_all_start_offsets,
                                                                             test_all_end_offsets, test_all_predictions):
        event = {
            "id": None,  # 占位id
            "trigger": trigger,
            "label": id2label[pred],
            "start_offset": int(start_offset),
            "end_offset": int(end_offset)
        }
        result = {
            "sen_id": int(sen_id),
            "doc_id": int(doc_id),
            "text": text,
            "events": [event]
        }
        writer.write(result)
