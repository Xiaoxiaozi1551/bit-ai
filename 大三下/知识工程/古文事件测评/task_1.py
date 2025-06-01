import json
import jsonlines
import os
import numpy as np
import pandas
import torch
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from torchcrf import CRF
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

# 文件路径
original_path = 'data-1120213587/datasets/CHED_data/触发词识别任务数据/'
doc2id_path = 'doc2id.jsonl'
fine_grained_labels_path = 'fine_grained_labels.jsonl'
train_path = 'train.jsonl'
valid_text_only_path = 'valid text_only_data.jsonl'
valid_path = 'valid.jsonl'
test_text_only_path = 'CHED2024-main/datasets/CHED_data/无答案测试集/test text_only_data.jsonl'

# 读取 jsonl 文件的函数
def read_jsonl(file_path):
    data = []
    if file_path != 'CHED2024-main/datasets/CHED_data/无答案测试集/test text_only_data.jsonl':
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
    for entry in data:
        sen_id = entry['sen_id']
        doc_id = entry['doc_id']
        text = entry['text']
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
# pd.set_option("display.max_rows", None)  # 设置最大行数为None，显示全部行数
# pd.set_option("display.max_columns", None)  # 设置最大列数为None，显示全部列数
# print(train_df[['texat', 'trigger', 'label']])
valid_df = pd.DataFrame(valid_event_data)
test_df = pd.DataFrame(test_event_data)

class EventDataset(Dataset):
    def __init__(self, texts, labels, sen_ids, doc_ids, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.sen_ids = sen_ids
        self.doc_ids = doc_ids
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        item['sen_id'] = self.sen_ids[idx]
        item['doc_id'] = self.doc_ids[idx]
        return item

def create_labels(text, events, max_length, tokenizer):
    tokens = tokenizer.tokenize(text)
    labels = ['O'] * len(tokens)

    for event in events:
        start, end = event['start_offset'], event['end_offset']
        event_tokens = tokenizer.tokenize(text[start:end + 1])

        labels[start] = 'B'
        for i in range(1, len(event_tokens)):
            labels[start + i] = 'I'

    # 将标签转换为数字格式
    label_map = {'O': 0, 'B': 1, 'I': 2}
    label_ids = [label_map[label] for label in labels]

    # 如果句子长度超过最大长度，则进行截断
    if len(label_ids) > max_length:
        label_ids = label_ids[:max_length]
    else:
        # 否则进行填充
        label_ids += [0] * (max_length - len(label_ids))

    return label_ids

# 准备数据集
tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
max_length = 128

# 处理好数据集，并将触发词的起始和终止位置标记为标签
def create_unique_labels(dataframe, max_length, tokenizer):
    unique_labels = []
    processed_sen_ids = set()
    unique_texts = []

    for _, row in dataframe.iterrows():
        sen_id = row['sen_id']
        if sen_id in processed_sen_ids:
            continue
        labels = create_labels(row['text'], dataframe[dataframe['sen_id'] == sen_id].to_dict('records'), max_length,
                               tokenizer)
        unique_labels.append(labels)
        unique_texts.append(row['text'])
        processed_sen_ids.add(sen_id)

    return unique_texts, unique_labels


# 为训练集和验证集创建唯一标签
train_texts, train_labels = create_unique_labels(train_df, max_length, tokenizer)
# for item in train_labels:
#     print(item)
valid_texts, valid_labels = create_unique_labels(valid_df, max_length, tokenizer)
test_texts, test_labels = create_unique_labels(test_df, max_length, tokenizer)

train_sen_ids = train_df['sen_id'].tolist()
valid_sen_ids = valid_df['sen_id'].tolist()
test_sen_ids = test_df['sen_id'].tolist()
train_doc_ids = train_df['doc_id'].tolist()
valid_doc_ids = valid_df['doc_id'].tolist()
test_doc_ids = test_df['doc_id'].tolist()

train_dataset = EventDataset(train_texts, train_labels, train_sen_ids, train_doc_ids, tokenizer, max_length)
valid_dataset = EventDataset(valid_texts, valid_labels, valid_sen_ids, valid_doc_ids, tokenizer, max_length)
test_dataset = EventDataset(test_texts, test_labels, test_sen_ids, test_doc_ids, tokenizer, max_length)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=16)
test_dataloader = DataLoader(test_dataset, batch_size=16)

class BertCRFModel(torch.nn.Module):
    def __init__(self, bert_model, num_labels):
        super(BertCRFModel, self).__init__()
        self.bert = bert_model
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(bert_model.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence_output)

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            return loss
        else:
            prediction = self.crf.decode(emissions, mask=attention_mask.bool())
            return prediction

# 准备模型
num_labels = 3  # 'O', 'B', 'I'
bert_model = BertModel.from_pretrained('./bert-base-chinese')
model = BertCRFModel(bert_model, num_labels)

# 训练参数
epochs = 5
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 训练模型
train_losses = []
for epoch in range(epochs):
    model.train()
    total_train_loss = 0

    for batch in tqdm(train_dataloader, desc=f'Training Epoch {epoch + 1}'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        loss = model(input_ids, attention_mask, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    print(f'Epoch {epoch + 1}/{epochs}')
    print(f'Train Loss: {avg_train_loss}')

# 绘制损失曲线
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
total_eval_loss = 0
all_predictions = []
all_labels = []
all_texts = []
all_sen_ids = []
all_doc_ids = []

# 验证
with torch.no_grad():
    progress_bar = tqdm(valid_dataloader, desc='Evaluating', leave=True)
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        loss = model(input_ids, attention_mask=attention_mask, labels=labels)
        total_eval_loss += loss.item()

        predictions = model(input_ids, attention_mask=attention_mask)

        # 确保预测的长度为max_length
        for pred in predictions:
            if len(pred) > max_length:
                pred = pred[:max_length]
            else:
                pred.extend([0] * (max_length - len(pred)))
            all_predictions.append(pred)

        all_labels.extend(labels.cpu().numpy())
        all_texts.extend(batch['input_ids'].cpu().numpy())
        all_sen_ids.extend(batch['sen_id'].cpu().numpy())
        all_doc_ids.extend(batch['doc_id'].cpu().numpy())

avg_eval_loss = total_eval_loss / len(valid_dataloader)
print(f'Evaluation Loss: {avg_eval_loss}')

# 测试
test_total_eval_loss = 0
test_all_predictions = []
test_all_labels = []
test_all_texts = []
test_all_sen_ids = []
test_all_doc_ids = []

with torch.no_grad():
    progress_bar = tqdm(test_dataloader, desc='Predicting', leave=True)
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        loss = model(input_ids, attention_mask=attention_mask, labels=labels)
        test_total_eval_loss += loss.item()

        predictions = model(input_ids, attention_mask=attention_mask)

        # 确保预测的长度为max_length
        for pred in predictions:
            if len(pred) > max_length:
                pred = pred[:max_length]
            else:
                pred.extend([0] * (max_length - len(pred)))
            test_all_predictions.append(pred)

        test_all_labels.extend(labels.cpu().numpy())
        test_all_texts.extend(batch['input_ids'].cpu().numpy())
        test_all_sen_ids.extend(batch['sen_id'].cpu().numpy())
        test_all_doc_ids.extend(batch['doc_id'].cpu().numpy())

# 计算 BLEU（2-gram）分数
def calculate_bleu(predictions, references):
    bleu_scores = []
    for pred, ref in zip(predictions, references):
        bleu_scores.append(sentence_bleu([ref], pred, weights=(0.5, 0.5)))
    return sum(bleu_scores) / len(bleu_scores)

# 计算 ROUGE-2 分数
def calculate_rouge(predictions, references):
    rouge = Rouge()
    rouge_scores = []
    for pred, ref in zip(predictions, references):
        scores = rouge.get_scores(' '.join(map(str, pred)), ' '.join(map(str, ref)))
        rouge_2_fmeasure = scores[0]['rouge-2']['f']
        rouge_scores.append(rouge_2_fmeasure)
    return sum(rouge_scores) / len(rouge_scores)

# 计算 Exact-match-score
def calculate_exact_match(predictions, references):
    exact_matches = 0
    for pred, ref in zip(predictions, references):
        if pred == ref.tolist():  # 将 ref 转换为列表
            exact_matches += 1
    return exact_matches / len(predictions)

# 计算总分
bleu_score = calculate_bleu(all_predictions, all_labels)
rouge_score = calculate_rouge(all_predictions, all_labels)
exact_match_score = calculate_exact_match(all_predictions, all_labels)
total_score = (bleu_score + rouge_score + exact_match_score) / 3

print(f'BLEU (2-gram) Score: {bleu_score}')
print(f'ROUGE-2 Score: {rouge_score}')
print(f'Exact-match-score: {exact_match_score}')
print(f'Total Score: {total_score}')

# 保存预测结果
output_path = 'task1_valid_predict.jsonl'
with jsonlines.open(output_path, mode='w') as writer:
    for i, (text, pred, sen_id, doc_id) in enumerate(zip(all_texts, all_predictions, all_sen_ids, all_doc_ids)):
        pred_events = []
        for j, label in enumerate(pred):
            if label == 1:  # 'B' 标签
                start_offset = j
                end_offset = j
                while end_offset + 1 < len(pred) and pred[end_offset + 1] == 2:  # 'I' 标签
                    end_offset += 1
                trigger = tokenizer.decode(text[start_offset+1:end_offset+1])
                pred_events.append({
                    'id': f'event{i}_{len(pred_events)}',
                    'trigger': trigger,
                    'label': 'Event',  # 填充占位标签
                    'start_offset': start_offset,
                    'end_offset': end_offset
                })

        writer.write({
            'sen_id': int(sen_id),
            'doc_id': int(doc_id),
            'text': tokenizer.decode(text, skip_special_tokens=True).replace(' ', ''),
            'events': pred_events
        })

output_path = 'task1_test_predict.jsonl'
with jsonlines.open(output_path, mode='w') as writer:
    for i, (text, pred, sen_id, doc_id) in enumerate(zip(test_all_texts, test_all_predictions, test_all_sen_ids, test_all_doc_ids)):
        pred_events = []
        for j, label in enumerate(pred):
            if label == 1:  # 'B' 标签
                start_offset = j
                end_offset = j
                while end_offset + 1 < len(pred) and pred[end_offset + 1] == 2:  # 'I' 标签
                    end_offset += 1
                trigger = tokenizer.decode(text[start_offset+1:end_offset+1]).replace(' ', '')
                pred_events.append({
                    'id': f'event{i}_{len(pred_events)}',
                    'trigger': trigger,
                    'label': 'Event',  # 填充占位标签
                    'start_offset': start_offset,
                    'end_offset': end_offset
                })

        writer.write({
            'sen_id': int(sen_id),
            'doc_id': int(doc_id),
            'text': tokenizer.decode(text, skip_special_tokens=True).replace(' ', ''),
            'events': pred_events
        })
