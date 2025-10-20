import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import json
import re
import random
import time
import math
import pandas as pd
from tqdm import tqdm

# Определение констант
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 150 # Максимальная длина предложения (для промптов)

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"Доступен GPU: {device_name}")
    device = torch.device("cuda")
else:
    print("GPU недоступен, используется CPU")
    device = torch.device("cpu")

# 1. Класс для словаря
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # SOS и EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# 2. Функции подготовки данных
def normalizeString(s):
    # Упрощенная нормализация
    s = s.lower().strip()
    # Убираем теги <s>, </s>, [INST], [/INST]
    s = re.sub(r'</?s>', '', s)
    s = re.sub(r'\[/?inst\]', '', s)
    s = s.strip()
    # Оставляем базовые символы
    s = re.sub(r"[^a-zA-Zа-яА-ЯёЁ0-9.!?]+", r" ", s)
    s = s.strip()
    return s

def read_data(filepath):
    pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data.get('text', '')
            
            # Извлечение INST и ответа
            match = re.search(r'\[INST\](.*?)\[/INST\](.*?)</s>', text, re.DOTALL)
            if match:
                inst_text = normalizeString(match.group(1))
                resp_text = normalizeString(match.group(2))
                
                # Ограничиваем длину для обучения
                if len(inst_text.split(' ')) < MAX_LENGTH and len(resp_text.split(' ')) < MAX_LENGTH:
                    pairs.append((inst_text, resp_text))
    return pairs

def prepareData(filepath):
    pairs = read_data(filepath)
    print(f"Загружено {len(pairs)} пар промптов")
    
    input_lang = Lang("input")
    output_lang = Lang("output")
    
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
        
    print("Количество слов:")
    print(f"{input_lang.name}: {input_lang.n_words}")
    print(f"{output_lang.name}: {output_lang.n_words}")
    
    return input_lang, output_lang, pairs

# 3. Функции для тензоров
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ') if word in lang.word2index]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

# 4. Определение архитектуры
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=n_layers,
                          dropout=(0 if n_layers == 1 else dropout_p), 
                          bidirectional=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        # Суммируем hidden states из двунаправленной GRU
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        return output, hidden

    def initHidden(self):
        # 2 * n_layers, т.к. bidirectional
        return torch.zeros(2 * self.n_layers, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=n_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        
        # Убедимся, что attn_weights имеет правильную размерность для bmm
        # encoder_outputs (seq_len, hidden) -> (1, seq_len, hidden)
        # attn_weights (1, max_len) -> (1, 1, max_len)
        
        # Обрезаем или дополняем attn_weights до seq_len
        seq_len = encoder_outputs.size(0)
        attn_weights_padded = torch.zeros(1, 1, self.max_length, device=device)
        attn_weights_padded[0, 0, :attn_weights.size(1)] = attn_weights
        
        # Используем фактическую длину seq_len
        attn_applied = torch.bmm(attn_weights_padded[:, :, :seq_len],
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size, device=device)

# 5. Цикл обучения
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # Этот тензор (encoder_outputs) определен корректно
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    # ЭТА СТРОКА БЫЛА ЛИШНЕЙ И НЕВЕРНОЙ (удаляем):
    # encoder_output_states = torch.zeros(input_length, encoder.hidden_size * 2, device=device)
    
    # Энкодер
    for ei in range(input_length):
        # Сохраняем и ВЫХОД (output), и СКРЫТОЕ СОСТОЯНИЕ (hidden)
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        
        # Сохраняем ВЫХОД (размером 128) в тензор ВЫХОДОВ
        encoder_outputs[ei] = encoder_output[0, 0]
        
        # ЭТА СТРОКА БЫЛА НЕВЕРНОЙ (удаляем):
        # encoder_output_states[ei] = encoder_hidden[0,0] + encoder_hidden[1,0] # Суммируем
        
    # Декодер
    decoder_input = torch.tensor([[SOS_token]], device=device)
    
    # Используем последний hidden state энкодера (суммированный по направлениям)
    decoder_hidden = encoder_hidden[0:decoder.n_layers] + encoder_hidden[decoder.n_layers:]

    # Teacher forcing
    for di in range(target_length):
        # Передаем правильный тензор ВЫХОДОВ (encoder_outputs)
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        
        loss += criterion(decoder_output, target_tensor[di])
        decoder_input = target_tensor[di]  # Teacher forcing

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

# 6. Вспомогательные функции
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

# 7. Запуск обучения
def trainIters(encoder, decoder, n_iters, pairs, print_every=1000, learning_rate=0.01):
    start = time.time()
    print_loss_total = 0 

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    # Используем случайные пары из данных
    training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

# 8. Функция инференса (для Части 4)
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, normalizeString(sentence))
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.initHidden()

        encoder_output_states = torch.zeros(input_length, encoder.hidden_size * 2, device=device)
        for ei in range(input_length):
            _, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_output_states[ei] = encoder_hidden[0,0] + encoder_hidden[1,0]

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden[0:decoder.n_layers] + encoder_hidden[decoder.n_layers:]

        decoded_words = []
        
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output_states)
            topv, topi = decoder_output.data.topk(1)
            
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return ' '.join(decoded_words)

# 9. Главный блок
if __name__ == "__main__":
    DATA_FILE = 'prompt_expander_train.jsonl'
    input_lang, output_lang, pairs = prepareData(DATA_FILE)
    
    # Уменьшаем hidden_size для ускорения обучения на CPU
    hidden_size = 128 
    n_layers = 1 # 1 слой для Bi-GRU (станет 2 скрытых)
    
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size, n_layers=n_layers).to(device)
    # В декодере hidden_size должен соответствовать выходу энкодера (который мы суммируем)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, n_layers=n_layers, dropout_p=0.1).to(device)
    
    # n_iters можно увеличить для лучшего качества
    trainIters(encoder1, attn_decoder1, 20000, pairs, print_every=500) 
    
    # Сохранение моделей
    torch.save(encoder1.state_dict(), 'seq2seq_encoder.pth')
    torch.save(attn_decoder1.state_dict(), 'seq2seq_decoder.pth')
    torch.save(input_lang, 'input_lang.pth')
    torch.save(output_lang, 'output_lang.pth')
    print("Модели сохранены.")

    # Пример инференса
    print("\n--- Пример работы ---")
    test_prompt = "Как открыть вклад"
    output = evaluate(encoder1, attn_decoder1, test_prompt)
    print(f"INPUT: {test_prompt}")
    print(f"OUTPUT: {output}")

    test_prompt = "Как настроить APN"
    output = evaluate(encoder1, attn_decoder1, test_prompt)
    print(f"INPUT: {test_prompt}")
    print(f"OUTPUT: {output}")
