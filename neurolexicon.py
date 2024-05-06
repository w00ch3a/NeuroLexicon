import os
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

# Download necessary nltk resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Lemmatizer and stop words setup
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Dataset class to handle text data for RNN
class TextDataset(Dataset):
    def __init__(self, sentences, vocab):
        self.sentences = [[vocab.get(token, vocab['<unk>']) for token in sentence] for sentence in sentences]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return torch.tensor(self.sentences[idx], dtype=torch.long)

# Custom collate function to handle padding for RNN
def collate_batch(batch):
    return pad_sequence(batch, batch_first=True, padding_value=0)  # 0 is '<unk>' in vocab

# Text preprocessing functions
def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(lemmatized_tokens)  # Return a single string per sentence

def process_files(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    sentences = sent_tokenize(text)
    processed_sentences = [clean_and_tokenize(sentence) for sentence in sentences]
    return processed_sentences

def save_to_csv(sentences, filename):
    df = pd.DataFrame(sentences, columns=['text'])
    df.to_csv(filename, index=False)
    print(f"Processed data saved to {filename}")

# Neural network model (RNN)
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.rnn(x)
        output = self.fc(output)
        return output

# Train the RNN model
def train_rnn_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs.transpose(1, 2), batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Average Loss: {total_loss / len(train_loader)}')

# Main function to orchestrate the whole process
def main():
    directory = '~/Documents/nlcon/txt/'
    output_directory = '~/Documents/nlcon/processed_data/'
    os.makedirs(output_directory, exist_ok=True)
    files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.txt')]

    all_sentences = []
    vocab = set()
    for file_path in files:
        sentences = process_files(file_path)
        all_sentences.extend(sentences)
        for sentence in sentences:
            vocab.update(sentence.split())

    vocab = {word: i + 1 for i, word in enumerate(vocab)}  # +1 because 0 is reserved for '<unk>'
    vocab['<unk>'] = 0

    train_sentences, _ = train_test_split(all_sentences, test_size=0.2, random_state=42)
    train_dataset = TextDataset(train_sentences, vocab)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)

    rnn_model = SimpleRNN(vocab_size=len(vocab) + 1, embedding_dim=100, hidden_dim=50)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)

    train_rnn_model(rnn_model, train_loader, criterion, optimizer)

    # Save processed data to CSV
    for file_path in files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        sentences = process_files(file_path)
        output_filename = os.path.join(output_directory, f'{base_name}_processed.csv')
        save_to_csv(sentences, output_filename)

if __name__ == '__main__':
    main()
