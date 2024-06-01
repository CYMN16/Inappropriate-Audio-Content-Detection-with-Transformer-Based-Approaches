Explanation for [**"lstmOLID.ipynb"**](./lstmOLID.ipynb)

# Pre-processing

## Importing Datasets
Import our datasets using pandas

```
df = pd.read_csv('datasets/cleaned_OLID.tsv', sep="\t")
```

These cleaned and trimmed datasets can be used with our models:

![available cleaned datasets](images/cleaned_datasets.png "datasets")

All of the datasets are separated by '\t' because the sentences need to have spaces for their own separation 


## Data Imbalance

This technique can be used to sample our data evenly on each label in the case of a data imbalance 

```
# sample_size = 8000
# positive_ratio = 0.5

# pos_df = df[df['label'] == 1].sample(n=np.floor(sample_size*positive_ratio).astype(int), random_state=1)

# neg_df = df[df['label'] == 0].sample(n=np.floor(sample_size*(1-positive_ratio)).astype(int), random_state=1)

# df = pd.concat([pos_df, neg_df])
df = df.sample(frac=1, random_state=42)
```

## Tokenization

All of our text data needs to be tokenized, **BertTokenizer** does the job here.
**BertTokenizer** also provides necessary attention masks (because of empty padding spaces) inside the encodings. 

```
# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Tokenize and encode the training and validation texts
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True)

vocab_size = len(tokenizer.get_vocab())
```

## Data Loaders

Here we prepare how our data is loaded.
**batch_size** enables us to utilize graphics card's parallel processing power.

```
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # item = {'input_ids': torch.tensor(self.encodings[idx])}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TweetDataset(train_encodings, train_labels)
val_dataset = TweetDataset(val_encodings, val_labels)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=12, shuffle=False)
```

This line provides a tensor for all the keys in our encodings(input_ids and attention_mask).

```
item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
```
# Modeling
## LSTM Model Structure
**\_\_init\_\_(...)** allows us to define layers for our model. It must be made sure to give correct size to every layer.

**forward(batch, device)** takes the batch and device as parameters.
**device** needs to be given because the tensors inside each batch must be moved to the gpu and we do it in the forward function(bad practice).

The forward() function is the heart of our model.
This is where all of our layer connections are defined.

```
# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, batch, device):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)

        # Apply attention mask after LSTM
        # Masking the lstm output by zeroing out the effects of padding
        expanded_mask = attention_mask.unsqueeze(-1).expand(lstm_out.size()).float()
        lstm_out = lstm_out * expanded_mask


        lstm_out = torch.cat((lstm_out[:, -1, :self.hidden_size], lstm_out[:, 0, self.hidden_size:]), dim=1)
        out = self.fc(lstm_out)
        return self.sigmoid(out)
```
## Model Setup

**to(device)** is used to send variables(tensors or models) into specified device.

```
# Model setup
input_size = len(tokenizer.get_vocab())
hidden_size = 512
output_size = 1
model = LSTMModel(input_size, hidden_size, output_size, num_layers=2)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```

## Training
We train the model using the loaders.
Sometimes the tensor object needs to be **squeeze**d or **unqueeze**d depending on the function they are used on. 



```
# Training
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        labels = batch['labels'].float().to(device)
        labels = labels.squeeze()
        
        optimizer.zero_grad()
        outputs = model(batch, device)
        outputs = outputs.squeeze()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss:.4f}")
```

**labels** are needed to be float in order to be used in loss calculation
```
labels = batch['labels'].float().to(device)
```
