Explanation for [**"testerlstm.ipynb"**](./testerlstm.ipynb)

# Pre-processing
Same steps as explained in [**"lstm_explanation.md"**](./lstm_explanation.md)

# Loading the Saved Model
The whole model structure needs to be redefined correctly here for this process to work.

```
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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


# Model setup
input_size = len(tokenizer.get_vocab())
hidden_size = 512
output_size = 1
model = LSTMModel(input_size, hidden_size, output_size, num_layers=2)
model.load_state_dict(torch.load('models/OLID_lstm.pth'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

```

Following function is used to load the model weights:
```
model.load_state_dict(torch.load('models/OLID_lstm.pth'))
```


# Evaluation
Perform evaluation on validation set and calculate metrics as needed.

```
import numpy as np
import time
# Evaluation
model.eval()
# Example: calculate accuracy
correct = 0
total = 0
i = 0
prediction_list = np.array([])

with torch.no_grad():
    test_start  = time.time()
    print('start')
    for batch in val_loader:
        labels = batch['labels'].float()

        outputs = model(batch, device).detach().cpu()
        predictions = torch.round(outputs.squeeze())
        correct += ((outputs.squeeze() > 0.5) == labels).sum().item()

        total += labels.size(0)
        prediction_list = np.append(prediction_list, predictions.numpy())
    print('end')
    
    test_end = time.time()

accuracy = correct / total
print(f'Validation Accuracy: {accuracy}')
```

Outputs needs to be detached and sent to the cpu to be able to be used in our later calculations.
```
outputs = model(batch, device).detach().cpu()
```

## Saving the Model for Further Use
This function saves the model weights to a file for future use.

```
torch.save(model.state_dict(), 'models/OLID_lstm.pth')
```

## Sklearn Metrics

```
from sklearn.metrics import classification_report

# Assuming you have the true labels in `val_labels` and the predicted labels in `prediction_list`
report = classification_report(val_labels, prediction_list)

print(report)
```

## Confusion Matrix Calculation

```
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(val_labels, prediction_list)

# Extract TP, TN, FP, FN from the confusion matrix
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
```

