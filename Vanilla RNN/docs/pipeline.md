# Complete Pipeline: Training a Vanilla RNN from Scratch with PyTorch

 Assumption:

 - You already have a **text dataset**.
- You want to train a **vanilla RNN** (not LSTM/GRU/Transformer).
- Goal: build a simple language model that learns to predict the next token.

 The complete pipeline looks like this:

```
Raw Text Dataset
        |
        v
1. Text Cleaning
        |
        v
2. Tokenization
        |
        v
3. Build Vocabulary
        |
        v
4. Convert Text -> Numbers
        |
        v
5. Create Training Sequences
        |
        v
6. Create PyTorch Dataset/DataLoader
        |
        v
7. Build RNN Model
        |
        v
8. Define Loss Function
        |
        v
9. Define Optimizer
        |
        v
10. Training Loop
        |
        v
11. Evaluation
        |
        v
12. Text Generation
```

---

 # 0\. Install Required Libraries

```
pip install torch numpy tqdm
```

 Optional:

```
pip install torchtext
```

 For learning, we will avoid `torchtext` and build everything manually.

---

 # 1\. Prepare Your Text Dataset

 Example dataset:

```
hello world
hello machine learning
deep learning is powerful
machine learning is fun
```

 Load it:

```
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(text[:200])
```

 Example output:

```
hello world
hello machine learning
deep learning is powerful
```

---

 # 2\. Text Cleaning

 Real datasets contain:

 - unnecessary spaces
- strange symbols
- inconsistent cases

 Example:

```
import re

def clean_text(text):
    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()

text = clean_text(text)
```

 Now:

 Before:

```
Hello, World!!!
```

 After:

```
hello world
```

---

 # 3\. Tokenization

 The computer cannot understand:

```
hello world
```

 It needs numbers.

 First split text into words:

```
tokens = text.split()

print(tokens[:10])
```

 Output:

```
[
'hello',
'world',
'hello',
'machine',
'learning'
]
```

---

 # 4\. Build Vocabulary

 A vocabulary maps:

```
word -> number
```

 Example:

```
hello      0
world      1
machine    2
learning   3
```

 Create vocabulary:

```
from collections import Counter

counter = Counter(tokens)

vocab = sorted(counter.keys())

word_to_id = {
    word:i
    for i, word in enumerate(vocab)
}

id_to_word = {
    i:word
    for word,i in word_to_id.items()
}

vocab_size = len(vocab)

print(vocab_size)
```

 Example:

```
50,000
```

---

 # 5\. Convert Text Into Numbers

 Example:

 Original:

```
hello world machine
```

 After encoding:

```
[5, 20, 13]
```

 Code:

```
encoded_text = [
    word_to_id[word]
    for word in tokens
]
```

 Now the dataset becomes:

```
[
5,
20,
13,
8,
...
]
```

---

 # 6\. Create Training Sequences

 For language modeling:

 Input:

```
hello world
```

 Target:

```
world machine
```

 The model learns:

```
previous words -> next word
```

 Example:

 Text:

```
I love AI today
```

 Create samples:

```
Input              Target

I                  love

I love             AI

I love AI          today
```

---

 Example implementation:

```
sequence_length = 5

inputs = []
targets = []

for i in range(
    len(encoded_text)-sequence_length
):

    seq = encoded_text[
        i:i+sequence_length
    ]

    target = encoded_text[
        i+1:i+sequence_length+1
    ]

    inputs.append(seq)
    targets.append(target)
```

---

 Example:

 Input:

```
[10,20,30,40,50]
```

 Target:

```
[20,30,40,50,60]
```

---

 # 7\. Create PyTorch Dataset

 PyTorch expects:

```
Dataset
   |
DataLoader
```

 Create Dataset:

```
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):

    def __init__(
        self,
        inputs,
        targets
    ):
        self.inputs = torch.tensor(
            inputs,
            dtype=torch.long
        )

        self.targets = torch.tensor(
            targets,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,index):

        return (
            self.inputs[index],
            self.targets[index]
        )
```

---

 Create DataLoader:

```
from torch.utils.data import DataLoader

dataset = TextDataset(
    inputs,
    targets
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
```

 Now each batch:

```
Input:

[
 [5,10,20],
 [8,15,30]
]

Target:

[
 [10,20,40],
 [15,30,50]
]
```

---

 # 8\. Build Vanilla RNN Model

 Architecture:

```
Input Word IDs

      |
      v

Embedding Layer

      |
      v

Vanilla RNN

      |
      v

Linear Layer

      |
      v

Next Word Prediction
```

---

 Code:

```
import torch.nn as nn

class VanillaRNN(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(self,x):

        x = self.embedding(x)

        output, hidden = self.rnn(x)

        prediction = self.fc(output)

        return prediction
```

---

 Create model:

```
model = VanillaRNN(
    vocab_size=vocab_size,
    embedding_dim=128,
    hidden_dim=256
)
```

---

 # 9\. Move Model to GPU

 Check GPU:

```
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model.to(device)
```

---

 # 10\. Define Loss Function

 For language prediction:

 Use:

```
CrossEntropyLoss
```

 because the model predicts one word among many.

```
criterion = nn.CrossEntropyLoss()
```

---

 # 11\. Define Optimizer

 Usually:

```
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

 Adam adjusts the model weights automatically.

---

 # 12\. Training Loop

 This is the heart of training.

```
from tqdm import tqdm

epochs = 20

for epoch in range(epochs):

    total_loss = 0

    for x,y in tqdm(loader):

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        output = model(x)

        loss = criterion(
            output.reshape(-1, vocab_size),
            y.reshape(-1)
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}, Loss:",
        total_loss / len(loader)
    )
```

---

 # What happens inside one training step?

 Example:

 Input:

```
hello world
```

 ↓

 Embedding:

```
hello -> [0.2,0.5,...]
```

 ↓

 RNN:

```
memory updated
```

 ↓

 Linear layer:

```
Probability:

hello: 0.01
world: 0.05
machine:0.80
```

 ↓

 Compare with real answer:

```
machine
```

 ↓

 Calculate error

 ↓

 Update weights

---

 # 13\. Evaluate the Model

 Switch to evaluation:

```
model.eval()
```

 Disable gradients:

```
with torch.no_grad():

    for x,y in loader:

        output=model(
            x.to(device)
        )
```

---

 # 14\. Generate Text

 After training:

 Input:

```
"machine"
```

 The model predicts:

```
learning
```

 Then:

```
machine learning
```

 Predict again:

```
is
```

 Finally:

```
machine learning is fun
```

---

 Simple generator:

```
def generate(
    start_word,
    length
):

    model.eval()

    words=[start_word]

    current = torch.tensor(
        [
            word_to_id[start_word]
        ]
    ).unsqueeze(0).to(device)

    for _ in range(length):

        with torch.no_grad():

            output=model(current)

        next_id = output[:,-1].argmax(
            dim=-1
        )

        next_word=id_to_word[
            next_id.item()
        ]

        words.append(next_word)

        current=torch.cat(
            [
                current,
                next_id.unsqueeze(0)
            ],
            dim=1
        )

    return " ".join(words)
```

 Example:

```
print(
    generate(
        "machine",
        10
    )
)
```

 Output:

```
machine learning is fun and useful ...
```

---

 # Complete Project Structure

 A clean project usually looks like:

```
rnn_project/

│
├── data/
│   └── data.txt
│
├── preprocess.py
│
├── dataset.py
│
├── model.py
│
├── train.py
│
├── generate.py
│
└── requirements.txt
```

---

 # Important Concepts You Have Now Connected

 | Concept | Role |
| --- | --- |
| Text dataset | Raw information |
| Tokenization | Convert words to symbols |
| Vocabulary | Word dictionary |
| Tensor | Numbers PyTorch understands |
| Embedding | Convert IDs into vectors |
| RNN | Learn sequence patterns |
| Hidden state | Memory |
| Loss | Measure mistakes |
| Backpropagation | Improve weights |
| Optimizer | Update model |
| Training loop | Repeat learning |

---

 # One Important Note

 A vanilla RNN is excellent for learning how sequence models work, but in real applications it is rarely used today because it struggles with long-term memory.

 The natural progression after this project is:

```
Vanilla RNN
     |
     v
LSTM
     |
     v
GRU
     |
     v
Attention
     |
     v
Transformer
     |
     v
Large Language Models
```

 Building this vanilla RNN pipeline yourself is actually the right first practical project before learning modern LLMs.