import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from Transformer.transformer_strightforward import Transformer


# Define a dummy dataset for sequence-to-sequence translation
class DummyDataset(Dataset):
    def __init__(self, num_samples, max_seq_len):
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.data = torch.randint(
            low=1, high=100, size=(num_samples, max_seq_len))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Dummy translation task where input and output are the same
        return self.data[idx], self.data[idx]

# Define the Transformer model


# Initialize the dataset and dataloader
dataset = DummyDataset(num_samples=1000, max_seq_len=10)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the Transformer model
num_layers = 2
d_model = 128
num_heads = 4
d_ff = 256
input_vocab_size = 100  # Assuming a vocabulary of integers from 1 to 100
target_vocab_size = 100  # Assuming the same vocabulary as input for simplicity
transformer = Transformer(num_layers, d_model, num_heads,
                          d_ff, input_vocab_size, target_vocab_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(transformer.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch_idx, (src, tgt) in enumerate(dataloader):
        optimizer.zero_grad()
        # For simplicity, no masking is applied in this example
        output = transformer(src, src_mask=None)
        loss = criterion(output.view(-1, target_vocab_size), tgt.view(-1))
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

# Test the trained model
# Generate a sample input sequence
sample_input = torch.randint(low=1, high=100, size=(1, 10))
output = transformer(sample_input, src_mask=None)
predicted_sequence = torch.argmax(output, dim=-1)
print("Sample Input:", sample_input)
print("Predicted Output:", predicted_sequence)
