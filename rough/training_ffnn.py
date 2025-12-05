import torch
import torch.nn as nn
import torch.optim as optim

# Simple FFNN model
class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Dummy data (replace with yours)
N = 1000
input_dim = 6
num_classes = 3

X = torch.randn(N, input_dim)                  # features
y = torch.randint(0, num_classes, (N,))        # class labels

# Simple train/val split
train_X, val_X = X[:800], X[800:]
train_y, val_y = y[:800], y[800:]

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_X, train_y),
    batch_size=32, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(val_X, val_y),
    batch_size=32
)

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"

model = FFNN(input_dim=input_dim, hidden_dim=64, output_dim=num_classes).to(device)
criterion = nn.CrossEntropyLoss()   # for classification
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train loop
for epoch in range(10):
    model.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    #  Validation accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1:02d} | Loss {total_loss/len(train_loader):.4f} | Val Acc {acc:.3f}")

print("Done!")
