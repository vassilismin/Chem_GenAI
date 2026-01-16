import torch
import torch.nn.functional as F


class GuacaMolTrainer:
    def __init__(self, model, dataloader, device="cpu", lr=1e-4):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        def train_epoch(self):
            """Train for one epoch."""
            self.model.train()
            total_loss = 0.0

            for batch in self.dataloader:
                loss = self.train_step(batch)
                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            return avg_loss

        def train_step(self, batch):
            """Single training step."""
            batch = batch.to(self.device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            self.optimizer.zero_grad()
            hidden = self.model.init_hidden(batch.size(0), self.device)
            output, _ = self.model(inputs, hidden)
            loss = F.cross_entropy(
                output.reshape(-1, output.size(-1)),
                targets.reshape(-1),
                ignore_index=self.pad_idx,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            return loss.item()

        def fit(self, epochs):
            """Train the model for a number of epochs"""
            for epoch in range(epochs):
                avg_loss = self.train_epoch()
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        def save(self, path):
            """Save the model state."""
            torch.save(self.model.state_dict(), path)
