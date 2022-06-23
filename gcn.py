from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

import torch
from torch_geometric.nn import GCNConv, GATConv, global_max_pool
from torch.nn import Linear, Dropout


class GCNNet(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, num_classes=2):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, (int)(hidden_channels / 2))
        self.lin2 = Linear((int)(hidden_channels / 2),
                           (int)(hidden_channels / 4))
        self.out = Linear((int)(hidden_channels / 4), num_classes)
        self.dropout = Dropout(p=0.3)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)
        x = x.relu()

        x = global_max_pool(x, batch)

        x = self.dropout(x)
        x = self.lin1(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.dropout(x)
        x = self.out(x)

        return x

    def _train(self, optimizer, criterion, train_loader, DEVICE="cpu"):
        globalLoss, y_true, y_pred = 0, [], []
        for data in train_loader:  # Iterate in batches over the training dataset.

            optimizer.zero_grad()  # Clear gradients.

            data = data.to(DEVICE)
            # Perform a single forward pass.
            out = self.forward(data.x, data.edge_index, data.batch)

            y_true.extend(data.y.cpu().tolist())
            y_pred.extend(out.argmax(dim=1).cpu().tolist())

            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.

            globalLoss = globalLoss + loss.item()

        # ( correct / len(train_loader.dataset) )
        return (globalLoss / len(train_loader)), accuracy_score(y_true, y_pred, normalize=True)

    def _test(self, loader, criterion, DEVICE="cpu"):
        loss, y_true, y_pred = 0, [], []
        # Iterate in batches over the training/test dataset.
        for data in loader:
            data = data.to(DEVICE)
            out = self.forward(data.x, data.edge_index, data.batch)
            # Use the class with highest probability.
            pred = out.argmax(dim=1).cpu()
            y_hat = data.y.cpu()

            loss = loss + criterion(out.cpu(), y_hat).item()

            y_true.append(y_hat.item())
            y_pred.append(pred.item())

        # Precision, Recall, F1 score
        precision, recall, f1score, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro')  # micro

        return (loss / len(loader.dataset)), accuracy_score(y_true, y_pred, normalize=True), precision, recall, f1score
