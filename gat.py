from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_max_pool
from torch.nn import Linear, Dropout


class GATNet2(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, heads=4, dropout=0.5, num_classes=2, training=False):
        super(GATNet2, self).__init__()
        self.gc1 = GATConv(num_node_features, hidden_channels,
                           heads=heads, dropout=dropout)
        self.gc2 = GATConv(hidden_channels*heads,
                           hidden_channels, heads=heads, dropout=dropout)
        self.gc3 = GATConv(hidden_channels*heads,
                           hidden_channels, heads=heads, dropout=dropout)
#         self.gc4 = GATConv(hidden_channels*heads,
#                            hidden_channels, heads=heads, dropout=dropout)

        self.dropout = dropout
        self.training = training
        self.lin1 = Linear(hidden_channels*heads, (int)
                           (hidden_channels*heads / 2))
        self.lin2 = Linear((int)(hidden_channels*heads / 2), num_classes)

        self.droput = dropout
#         self.dropout = Dropout(p=0.3)

    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.dropout(x)
        x = self.gc1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.dropout(x)
        x = self.gc2(x, edge_index)
        x = F.elu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.dropout(x)
#         x = self.gc3(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.dropout(x)
#         x = self.gc4(x, edge_index)
#         x = F.relu(x)

        x = global_max_pool(x, batch)
        x = F.dropout(x)
        x = self.lin1(x)
        x = F.dropout(x)
        x = self.lin2(x)

        return x

    def _train(self, optimizer, criterion, train_loader, DEVICE="cpu"):
        globalLoss, y_true, y_pred = 0, [], []
        for data in train_loader:
            # Iterate in batches over the training dataset.
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

    def _testEarly(self, loader, criterion, DEVICE="cpu"):
        loss, y_true, y_pred = 0, [], []
        # Iterate in batches over the training/test dataset.
        for data in loader:
            data = data.to(DEVICE)
            out = self.forward(data.x, data.edge_index, data.batch)
            # Use the class with highest probability.
            y_hat = data.y.cpu()

            loss = loss + criterion(out.cpu(), y_hat).item()

            y_true.extend(data.y.cpu().tolist())
            y_pred.extend(out.argmax(dim=1).cpu().tolist())

        # Precision, Recall, F1 score
        precision, recall, f1score, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro')  # micro

        return (loss / len(loader.dataset)), accuracy_score(y_true, y_pred, normalize=True), precision, recall, f1score


class GATNet3(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, heads=4, dropout=0.5, num_classes=2, training=False):
        super(GATNet3, self).__init__()
        self.gc1 = GATConv(num_node_features, hidden_channels,
                           heads=heads, dropout=dropout)
        self.gc2 = GATConv(hidden_channels*heads,
                           hidden_channels, heads=heads, dropout=dropout)

        self.dropout = dropout
        self.training = training
        self.lin1 = Linear(hidden_channels*heads, (int)
                           (hidden_channels*heads / 2))
        self.lin2 = Linear((int)(hidden_channels*heads / 2), num_classes)
        
        self.dropout = dropout

#         self.dropout = Dropout(p=0.3)

    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.dropout(x)
        x = self.gc1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.dropout(x)
        x = self.gc2(x, edge_index)
        x = F.elu(x)

        x = global_max_pool(x, batch)
        x = F.dropout(x)
        x = self.lin1(x)
        x = F.dropout(x)
        x = self.lin2(x)

        return x

    def _train(self, optimizer, criterion, train_loader, DEVICE="cpu"):
        globalLoss, y_true, y_pred = 0, [], []
        for data in train_loader:
            # Iterate in batches over the training dataset.
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

    def _testEarly(self, loader, criterion, DEVICE="cpu"):
        loss, y_true, y_pred = 0, [], []
        # Iterate in batches over the training/test dataset.
        for data in loader:
            data = data.to(DEVICE)
            out = self.forward(data.x, data.edge_index, data.batch)
            # Use the class with highest probability.
            y_hat = data.y.cpu()

            loss = loss + criterion(out.cpu(), y_hat).item()

            y_true.extend(data.y.cpu().tolist())
            y_pred.extend(out.argmax(dim=1).cpu().tolist())

        # Precision, Recall, F1 score
        precision, recall, f1score, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro')  # micro

        return (loss / len(loader.dataset)), accuracy_score(y_true, y_pred, normalize=True), precision, recall, f1score
