import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationalGraphConvLayer(nn.Module):
    """
    Relational Graph Convolutional Layer (R-GCN) with relation-specific degree normalization:
        h_i^(l+1) = sigma( sum_{y in Y} sum_{j in N_i^y} (1 / |N_i^y|) W_y^(l) h_j^(l) + W_self^(l) h_i^(l) )
    """
    def __init__(self, in_features, out_features, num_edge_types, activation=nn.LeakyReLU(0.2), dropout_rate=0.0):
        super(RelationalGraphConvLayer, self).__init__()
        self.num_edge_types = num_edge_types
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        # Dedicated projection weight matrix per bond type
        self.edge_weights = nn.ModuleList([
            nn.Linear(in_features, out_features, bias=False) for _ in range(num_edge_types)
        ])
        self.self_weight = nn.Linear(in_features, out_features, bias=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, node_features, adj_tensor):
        """
        Args:
            node_features: [B, N, in_features]
            adj_tensor:    [B, N, N, num_edge_types]
        Returns:
            updated_nodes: [B, N, out_features]
        """
        B, N, _ = node_features.size()
        messages = torch.zeros(B, N, self.out_features, device=node_features.device)

        for rel_idx in range(self.num_edge_types):
            A_rel = adj_tensor[:, :, :, rel_idx]                    # [B, N, N]
            
            # Degree normalization: 1 / |N_i^y|
            deg = torch.clamp(A_rel.sum(dim=-1, keepdim=True), min=1.0) # [B, N, 1]
            A_norm = A_rel / deg                                       # [B, N, N]

            transformed = self.edge_weights[rel_idx](node_features)    # [B, N, out]
            messages = messages + torch.bmm(A_norm, transformed)

        out = messages + self.self_weight(node_features)
        if self.activation is not None:
            out = self.activation(out)
        return self.dropout(out)


class RelationalGCN(nn.Module):
    """
    Multi-layer R-GCN with masked Global Mean Pooling readout over active nodes:
        h_Gt = Readout({h_i^(L) | i in V_t}) = (1 / |V_t|) sum_{i in V_t} h_i^(L)
    """
    def __init__(self, node_dim, edge_types, hidden_dims=[128, 64], dropout_rate=0.0):
        super(RelationalGCN, self).__init__()
        self.layers = nn.ModuleList()
        current_dim = node_dim

        for h_dim in hidden_dims:
            self.layers.append(
                RelationalGraphConvLayer(
                    current_dim, h_dim, num_edge_types=edge_types,
                    activation=nn.LeakyReLU(0.2), dropout_rate=dropout_rate
                )
            )
            current_dim = h_dim

        self.out_dim = current_dim

    def forward(self, node_features, adj_tensor, node_mask=None):
        """
        Args:
            node_features: [B, N, node_dim]
            adj_tensor:    [B, N, N, edge_types]
            node_mask:     Optional [B, N] or [B, N, 1] binary mask indicating active nodes V_t
        Returns:
            node_embeddings: [B, N, out_dim]
            graph_embedding: [B, out_dim] (mean pooled across active nodes V_t)
        """
        h = node_features
        for layer in self.layers:
            h = layer(h, adj_tensor)

        if node_mask is not None:
            if node_mask.dim() == 2:
                mask = node_mask.unsqueeze(-1).float()  # [B, N, 1]
            else:
                mask = node_mask.float()
            masked_h = h * mask
            active_counts = torch.clamp(mask.sum(dim=1), min=1.0)  # [B, 1]
            graph_embedding = masked_h.sum(dim=1) / active_counts  # [B, out_dim]
        else:
            graph_embedding = torch.mean(h, dim=1)  # [B, out_dim]

        return h, graph_embedding
