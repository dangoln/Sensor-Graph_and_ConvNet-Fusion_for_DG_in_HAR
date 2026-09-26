"""
Sensor-Graph Model
==================
Architecture:
  1. Per-node 1D CNN encoder: extracts temporal features from each sensor's
     60-timestep signal → d-dimensional embedding
  2. GNN message passing: propagates information across sensor nodes using
     physically meaningful edges
  3. Graph readout: global_mean_pool → classification head

Weight sharing modes for the CNN encoder:
  - "shared":        all 6 channels use the same CNN (most parameter-efficient)
  - "per_modality":  accel channels share one CNN, gyro channels share another
                     (recommended: respects that accel/gyro have different units)
  - "independent":   each channel has its own CNN (most flexible, most params)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATv2Conv, GINConv, BatchNorm, global_mean_pool
)


class CNN1DEncoder(nn.Module):
    """
    1D CNN that maps a T-length signal to a d-dimensional embedding.

    Architecture: Conv1d layers with ReLU + BatchNorm, followed by a
    temporal-pooling step that collapses the time axis into a fixed-size
    per-channel vector, then a linear projection to embed_dim.

    Temporal pooling modes (P11 — temporal node encoder):
      - "avg":       AdaptiveAvgPool1d(1) — the *mean* over time.
                     This is the original P1–P10 behaviour (default), kept so
                     prior phases reproduce exactly.
      - "bigru":     1-layer bidirectional GRU; the node embedding is the
                     concatenated final forward/backward hidden state (the
                     *last recurrent state*, not the mean). This is the
                     closest analog to ConvNet's LSTM temporal modelling.
      - "attention": learned attention weights over the conv timesteps; the
                     node embedding is the attention-weighted sum.

    Only the temporal-pooling block changes between modes — the conv stack and
    the final projection (and therefore embed_dim) are identical, so the graph,
    DANN and readout downstream are untouched.
    """

    def __init__(self, input_length=60, channels=None, kernel_size=5,
                 embed_dim=64, temporal_pool="avg"):
        super().__init__()
        if channels is None:
            channels = [32, 64]

        layers = []
        in_ch = 1  # single-channel 1D signal
        for out_ch in channels:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size,
                                    padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)
        self.temporal_pool = temporal_pool
        c_out = channels[-1]

        if temporal_pool == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif temporal_pool == "bigru":
            # hidden = c_out // 2 per direction → concat = c_out, so the
            # projection Linear(c_out, embed_dim) is reused unchanged.
            assert c_out % 2 == 0, "bigru temporal_pool needs even final CNN width"
            self.gru = nn.GRU(
                input_size=c_out, hidden_size=c_out // 2,
                num_layers=1, batch_first=True, bidirectional=True,
            )
        elif temporal_pool == "attention":
            self.attn_score = nn.Linear(c_out, 1)
        else:
            raise ValueError(f"Unknown temporal_pool mode: {temporal_pool}")

        self.proj = nn.Linear(c_out, embed_dim)

    def forward(self, x):
        """
        Args:
            x: [batch_nodes, T] — raw timestep values for one sensor channel

        Returns:
            [batch_nodes, embed_dim]
        """
        # Reshape to [batch_nodes, 1, T] for Conv1d
        x = x.unsqueeze(1)
        x = self.conv(x)               # [batch_nodes, C_out, T']

        if self.temporal_pool == "avg":
            x = self.pool(x)           # [batch_nodes, C_out, 1]
            x = x.squeeze(-1)          # [batch_nodes, C_out]
        elif self.temporal_pool == "bigru":
            h = x.transpose(1, 2)      # [batch_nodes, T', C_out]
            _, h_n = self.gru(h)       # h_n: [2, batch_nodes, C_out//2]
            x = torch.cat([h_n[0], h_n[1]], dim=-1)  # [batch_nodes, C_out]
        elif self.temporal_pool == "attention":
            h = x.transpose(1, 2)              # [batch_nodes, T', C_out]
            w = torch.softmax(self.attn_score(h), dim=1)  # [batch_nodes, T', 1]
            x = (h * w).sum(dim=1)            # [batch_nodes, C_out]

        x = self.proj(x)               # [batch_nodes, embed_dim]
        return x


class DualViewEncoder(nn.Module):
    """
    P12 — dual-view node encoder with gated fusion.

    A node's feature vector is the concatenation [ time[time_len] | freq[freq_len] ]
    (the P9/P11 "hybrid" layout). Instead of running ONE conv over the glued
    90-length axis (which mixes two non-comparable axes), this encodes the time
    and frequency parts with their OWN 1D CNNs in their native axes, then fuses
    the two embeddings with a learned per-dimension gate:

        g = sigmoid( W [ e_time ; e_freq ] )        # gate in [0,1], one per dim
        e = g * e_time + (1 - g) * e_freq

    Output dim = embed_dim, identical to CNN1DEncoder — so the graph, DANN and
    readout downstream are untouched and this is a drop-in replacement.

    Why: the two views transfer differently across domains (time favours
    sit/stand, frequency favours walk/run/stairs); a gate lets each node/feature
    pick its view rather than being forced through a single shared conv.
    """

    def __init__(self, time_len=60, freq_len=30, channels=None, kernel_size=5,
                 embed_dim=64, temporal_pool="avg"):
        super().__init__()
        self.time_len = time_len
        self.freq_len = freq_len
        self.enc_time = CNN1DEncoder(
            input_length=time_len, channels=channels, kernel_size=kernel_size,
            embed_dim=embed_dim, temporal_pool=temporal_pool,
        )
        self.enc_freq = CNN1DEncoder(
            input_length=freq_len, channels=channels, kernel_size=kernel_size,
            embed_dim=embed_dim, temporal_pool=temporal_pool,
        )
        self.gate = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, x):
        """
        Args:
            x: [batch_nodes, time_len + freq_len]
        Returns:
            [batch_nodes, embed_dim]
        """
        x_time = x[:, :self.time_len]
        x_freq = x[:, self.time_len:self.time_len + self.freq_len]
        e_time = self.enc_time(x_time)          # [nodes, embed_dim]
        e_freq = self.enc_freq(x_freq)          # [nodes, embed_dim]
        g = torch.sigmoid(self.gate(torch.cat([e_time, e_freq], dim=-1)))
        return g * e_time + (1.0 - g) * e_freq


def _make_node_encoder(config, input_length, channels, kernel_size, embed_dim,
                       temporal_pool):
    """
    Factory: return a fresh node encoder per the configured `encoder_mode`.
      - "single"   : one CNN1DEncoder over the whole feature axis (P9/P11 default).
      - "dualview" : DualViewEncoder (P12) — split time/freq, gate-fuse. Requires
                     the "hybrid" feature layout so the axis splits into
                     time[time_len] + freq[input_length - time_len].
    """
    encoder_mode = config.get("encoder_mode", "single")
    if encoder_mode == "dualview":
        time_len = config.get("dualview_time_len", 60)
        freq_len = input_length - time_len
        assert freq_len > 0, (
            f"dualview needs input_length ({input_length}) > dualview_time_len "
            f"({time_len}); use feature_mode='hybrid'.")
        return DualViewEncoder(
            time_len=time_len, freq_len=freq_len, channels=channels,
            kernel_size=kernel_size, embed_dim=embed_dim,
            temporal_pool=temporal_pool,
        )
    return CNN1DEncoder(
        input_length=input_length, channels=channels, kernel_size=kernel_size,
        embed_dim=embed_dim, temporal_pool=temporal_pool,
    )


class SensorGraphModel(nn.Module):
    """
    Sensor-graph GNN for HAR classification.

    Combines CNN temporal encoding with GNN cross-sensor reasoning.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        input_length = config.get("daghar_timesteps", 60)
        cnn_channels = config.get("cnn_channels", [32, 64])
        kernel_size  = config.get("cnn_kernel_size", 5)
        embed_dim    = config.get("cnn_embed_dim", 64)
        sharing      = config.get("cnn_weight_sharing", "per_modality")
        temporal_pool = config.get("cnn_temporal_pool", "avg")  # P11: avg|bigru|attention
        gnn_type     = config.get("gnn_type", "sage")
        gnn_layers   = config.get("gnn_layers", 2)
        gnn_hidden   = config.get("gnn_hidden", 64)
        dropout      = config.get("gnn_dropout", 0.2)
        num_classes  = config.get("num_classes", 6)
        use_edge_type = config.get("use_edge_type", True)

        self.sharing = sharing
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.use_edge_type = use_edge_type
        # P24 NC — original channel ids per node (default: all 6). Channels < 3
        # are accelerometer, >= 3 gyroscope; with a subset, graphs have fewer
        # nodes and the per-modality encoder split follows the original ids.
        self.node_channel_ids = list(config.get("node_channel_ids", range(6)))
        self.nodes_per_graph = len(self.node_channel_ids)

        # ── Node Encoders (P12: single CNN or dual-view gated fusion) ──────
        def make_encoder():
            return _make_node_encoder(
                config, input_length, cnn_channels, kernel_size,
                embed_dim, temporal_pool,
            )

        if sharing == "shared":
            self.cnn_accel = make_encoder()
            self.cnn_gyro = self.cnn_accel  # same object
        elif sharing == "per_modality":
            self.cnn_accel = make_encoder()
            self.cnn_gyro  = make_encoder()
        elif sharing == "independent":
            self.cnns = nn.ModuleList([make_encoder() for _ in range(6)])
        else:
            raise ValueError(f"Unknown weight sharing mode: {sharing}")

        # ── Edge type embedding (optional) ────────────────────────────────
        if use_edge_type:
            # 3 edge types: intra-accel, intra-gyro, cross-modal
            self.edge_type_embed = nn.Embedding(3, embed_dim)

        # ── GNN Layers ────────────────────────────────────────────────────
        self.gnn_convs = nn.ModuleList()
        self.gnn_bns   = nn.ModuleList()

        for i in range(gnn_layers):
            in_dim = embed_dim if i == 0 else gnn_hidden
            out_dim = gnn_hidden

            if gnn_type == "gcn":
                conv = GCNConv(in_dim, out_dim)
            elif gnn_type == "sage":
                conv = SAGEConv(in_dim, out_dim)
            elif gnn_type == "gatv2":
                conv = GATv2Conv(in_dim, out_dim, heads=1)
            elif gnn_type == "gin":
                mlp = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim),
                )
                conv = GINConv(mlp)
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")

            self.gnn_convs.append(conv)
            self.gnn_bns.append(BatchNorm(out_dim))

        # ── Classification Head ───────────────────────────────────────────
        self.classifier = nn.Linear(gnn_hidden, num_classes)

    def encode_nodes(self, x):
        """
        Apply CNN encoders to produce node embeddings.

        Args:
            x: [total_nodes, T] where total_nodes = batch_size * 6
               Nodes are ordered: for each graph, nodes 0-2 are accel,
               nodes 3-5 are gyro.

        Returns:
            [total_nodes, embed_dim]
        """
        total_nodes, T = x.shape

        if self.sharing == "independent":
            # Each sensor channel gets its own CNN
            # Nodes come in groups of 6 (per graph)
            num_graphs = total_nodes // 6
            embeddings = torch.zeros(total_nodes, self.embed_dim,
                                     device=x.device, dtype=x.dtype)
            for ch in range(6):
                # Extract all nodes for this channel across all graphs
                idx = torch.arange(ch, total_nodes, 6, device=x.device)
                embeddings[idx] = self.cnns[ch](x[idx])
            return embeddings

        # For shared / per_modality: separate accel (orig id < 3) and gyro
        # (orig id >= 3) nodes. P24 NC: graphs may have < 6 nodes; the split
        # follows self.node_channel_ids instead of assuming groups of 6.
        npg = self.nodes_per_graph
        num_graphs = total_nodes // npg
        accel_local = [i for i, c in enumerate(self.node_channel_ids) if c < 3]
        gyro_local  = [i for i, c in enumerate(self.node_channel_ids) if c >= 3]

        accel_idx = []
        gyro_idx = []
        for g in range(num_graphs):
            base = g * npg
            accel_idx.extend(base + i for i in accel_local)
            gyro_idx.extend(base + i for i in gyro_local)

        # dtype follows the input so fp16 (half) evaluation works (P24 fix)
        embeddings = torch.zeros(total_nodes, self.embed_dim, device=x.device,
                                 dtype=x.dtype)
        if accel_idx:
            accel_idx = torch.tensor(accel_idx, device=x.device)
            embeddings[accel_idx] = self.cnn_accel(x[accel_idx])
        if gyro_idx:
            gyro_idx = torch.tensor(gyro_idx, device=x.device)
            embeddings[gyro_idx] = self.cnn_gyro(x[gyro_idx])

        return embeddings

    def forward(self, data):
        """
        Forward pass.

        Args:
            data: PyG Batch with:
              - x:          [total_nodes, T]
              - edge_index: [2, total_edges]
              - batch:      [total_nodes]
              - edge_type:  [total_edges] (optional)

        Returns:
            logits: [batch_size, num_classes]
        """
        x = data.x                 # [total_nodes, T]
        edge_index = data.edge_index
        batch = data.batch

        # Step 1: CNN encoding
        h = self.encode_nodes(x)   # [total_nodes, embed_dim]

        # Step 2: Add edge type information (optional)
        if self.use_edge_type and hasattr(data, 'edge_type'):
            # For GATv2: edge features aren't directly supported in basic form
            # For SAGE/GIN: we add edge type embeddings to source node features
            # This is a lightweight approach that works across all GNN types
            edge_type_emb = self.edge_type_embed(data.edge_type)  # [E, embed_dim]
            # Scatter edge type info to source nodes (additive)
            src_nodes = edge_index[0]
            h = h.clone()  # avoid in-place modification
            h.index_add_(0, src_nodes, edge_type_emb * 0.1)  # scaled addition

        # Step 3: GNN message passing
        for conv, bn in zip(self.gnn_convs, self.gnn_bns):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)

        # Step 4: Graph-level readout
        h = global_mean_pool(h, batch)  # [batch_size, gnn_hidden]

        # Step 5: Classification
        return self.classifier(h)

    def get_graph_embedding(self, data):
        """
        Get the graph-level embedding (before classification head).
        Used for domain adversarial training and t-SNE visualization.
        """
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        h = self.encode_nodes(x)

        if self.use_edge_type and hasattr(data, 'edge_type'):
            edge_type_emb = self.edge_type_embed(data.edge_type)
            src_nodes = edge_index[0]
            h = h.clone()
            h.index_add_(0, src_nodes, edge_type_emb * 0.1)

        for conv, bn in zip(self.gnn_convs, self.gnn_bns):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)

        return global_mean_pool(h, batch)


class CNNOnlyBaseline(nn.Module):
    """
    CNN-only baseline (no GNN message passing).

    Same CNN encoder as SensorGraphModel, but instead of GNN layers,
    simply concatenates all 6 node embeddings and classifies.

    This serves as an ablation to measure the value added by GNN
    message passing on the sensor graph.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        input_length = config.get("daghar_timesteps", 60)
        cnn_channels = config.get("cnn_channels", [32, 64])
        kernel_size  = config.get("cnn_kernel_size", 5)
        embed_dim    = config.get("cnn_embed_dim", 64)
        num_classes  = config.get("num_classes", 6)
        dropout      = config.get("gnn_dropout", 0.2)
        temporal_pool = config.get("cnn_temporal_pool", "avg")  # P11: avg|bigru|attention

        self.embed_dim = embed_dim
        self.dropout = dropout

        # Per-modality encoder (single CNN or P12 dual-view, per encoder_mode)
        self.cnn_accel = _make_node_encoder(
            config, input_length, cnn_channels, kernel_size, embed_dim, temporal_pool)
        self.cnn_gyro = _make_node_encoder(
            config, input_length, cnn_channels, kernel_size, embed_dim, temporal_pool)

        # Classifier takes concatenated embeddings from all 6 nodes
        self.classifier = nn.Sequential(
            nn.Linear(6 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, data):
        x = data.x          # [total_nodes, T]
        batch = data.batch   # [total_nodes]

        total_nodes = x.shape[0]
        num_graphs = total_nodes // 6

        # Encode accel and gyro nodes
        accel_idx, gyro_idx = [], []
        for g in range(num_graphs):
            base = g * 6
            accel_idx.extend([base, base + 1, base + 2])
            gyro_idx.extend([base + 3, base + 4, base + 5])

        accel_idx = torch.tensor(accel_idx, device=x.device)
        gyro_idx  = torch.tensor(gyro_idx, device=x.device)

        embeddings = torch.zeros(total_nodes, self.embed_dim, device=x.device)
        embeddings[accel_idx] = self.cnn_accel(x[accel_idx])
        embeddings[gyro_idx]  = self.cnn_gyro(x[gyro_idx])

        # Reshape to [num_graphs, 6, embed_dim] then flatten to [num_graphs, 6*embed_dim]
        embeddings = embeddings.view(num_graphs, 6, self.embed_dim)
        embeddings = embeddings.view(num_graphs, -1)

        embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)
        return self.classifier(embeddings)

    def get_graph_embedding(self, data):
        """
        Get graph-level embedding (before classification head).
        Used for DANN and t-SNE visualization.
        """
        x = data.x
        total_nodes = x.shape[0]
        num_graphs = total_nodes // 6

        accel_idx, gyro_idx = [], []
        for g in range(num_graphs):
            base = g * 6
            accel_idx.extend([base, base + 1, base + 2])
            gyro_idx.extend([base + 3, base + 4, base + 5])

        accel_idx = torch.tensor(accel_idx, device=x.device)
        gyro_idx  = torch.tensor(gyro_idx, device=x.device)

        embeddings = torch.zeros(total_nodes, self.embed_dim, device=x.device)
        embeddings[accel_idx] = self.cnn_accel(x[accel_idx])
        embeddings[gyro_idx]  = self.cnn_gyro(x[gyro_idx])

        embeddings = embeddings.view(num_graphs, 6, self.embed_dim)
        embeddings = embeddings.view(num_graphs, -1)  # [num_graphs, 6*embed_dim]

        # Pass through the first layer of classifier (FC + ReLU) to get
        # a comparable embedding size to SensorGraphModel
        embeddings = self.classifier[0](embeddings)  # Linear(6*embed_dim, embed_dim)
        embeddings = self.classifier[1](embeddings)  # ReLU
        return embeddings  # [num_graphs, embed_dim]
