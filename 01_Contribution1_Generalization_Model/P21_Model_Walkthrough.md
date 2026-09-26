# P21 — Step-by-Step Walkthrough of the Locked Model

This walks through the thesis's main-contribution system (Methodology Chapter 3,
§3.2 "Proposed Generalization Framework") exactly as the code executes it: from
reading a DAGHAR CSV off disk to a final six-way activity prediction. It follows one
concrete run of the pipeline — one leave-one-dataset-out (LODO) fold, e.g. KuHar held
out as the target — since that's the unit the code actually operates on.

Code paths are relative to `SO_DAGHAR_Experiments/PROJECTS/P20_GraphRevival_DG/P21_PearsonGraph/`,
which is the final, locked implementation referenced as "P21" throughout the thesis
and in prior conversation. The orchestration entry point tying every step below
together is `p21/gnn_predict.py`, and it's worth opening that file alongside this
document — every step names the exact function it calls.

---

## Step 1 — Load the Six Domains and Form the Source/Target Split

**Overview:** Before any model logic runs, the pipeline decides which of the six
DAGHAR datasets is the held-out "target" for this fold, and pools the other five as
"sources."

**What happens:** `build_lodo_sensor_graphs(data_root, target_dataset, config, folder_map)`
is called once per fold with `target_dataset` fixed (e.g. `"KuHar"`). It computes
`sources = [d for d in DAGHAR_DATASETS if d != target_dataset]` — the other five names.
It then reads each source dataset's `train.csv` and its `validation.csv`
off disk and concatenates all five of each together (`load_daghar_aggregated`), giving
one large pooled source-training array and one pooled source-validation array. Only
the *target* dataset's `test.csv` is read (`load_daghar_dataset(..., split="test")`) —
its train and validation splits are never touched anywhere in this fold. Each row
loaded is a single 3-second window: 60 timesteps × 6 channels
(accel-x/y/z, gyro-x/y/z), reshaped from the CSV's flat columns into a `[T=60, C=6]`
array per window, alongside its integer activity label (0–5) and which dataset it came
from.

**Code reference:**
- `gnn/data/daghar_loader.py::build_lodo_sensor_graphs()` — the fold orchestrator.
- `load_daghar_dataset()` — reads one dataset's one split's CSV, builds `X` (via
  `df[signal_cols].to_numpy()` reshaped `[N,C,T]→[N,T,C]`) and `y` (from the
  `"standard activity code"` column).
- `load_daghar_aggregated()` — calls `load_daghar_dataset()` once per source and
  `torch.cat`s the results, also recording `sources[i]` (which dataset each row
  came from — needed later for the domain-adversarial head).
- `_make_domain_labels(src_tr)` — converts the five source-dataset name strings into
  integer IDs 0–4 for that purpose.

**Major components in this step:**

- *Leave-one-dataset-out (LODO) split.* This is the thesis's cross-dataset evaluation
  protocol (Chapter 3 §3.1.4/§3.4.1): the target domain must never be seen — not just
  its labels, but its raw signal too — anywhere in training or model selection. That's
  why only `target_dataset`'s *test* split is loaded at all in this step, and why it's
  kept in a separate variable (`X_te, y_te`) rather than being pooled with anything.
  This single design choice is what makes "does this generalize to an unseen dataset"
  an honest question rather than one the model was quietly allowed to peek at.

---

## Step 2 — Build the Hybrid Time + Frequency Node Feature

**Overview:** Each raw 60-timestep channel gets turned into a 90-length feature vector
before anything else happens to it — this is the "hybrid" feature named throughout the
Methodology chapter.

**What happens:** For each channel's 60 raw values, an FFT is computed
(`np.fft.rfft`), producing 31 magnitude values; the first one (the zero-frequency/DC
bin) is dropped, leaving 30 frequency values. Those 30 are concatenated onto the
original 60 raw values, giving 90 numbers per channel per window
(`X_hybrid = np.concatenate([X_np, X_freq], axis=1)`). This happens independently for
every one of the 6 channels, so the array's shape goes from `[N, 60, 6]` to
`[N, 90, 6]` — still 6 channels, just each one now carrying both a time view and a
frequency view of itself.

**Code reference:**
- `daghar_loader.py::_compute_node_features(X, feature_mode="hybrid")` — the
  `elif feature_mode == "hybrid":` branch does exactly this concatenation.
- Called from inside `build_sensor_graphs()`, which is itself called three times per
  fold (once each for the pooled source-train set, the pooled source-validation set,
  and the target-test set) — always with the same `feature_mode="hybrid"` config
  value, so all three splits get an identical feature recipe.

**Major components in this step:**

- *Hybrid time+frequency feature.* Chapter 3's EDA (§3.1.2) found frequency content
  much more activity-discriminative than raw time values via t-SNE, but didn't find
  time-domain information useless either — so rather than picking one, the framework
  keeps both views side by side and lets later layers (Step 6's CNN encoder) learn
  which parts of the 90 numbers matter. Earlier project phases tried this same
  raw+FFT-magnitude combination on a *different* graph structure (timesteps as nodes)
  and it underperformed frequency-alone; it only became a genuine improvement once
  paired with the sensor-as-node graph described in the next step — a reminder that
  this feature's value depends on what it's assembled with, not solely on its own
  informational content.

---

## Step 3 — Normalize Every Feature Using Source-Only Statistics

**Overview:** Before anything is fed to the network, every one of the 90 values per
channel is z-score normalized (mean-subtracted, divided by standard deviation) — but
the mean and standard deviation used are computed *only* from the source-training
data, and then reused unchanged everywhere else.

**What happens:** Inside `build_sensor_graphs()`, when processing the source-training
split, `stats=None` is passed, so the function computes
`means = X_feat.mean(axis=(0,1))` and `stds = X_feat.std(axis=(0,1))` — one mean and
one standard deviation per channel (6 of each), averaged across every training window
and every one of its 90 feature values. These six pairs of numbers are then handed
back out of the function (`stats`) and explicitly passed in as the fixed `stats=`
argument when normalizing the source-validation split and, separately, the
target-test split. Neither the validation split nor the target split is ever allowed
to compute its own statistics.

**Code reference:**
- `build_sensor_graphs()`'s `normalize=True` branch — the `if stats is None:` case
  computes the six-channel means/stds from that call's own data; the `else:` case
  applies a passed-in `(means, stds)` pair.
- In `gnn_predict.py::train_and_predict()`, this is invisible at the call-site level
  because `build_lodo_sensor_graphs()` handles the plumbing internally — but its
  return dict's `norm_stats` field is exactly the frozen `(means, stds)` pair computed
  once on source-train and reused for both source-val and target-test.

**Major components in this step:**

- *Source-only normalization discipline.* This is a small but easy-to-miss form of
  the same "never let the target influence anything" rule from Step 1. If the
  target's *own* mean and standard deviation were used to normalize the target's data,
  the model would implicitly be told something about the target domain's statistics
  before ever making a prediction — a subtle form of leakage. Fitting normalization
  once, only on the five source domains, and freezing it there keeps the target truly
  unseen in every sense, not just in its labels.

---

## Step 4 — Build the Sensor Graph's Edges: Adaptive Pearson-Blend Construction

**Overview:** Each window is turned into a small graph of 6 nodes (one per sensor
channel); this step decides *which pairs of those 6 nodes get connected*, and it's the
central methodological contribution of the thesis.

**What happens:** For every window, the raw (not yet normalized, not the 90-length
hybrid vector) 60-step signal for each of the 6 channels is used to compute the
absolute Pearson correlation between every one of the
$\binom{6}{2}=15$ possible channel pairs, giving a `[6,6]` correlation matrix per
window (`pearson_abs_matrix`). Each of the 15 pairs then receives a blended score:
`score = 0.5 * |correlation| + 0.5 * (1 if pair is in the fixed physical-prior set else 0)`
— an equal mix of "how correlated were these two channels in *this particular*
window" and "are these two channels physically related regardless of what's happening
in the window" (the fixed 9-pair prior: 3 accel-only pairs, 3 gyro-only pairs, 3
axis-aligned cross-modal pairs). The 9 highest-scoring pairs, by this blended score,
become that window's edges (`select_edges(..., mode="topk", topk=9, alpha=0.5)`). This
whole computation is redone independently for every single window — one window's edge
set has no bearing on another's.

**Code reference:**
- `gnn/data/pearson.py::pearson_abs_matrix(X)` — vectorized per-window `|ρ|` computation
  via centering, `np.einsum` covariance, and normalization by per-channel std.
- `pearson.py::select_edges(R, mode="topk", topk=9, alpha=0.5)` — computes
  `scores = alpha * rho_vec + (1-alpha) * prior_vec`, sorts descending
  (`np.argsort(-scores)`), and keeps the top 9 indices per window.
- `pearson.py::PHYSICAL_PRIOR = {(0,1),(0,2),(1,2),(3,4),(3,5),(4,5),(0,3),(1,4),(2,5)}`
  — the fixed 9-pair set (nodes 0,1,2 = accel-x,y,z; 3,4,5 = gyro-x,y,z).
- Invoked from `build_lodo_sensor_graphs()`'s `if edge_mode == "pearson":` branch,
  which calls `select_edges()` separately for the train, val, and (unlabeled, if
  used) target-unlabeled splits, producing a *different* edge list per window
  (`es_tr`, `es_val`, `es_te`), rather than one shared edge list for the whole
  dataset.
- `daghar_loader.py::build_edge_index(edge_list, use_edge_type=True)` — turns one
  window's list of `(src, dst)` pairs into PyTorch Geometric's bidirectional
  `edge_index` tensor, and tags every edge with a type (0 = intra-accel, 1 =
  intra-gyro, 2 = cross-modal) via `_get_edge_type()`.

**Major components in this step:**

- *Sensor-as-node graph.* Each of the 6 channels — not each timestep — is one graph
  node. This is what makes "which sensor pairs relate to each other" a meaningful
  question to ask at all, and is the structural prerequisite for everything else in
  this step (§3.2.1 of the chapter).
- *Adaptive Pearson-blend construction.* The two ingredients being blended represent
  two different kinds of knowledge: a *physical* prior that never changes (some
  sensor pairs are related purely by how the device is built and worn) and a
  *data-driven* signal that changes every window (which channels happened to move
  together in this particular 3 seconds). Using α=0.5 rather than pure correlation
  (α=1.0) matters because pure correlation, fit only on the five *source* domains, can
  latch onto correlation patterns specific to those domains that don't transfer to an
  unseen sixth one — the fixed physical term acts as a safety net against exactly that
  kind of overfitting-to-source-correlations. GraphSAGE (Step 8) never sees the actual
  score `s_ij` — only whether a pair made the top 9 or not — so this step's real job
  is choosing *topology*, not edge strength.

---

## Step 5 — Assemble Each Window into a Graph Object and Batch Them

**Overview:** The normalized 90-length features (Step 3) and the per-window edge list
(Step 4) are combined into one PyTorch Geometric `Data` object per window, and many
such windows are grouped into mini-batches for training.

**What happens:** For window `i`, its `[6, 90]` array of node features (one row per
sensor channel) becomes `data.x`; its own edge list from Step 4 becomes
`data.edge_index` and `data.edge_type`; its activity label becomes `data.y`; and, for
training windows only, its source-dataset ID becomes `data.domain` (used later by the
domain-adversarial head, Step 9). This is repeated once per window across all three
splits — the pooled 5-source training windows, the pooled 5-source validation windows,
and the held-out target's test windows — producing three lists of `Data` objects.
PyTorch Geometric's `DataLoader` then groups these into batches (64 windows per batch
for training, per the training configuration table), which is what actually gets
handed to the model on each forward pass; because every graph has a fixed 6 nodes,
batching many graphs together is done by simply concatenating all their nodes and
edges into one big disconnected graph, tracked with a `batch` index so pooling (Step 8)
knows which nodes belong to which original window.

**Code reference:**
- `build_sensor_graphs()` — the loop building each `Data(x=..., edge_index=...,
  y=...)` object, one per window, with `data.domain` attached when
  `domain_labels is not None`.
- `p21/gnn_predict.py::train_and_predict()` — where the three `DataLoader(...,
  batch_size=cfg["batch_size"])` calls turn the three `Data` lists into iterable,
  batched training/validation/test loaders.

**Major components in this step:** *(no new conceptual component here — this step is
plumbing that turns Steps 1–4's outputs into the concrete tensors the neural network
in Steps 6–9 actually consumes.)*

---

## Step 6 — Per-Modality CNN Node Encoding

**Overview:** Before any information is exchanged between sensor nodes, each node's
own 90-length feature vector is first summarized into a compact 64-number embedding
by a small 1D convolutional network — one CNN for the 3 accelerometer nodes, a
*separate* CNN for the 3 gyroscope nodes.

**What happens:** `SensorGraphModel.encode_nodes(x)` splits the incoming batch of node
features into an accelerometer group (nodes 0,1,2 of every graph) and a gyroscope
group (nodes 3,4,5 of every graph), and runs each group through its own
`CNN1DEncoder`: two `Conv1d` layers (kernel size 5, widening from 1 input channel to
32, then to 64), each followed by batch normalization and a ReLU nonlinearity; then
`AdaptiveAvgPool1d(1)` collapses the length-90 axis down to a single number per
convolutional channel; then a final linear layer projects that down to a fixed
64-dimensional embedding. The result: every node, regardless of which of the 6
channels it represents, now holds a 64-number embedding rather than its original raw
90-number vector.

**Code reference:**
- `gnn/models/sensor_graph.py::CNN1DEncoder` — the encoder itself: `self.conv =
  nn.Sequential(Conv1d, BatchNorm1d, ReLU, Conv1d, BatchNorm1d, ReLU)`, then
  `self.pool = nn.AdaptiveAvgPool1d(1)`, then `self.proj = nn.Linear(64, embed_dim)`.
- `SensorGraphModel.__init__`'s `sharing == "per_modality"` branch — constructs
  `self.cnn_accel = make_encoder()` and `self.cnn_gyro = make_encoder()` as two
  independent instances (not the same object, unlike the `"shared"` mode).
- `SensorGraphModel.encode_nodes(x)` — builds `accel_idx`/`gyro_idx` index tensors
  picking out every graph's nodes 0–2 vs. 3–5 across the whole batch, then calls
  `self.cnn_accel(x[accel_idx])` and `self.cnn_gyro(x[gyro_idx])` separately.

**Major components in this step:**

- *Per-modality CNN encoder.* A GraphSAGE layer (Step 8) is designed to combine
  information *between* nodes; it isn't built to discover useful local patterns
  *within* one long raw feature vector first. Splitting this local-pattern-finding
  work into its own dedicated CNN step, before any cross-node reasoning, lets the
  network learn things like "does this channel show a repeating rhythm" independently
  per channel. Using two separate encoders (accel vs. gyro) rather than one shared
  encoder respects that accelerometer and gyroscope measure physically different
  quantities (linear acceleration vs. rotational rate) with different units and
  scales, without going all the way to six fully-independent encoders, which would use
  more parameters without a matching physical justification (three accelerometer axes
  really are more alike to each other than to a gyroscope axis).

---

## Step 7 — Inject Edge-Type Information into Node Embeddings

**Overview:** Just before message passing, a small amount of information about *what
kind* of edge connects to each node (intra-accelerometer, intra-gyroscope, or
cross-modal) is added into that node's embedding.

**What happens:** Every edge built in Step 4 was tagged with a type: 0
(intra-accelerometer), 1 (intra-gyroscope), or 2 (cross-modal). A small learned lookup
table converts each edge's type into its own 64-dimensional vector; this vector is then
added, scaled down to 10% strength, onto the embedding of that edge's *source* node
only. This gives every node a small amount of extra context about the kinds of edges
that touch it, even though the GraphSAGE layer that runs next (Step 8) will otherwise
treat every edge identically regardless of type.

**Code reference:**
- `sensor_graph.py::SensorGraphModel.__init__` — `self.edge_type_embed =
  nn.Embedding(3, embed_dim)`, a 3-row lookup table (one row per edge type).
- `SensorGraphModel.forward()` / `get_graph_embedding()` — `edge_type_emb =
  self.edge_type_embed(data.edge_type)` then
  `h.index_add_(0, src_nodes, edge_type_emb * 0.1)`, the explicit 0.1 scaling
  ("scaled addition," per the code's own comment).

**Major components in this step:**

- *Edge-type embedding.* This exists specifically because GraphSAGE's own
  message-passing step (next) has no native way to use edge weights or edge types —
  it only sees "which nodes are connected," treating every connection identically.
  Injecting the edge type here, additively and at reduced strength, is a workaround
  that lets *some* structural information about the kind of connection reach the
  network, without changing which GNN operator is used.

---

## Step 8 — GraphSAGE Message Passing and Graph-Level Readout

**Overview:** The 6 node embeddings are now allowed to exchange information with each
other across the edges chosen in Step 4, through two stacked GraphSAGE layers, and are
finally averaged into one single embedding representing the whole window.

**What happens:** Each `SAGEConv` layer updates every node's embedding by combining
its own current embedding with the average of its directly-connected neighbours'
embeddings (via a learned linear transformation), followed by batch normalization,
ReLU, and dropout (p=0.2). This happens twice (`gnn_layers=2`), so information from a
node's neighbours' neighbours can reach it as well, not just its immediate
neighbours. Because a graph here only ever has 6 nodes and up to 9 edges, two layers
is generally enough for most nodes to reach most others. After both layers, the six
resulting node embeddings for one window are averaged together into a single
64-dimensional graph embedding via `global_mean_pool` — this one vector is what the
rest of the pipeline treats as "the model's summary of this window."

**Code reference:**
- `sensor_graph.py::SensorGraphModel.__init__` — `conv = SAGEConv(in_dim, out_dim)`,
  built from PyTorch Geometric's own `SAGEConv` (imported at the top of the file),
  looped `gnn_layers` (=2) times into `self.gnn_convs`.
- `SensorGraphModel.forward()` — the loop `for conv, bn in zip(self.gnn_convs,
  self.gnn_bns): h = F.dropout(...); h = conv(h, edge_index); h = bn(h); h =
  F.relu(h)`, called with `edge_index` from Step 4 — note `edge_index` alone, with no
  `edge_weight` argument, which is exactly why the Pearson-blend score from Step 4
  never reaches the network directly.
- `global_mean_pool(h, batch)` — the graph-level readout, using the `batch` index
  from Step 5 to know which of the batched nodes belong to which original window.

**Major components in this step:**

- *GraphSAGE backbone.* Chosen, per the chapter, after a broader architecture
  comparison found it the most consistently strong performer once paired with
  domain-adversarial training (the comparison itself is Chapter 4 material, not
  repeated in the Methodology). The property that matters most for the rest of the
  chapter is that GraphSAGE's neighbour-averaging only asks "is this edge present,"
  never "how strong is it" — which is precisely why the adaptive graph construction in
  Step 4 can only act as a *topology selector* rather than a graph re-weighter.

---

## Step 9 — Domain-Adversarial Head (Training-Time Only)

**Overview:** During training, the same 64-dimensional graph embedding from Step 8 is
also fed into a second small classifier whose job is to guess which of the five
*source* datasets the window came from — and the embedding is trained to make that
job as hard as possible, so it becomes less dataset-specific.

**What happens:** Between the graph embedding and this domain-guessing network sits a
Gradient Reversal Layer (GRL): on the forward pass it does nothing (passes the
embedding through unchanged); on the backward pass, when gradients are computed to
adjust the network's weights, it flips the sign of whatever gradient comes from the
domain-classification loss, scaled by a strength value λ. The practical effect: normal
gradient descent applied to this sign-flipped gradient ends up *increasing* the domain
classifier's loss during training on the shared representation, rather than decreasing
it — pushing the graph embedding toward representing "what activity is happening"
while making "which of the five source datasets this came from" progressively harder
to recover. λ itself isn't constant: it ramps up smoothly from 0 to 1 over the course
of training, following an S-shaped curve, so the model spends its early epochs mostly
learning to classify activities well before this extra pressure is applied at full
strength. The domain discriminator only ever has 5 possible outputs (the five source
domains) — the held-out target is never a class it can guess, since that would require
target-domain labels the model is never given.

**Code reference:**
- `gnn/utils/gradient_reversal.py::GradientReversalFunction` — `forward()` returns
  `x.clone()` unchanged; `backward()` returns `-ctx.lambda_val * grad_output`, the
  literal sign flip.
- `gradient_reversal.py::ganin_lambda_schedule(progress, gamma=10.0)` — computes
  `2.0/(1.0+exp(-gamma*progress)) - 1.0`, called once per epoch inside
  `DANNTrainer._get_lambda()` with `progress = current_epoch / total_epochs`.
- `gnn/models/da_sensor_graph.py::DomainAdversarialWrapper` — wraps the
  `SensorGraphModel` backbone: `h = self.backbone.get_graph_embedding(data)`, then
  `activity_logits = self.activity_classifier(h)` (normal gradient flow) and
  `domain_logits = self.domain_discriminator(self.grl(h, lambda_val))` (reversed
  gradient flow), where `self.domain_discriminator` is a small
  `Linear→ReLU→Dropout→Linear` head with exactly `num_domains=5` outputs.

**Major components in this step:**

- *Domain-adversarial training (DANN) and the gradient-reversal layer (GRL).* The
  goal is a representation that carries information about the activity but as little
  as possible about which specific source dataset produced it — the reasoning being
  that a representation that leans on dataset-specific quirks is less likely to
  transfer to a sixth, unseen dataset. The GRL is what makes this trainable with an
  otherwise ordinary optimizer: rather than writing special "maximize this loss"
  logic, a single sign flip on the backward pass repurposes standard gradient descent
  into gradient *ascent* for exactly one branch of the network, with no other change
  to the training code. This head is discarded entirely at inference time (Step 11) —
  it exists purely to shape the embedding during training.

---

## Step 10 — Train the Graph Branch: The Combined Loss and Optimization Loop

**Overview:** Steps 6–9 describe one forward pass through the network; this step is
what happens across many such passes, epoch after epoch, to actually learn the
weights used in every step above.

**What happens:** For each batch of training windows, the model computes both
`activity_logits` and `domain_logits` (Step 9), and combines two losses:
cross-entropy between the predicted and true activity label, plus λ times
cross-entropy between the predicted and true source-domain label
(`loss = act_loss + lambda_val * dom_loss`). This combined loss is backpropagated,
gradients are clipped to a maximum norm of 1.0 (to prevent unstable, oversized weight
updates), and the Adam optimizer updates the weights, with the learning rate following
a cosine-annealing schedule that decays smoothly from `1e-3` toward `1e-5` over the
run. After every epoch of training batches, the model is switched to evaluation mode
and run once over the *source-validation* windows (never the target) to compute a
validation accuracy; if this is the best validation accuracy seen so far, the current
weights are saved as a checkpoint. If 20 consecutive epochs pass without a new best
validation accuracy, training stops early. At the very end, the single best checkpoint
(by source-validation accuracy) is restored — not necessarily the weights from the
final epoch — and that is the model used going forward.

**Code reference:**
- `gnn/training/dann_trainer.py::DANNTrainer.train_epoch()` — computes
  `act_loss = self.activity_criterion(activity_logits, batch.y)`,
  `dom_loss = self.domain_criterion(domain_logits, batch.domain)`,
  `loss = act_loss + lambda_val * dom_loss`, then `loss.backward()`,
  `torch.nn.utils.clip_grad_norm_(..., max_grad_norm=1.0)`, `optimizer.step()`.
- `DANNTrainer.validate()` — the source-validation-only accuracy/loss computation,
  using `model.predict(batch)` (activity head only, no domain head).
- `DANNTrainer.update_checkpoint()` — the "did validation accuracy improve" check and
  the early-stopping counter (`patience=20`).
- `DANNTrainer.restore_best()` — loads back `self.best_weights`, the checkpoint from
  whichever epoch had the best validation accuracy.
- `p21/gnn_predict.py::train_and_predict()` — the outer loop calling
  `trainer.train_epoch()` then `trainer.validate()` once per epoch, up to
  `cfg["epochs"]` (100) times, with `optim.Adam(..., lr=1e-3, weight_decay=5e-4)` and
  `lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)`.

**Major components in this step:** *(DANN's loss combination and the GRL are covered
in Step 9; the remaining components here — Adam, cosine annealing, gradient clipping,
early stopping on source-validation accuracy — are standard training-stability
practices rather than contributions specific to this thesis, and are listed in full in
the chapter's Table 3.x "Training Configuration Summary.")*

---

## Step 11 — Run the Trained Graph Branch on the Held-Out Target

**Overview:** With training complete and the best checkpoint restored, the graph
branch is run, exactly once, on the target dataset's test windows — the only point in
the whole graph-branch pipeline where the held-out domain's data is used at all.

**What happens:** The model is set to evaluation mode (disabling dropout and
batch-norm's training-time behaviour) and run over every batch of the target's test
windows; only `model.predict(batch)` is called, which uses the activity classifier
head and completely ignores the domain-discriminator head (that head, and the GRL,
play no role at inference — a point the chapter states explicitly). The raw output
logits are converted into a probability distribution over the six activity classes via
softmax, and these per-window probability vectors — one `[6]`-length vector per target
test window — are what this branch ultimately hands off to the fusion step (Step 13).

**Code reference:**
- `p21/gnn_predict.py::train_and_predict()` — the final block:
  `model.eval()`, then for each `batch` in `test_loader`:
  `logits = model.predict(batch)` followed by
  `F.softmax(logits, dim=1).cpu().numpy()`, concatenated across all batches into `P`
  (shape `[N_target_test, 6]`) alongside the true labels `y`.
- `DomainAdversarialWrapper.predict(data)` — the inference-only path:
  `h = self.backbone.get_graph_embedding(data); return
  self.activity_classifier(h)` — no GRL, no domain discriminator call at all.

**Major components in this step:** *(no new component — this is the one moment where
everything built in Steps 1–10 is finally pointed at the domain that was held out the
entire time.)*

---

## Step 12 — The Independent ConvNet Complementary Branch

**Overview:** Entirely separately from the graph branch above, a second model — a
DeepConvLSTM reimplementation — is trained on its own, different view of the same
windows, and produces its own independent probability prediction.

**What happens:** Rather than the hybrid time+frequency node feature used by the
graph branch, this branch computes its own frequency-only feature for each window: a
*log*-magnitude FFT that, unlike the graph branch's feature, *keeps* the
zero-frequency bin (31 bins per channel instead of 30). This spectral view of the
whole 6-channel window is passed through four convolutional layers that mix
information across time while keeping each sensor channel's own feature maps
separate, then through two stacked LSTM layers that aggregate the resulting sequence,
finishing with a linear classifier on the final aggregated hidden state. This branch
is trained once, using the same source-only LODO discipline as the graph branch (same
five sources, same held-out target, same never-touch-target-during-training rule), and
then frozen: its predicted probabilities for the target's test windows are computed
once and saved to disk, to be reused unchanged no matter how many different graph
branch configurations are later tested against them.

**Code reference:**
- `convnet/src/models/deepconvlstm.py` (present per-phase, e.g. under
  `PROJECTS/.../Project_DA_Baseline_P17/convnet/src/models/`) — the DeepConvLSTM
  architecture itself, independent of anything in `gnn/models/`.
- `convnet_predict.py` — the training/inference script producing this branch's saved
  probabilities.
- The output lands as `results/sweep/<config>_s<seed>/convnet_<target>.npz`, holding
  `probs` (shape `[N_target_test, 6]`) and `labels` — the same shape and convention
  as the graph branch's own `gnn_<target>.npz` output from Step 11, which is exactly
  what lets Step 13 combine them directly.

**Major components in this step:**

- *DeepConvLSTM (the ConvNet branch).* Included specifically because DAGHAR's own
  benchmark paper identified convolutional-recurrent architectures as the most robust
  to cross-dataset domain shift among everything tested. Giving the overall system a
  second, architecturally very different "opinion" — one that sees a different feature
  (log-magnitude FFT with the DC bin, rather than the graph branch's raw+FFT hybrid)
  through a fundamentally different mechanism (sequential convolution+recurrence,
  rather than graph message passing) — is what makes combining the two branches
  worthwhile: two models with different blind spots correct each other on average
  more than two models that tend to fail on the same windows.

---

## Step 13 — Late Fusion: Combining the Two Branches' Predictions

**Overview:** The graph branch's and ConvNet branch's independently-computed
probability vectors for the target's test windows are combined by simple weighted
averaging into one final probability vector per window.

**What happens:** For a chosen blend weight $w \in [0,1]$, the final probability for
each window is $P = w \cdot P_{\text{gnn}} + (1-w) \cdot P_{\text{conv}}$ — literally
averaging the two branches' six-class probability vectors, weighted toward whichever
branch $w$ favours. Because both branches are already frozen (their probabilities were
each computed once, in Steps 11 and 12, and saved to `.npz` files), choosing $w$ does
not require retraining either branch — it's swept over a small fixed grid of candidate
values (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0), and whichever value produces the best mean
fused accuracy across all six LODO folds is adopted. In the final locked
configuration, this favours the graph branch somewhat ($w \approx 0.6$). This is the
one deliberate exception to the "never tune anything against the target" rule
elsewhere in the thesis (Step 1): because both branches are already finished and
frozen, and only a single blend number is being chosen (not an architecture or
hyperparameter), it's tuned directly against the aggregate six-fold result rather than
against source-validation performance — and the chapter flags this explicitly as a
limitation rather than leaving it implicit.

**Code reference:**
- `p21/run_report.py::fuse_best_w(cond)` — for one seed's results, sweeps every $w$ in
  `WEIGHTS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]`, computing
  `a = metrics_acc(w * Pg + (1 - w) * Pc, y)` for each of the six targets, and picks
  `best_w = max(WEIGHTS, key=lambda w: np.mean(sweep[w]))` — the literal weighted-sum
  fusion formula, applied per fold, then averaged.
- `metrics_acc(P, y)` — `(P.argmax(1) == y).mean()`, i.e., accuracy computed from the
  fused probability vector's most-likely class per window.

**Major components in this step:** *(the ConvNet branch and the fusion mechanism
itself are the components; both are described together in Steps 12–13 above since
"why fusion helps" only makes sense once both branches' independence has been
established.)*

---

## Step 14 — Final Activity Prediction and Reporting

**Overview:** The fused probability vector for each target window is converted into a
single predicted activity label, and the resulting fold accuracy is aggregated and
checked against the thesis's adoption rule before a configuration is considered "kept."

**What happens:** For each window, the final predicted activity is simply
`argmax` of its fused 6-class probability vector — whichever of the six activities
(sit, stand, walk, stair-up, stair-down, run) has the highest combined probability.
Averaged over all of a target dataset's test windows, this gives that fold's accuracy;
averaged again over all six LODO folds (each dataset taking its turn as the held-out
target), this gives the headline mean accuracy figure quoted throughout the thesis
(77.96% ± 0.21% across 5 seeds, for the adopted `topk9, α=0.5` configuration). Before
any new configuration change (a different α, a different top-k, a different backbone)
is adopted in place of the current best, its seed-mean fused accuracy must beat the
existing configuration by at least 1 percentage point, *and* no individual one of the
six folds may be more than 2 percentage points worse than under the existing
configuration — a check applied automatically by the reporting script, not left to
informal judgment.

**Code reference:**
- `p21/run_report.py::aggregate(seed_map)` — computes, per configuration, the
  seed-mean fused accuracy (`np.mean(means)`), its standard deviation, and per-fold
  means (`fold_mean[t]` for each of the six datasets).
- `run_report.py::main()`'s gate check —
  `keep = (dmean >= KEEP_MEAN_PP) and (worst > COLLAPSE_PP)`, where
  `KEEP_MEAN_PP = 1.0` and `COLLAPSE_PP = -2.0`, printed as `"KEEP"` or `"revert"` next
  to every candidate configuration compared against the current baseline.

**Major components in this step:**

- *The two-condition adoption gate.* Requiring both a meaningful average improvement
  *and* no badly regressed individual fold exists because a configuration can look
  good on average while quietly making the system's worst-case behaviour worse — and
  since the entire point of this thesis is robustness across genuinely different
  datasets, a change that trades overall accuracy for a much worse worst-case fold is
  arguably a step backward for the actual goal, even when its raw average technically
  improved.

---

*This walkthrough traces the graph branch's forward pass (Steps 2–11) and the
ConvNet branch (Step 12) as they are actually invoked by `p21/gnn_predict.py` and its
ConvNet counterpart, fused by `p21/run_report.py` (Step 13), for one LODO fold; the
same 14 steps repeat once per held-out target (6 times) and once per random seed (5
times) to produce the seed-mean, six-fold results reported as the thesis's main
accuracy contribution.*
