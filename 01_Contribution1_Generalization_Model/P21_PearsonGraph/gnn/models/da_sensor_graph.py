"""
Domain-Adversarial Wrapper (Phase 2)
=====================================
Wraps ANY backbone model (SensorGraphModel or CNNOnlyBaseline) with a
domain discriminator head and gradient reversal for domain-invariant
representation learning.

Architecture:
  Backbone (CNN+GNN or CNNOnly) → graph embedding h
                                      |
                       +--------------+--------------+
                       v                             v
               Activity Classifier          GRL → Domain Discriminator
               FC → 6 classes               FC → K source datasets
               (classification loss)        (adversarial loss)

  Total loss = L_activity + lambda * L_domain
  (GRL reverses gradients on the domain branch, so minimizing L_domain
   w.r.t. discriminator + maximizing it w.r.t. encoder)
"""

import torch
import torch.nn as nn

from utils.gradient_reversal import GradientReversalLayer


class DomainAdversarialWrapper(nn.Module):
    """
    DANN wrapper for any backbone that implements get_graph_embedding().

    Works with both SensorGraphModel and CNNOnlyBaseline.
    """

    def __init__(self, backbone, num_domains, config):
        super().__init__()

        self.config = config
        self.num_domains = num_domains
        domain_hidden = config.get("dann_domain_hidden", 64)

        # The backbone produces graph-level embeddings
        self.backbone = backbone

        # Remove the backbone's own classifier — we replace it
        if hasattr(self.backbone, 'classifier'):
            self._backbone_type = type(self.backbone).__name__
            num_classes = config.get("num_classes", 6)

            if self._backbone_type == "CNNOnlyBaseline":
                # CNNOnly's get_graph_embedding() returns dim = cnn_embed_dim
                graph_embed_dim = config.get("cnn_embed_dim", 64)
                self.activity_classifier = nn.Linear(graph_embed_dim, num_classes)
            else:
                # SensorGraphModel's get_graph_embedding() returns dim = gnn_hidden
                graph_embed_dim = config.get("gnn_hidden", 64)
                self.backbone.classifier = nn.Identity()
                self.activity_classifier = nn.Linear(graph_embed_dim, num_classes)
        else:
            raise ValueError("Backbone must have a 'classifier' attribute")

        # Domain discriminator head (uses same embedding dim as activity head)
        self.grl = GradientReversalLayer()
        self.domain_discriminator = nn.Sequential(
            nn.Linear(graph_embed_dim, domain_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(domain_hidden, num_domains),
        )

    def forward(self, data, lambda_val=1.0):
        """
        Forward pass with both heads.

        Returns:
            activity_logits: [batch_size, num_classes]
            domain_logits:   [batch_size, num_domains]
        """
        # Get graph-level embeddings from backbone
        h = self.backbone.get_graph_embedding(data)  # [batch_size, embed_dim]

        # Activity prediction (normal gradient flow)
        activity_logits = self.activity_classifier(h)

        # Domain prediction (reversed gradient flow)
        h_reversed = self.grl(h, lambda_val)
        domain_logits = self.domain_discriminator(h_reversed)

        return activity_logits, domain_logits

    def forward_features(self, data, lambda_val=1.0):
        """P18b — like forward(), but also returns the graph embedding h so the
        trainer can apply a source↔target alignment loss (CORAL/MMD) on it.

        Returns:
            activity_logits [B, num_classes], domain_logits [B, num_domains], h [B, d]
        """
        h = self.backbone.get_graph_embedding(data)
        activity_logits = self.activity_classifier(h)
        domain_logits = self.domain_discriminator(self.grl(h, lambda_val))
        return activity_logits, domain_logits, h

    def embed(self, data):
        """Graph embedding only (for the target-unlabeled alignment branch)."""
        return self.backbone.get_graph_embedding(data)

    def predict(self, data):
        """Inference mode: only activity prediction."""
        h = self.backbone.get_graph_embedding(data)
        return self.activity_classifier(h)
