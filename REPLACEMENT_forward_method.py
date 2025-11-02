"""
REPLACEMENT CODE FOR model_g3_heatmap.py forward() method
Copy this entire forward method to replace the existing one in your Model class
"""

def forward(self, x, labels=None, return_attention=False):
    """
    Forward pass with optional attention collection
    
    Args:
        x: Input audio tensor (B, T) or (B, T, 1)
        labels: Ground truth labels (B,) - optional
        return_attention: If True, collect and store attention weights
    
    Returns:
        output: Log probabilities (B, 2)
        contrastive_loss: Contrastive loss if training and labels provided
        supcon_loss: Supervised contrastive loss if training and labels provided
        utt_emb: Utterance embeddings (B, d_model)
    """
    try:
        # Extract SSL features
        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1))
        _, fullfeature = getAttenF(layerResult)  # (B, L, T, C)

        B, L, T, C = fullfeature.shape
        
        # Temporal attention over each layer
        layer_tokens = fullfeature.contiguous().view(B * L, T, C)
        layer_emb, temporal_attn = self.temporal_attn(layer_tokens)
        layer_emb = layer_emb.view(B, L, C)
        
        # Store temporal attention weights if requested
        collect_attn = return_attention or getattr(self, '_collect_attention', False)
        if collect_attn:
            self.attention_weights['temporal'] = temporal_attn.view(B, L, T).detach()

        # Intra-group attention
        groups = torch.split(layer_emb, self.group_size, dim=1)
        group_vecs = []
        intra_attns = []
        
        for g in groups:
            g_vec, intra_attn = self.intra_attn(g)
            g_vec = self.group_refine(g_vec)
            group_vecs.append(g_vec)
            if collect_attn:
                intra_attns.append(intra_attn.detach())
        
        # Store intra-group attention weights if requested
        if collect_attn and len(intra_attns) > 0:
            self.attention_weights['intra'] = torch.stack(intra_attns, dim=1)  # (B, num_groups, group_size)

        # Inter-group attention
        group_stack = torch.stack(group_vecs, dim=1)
        utt_emb, inter_attn = self.inter_attn(group_stack)
        utt_emb = self.utt_refine(utt_emb)
        
        # Store inter-group attention weights if requested
        if collect_attn:
            self.attention_weights['inter'] = inter_attn.detach()  # (B, num_groups)

        # Add gradient clipping to embeddings
        if self.training:
            utt_emb = torch.clamp(utt_emb, min=-10, max=10)

        # Get classification logits
        logits = self.classifier(utt_emb)
        log_probs = F.log_softmax(logits, dim=1)
        
        output = log_probs  # Return log probabilities
        
        # Compute contrastive losses if labels are provided and training
        contrastive_loss = None
        supcon_loss = None
        
        if self.training and labels is not None and self.use_contrastive and B > 1:
            # Project embeddings for contrastive learning
            projected_emb = self.projection_head(utt_emb)
            
            # Compute both contrastive losses
            contrastive_loss = self.contrastive_loss(projected_emb, labels)
            supcon_loss = self.supcon_loss(projected_emb, labels)
        
        return output, contrastive_loss, supcon_loss, utt_emb
        
    except Exception as e:
        print(f"❌ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        raise
