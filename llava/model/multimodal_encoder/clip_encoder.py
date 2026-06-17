import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, CLIPTokenizer, CLIPTextModel, CLIPModel
import math
import matplotlib.pyplot as plt
import numpy as np
import json

def complement_idx(idx, dim):
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl

outputs = {}
def hook_k(module, input, output):
    outputs['desired_k'] = output

def hook_q(module, input, output):
    outputs['desired_q'] = output

def outlier_dectection(attn):
    attn_np = attn.to(dtype=torch.float32).cpu().numpy().flatten()

    Q1 = np.percentile(attn_np, 25)
    Q3 = np.percentile(attn_np, 75)
    IQR = Q3 - Q1

    # lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_indices = np.where((attn_np > upper_bound))[0]

    ratio = len(outlier_indices) / len(attn_np)
    return ratio

# LTAM algorithm
def build_token_affinity(tokens, window_size=3, w1=1.0, w2=1.0, w3=0.5):
    """
    Build token affinity kernel and calculate importance (supports batch processing)
    Args:
        tokens: Input token features [bs, 576, 1024]
        window_size: Local window size
        w1, w2: Feature and position smoothing parameters
        w3: Position term weight
    Returns:
        local_token_importance: [bs, 576]
    """
    bs, B, C = tokens.shape  # bs=batch_size, B=576, C=1024
    H = W = int(math.sqrt(B))  # H=W=24
    
    # Build position encoding P_ij (shared across all batches)
    y_coords = torch.arange(H).float().cuda()
    x_coords = torch.arange(W).float().cuda()
    y_coords = y_coords.view(-1, 1).repeat(1, W).view(-1)
    x_coords = x_coords.repeat(H)
    positions = torch.stack([y_coords, x_coords], dim=1)  # [576, 2]
    
    # Prepare result storage for each batch
    affinity = torch.zeros(bs, B, window_size * window_size).cuda()
    
    pad_size = window_size // 2
    
    # Process each batch separately
    for batch_idx in range(bs):
        # Calculate standard deviation for current batch
        sigma_feat = tokens[batch_idx].var()
        sigma_pos = positions.var()
        
        for i in range(H):
            for j in range(W):
                curr_idx = i * W + j
                # F_ij: Current position features
                curr_feat = tokens[batch_idx, curr_idx]  # [1024]
                # P_ij: Current position coordinates
                curr_pos = positions[curr_idx]  # [2]
                
                # Get local neighborhood 𝒩(i,j)
                local_indices = []
                for di in range(-pad_size, pad_size + 1):
                    for dj in range(-pad_size, pad_size + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < H and 0 <= nj < W:
                            local_indices.append(ni * W + nj)
                
                # Calculate κ_feat
                local_feats = tokens[batch_idx, local_indices]  # [n_neighbors, 1024]
                feat_diff = ((curr_feat.unsqueeze(0) - local_feats) ** 2).sum(dim=1)
                kappa_feat = -feat_diff / (w1 * sigma_feat)
                
                # Calculate κ_pos
                local_pos = positions[local_indices]  # [n_neighbors, 2]
                pos_diff = ((curr_pos.unsqueeze(0) - local_pos) ** 2).sum(dim=1)
                kappa_pos = -pos_diff / (w2 * sigma_pos)
                
                # Calculate combined affinity kernel κ
                kappa = kappa_feat + w3 * kappa_pos
                
                # Fill results
                affinity[batch_idx, curr_idx, :len(local_indices)] = kappa
    
    # Calculate token importance I_ij, remove last dimension
    local_token_importance = affinity.mean(dim=2)  # [bs, 576]
    local_token_importance = torch.softmax(local_token_importance, dim=-1)
    
    return local_token_importance

# Adaptive Variance-based Weighting
def combine_importance(cls_attn, local_token_importance, method='weighted_sum', alpha=0.5):
    """
    Combine global attention and local importance.
    Args:
        cls_attn: Global attention scores [bs, 576]
        local_token_importance: Local importance scores [bs, 576]
        method: Combination method ['weighted_sum', 'geometric', 'max', 'adaptive']
        alpha: Weight of global importance (0~1)
    Returns:
        combined_importance: Combined importance scores [bs, 576]
    """
    # 1. Normalize both metrics to the same scale
    cls_attn_norm = torch.softmax(cls_attn, dim=-1)
    local_imp_norm = torch.softmax(local_token_importance, dim=-1)
    # 'weighted_sum' 'geometric' 'max' 'harmonic' 'adaptive'
    if method == 'weighted_sum':
        # Simple weighted sum
        return alpha * cls_attn_norm + (1 - alpha) * local_imp_norm
    elif method == 'geometric':
        # Geometric mean, can better balance the two metrics
        return torch.sqrt(cls_attn_norm * local_imp_norm)
    elif method == 'max':
        # Take the larger value of the two metrics, emphasizing the upper bound of importance
        return torch.maximum(cls_attn_norm, local_imp_norm)
    elif method == 'harmonic':
        # Harmonic mean: more sensitive to extreme values
        return 2 * (cls_attn_norm * local_imp_norm) / (cls_attn_norm + local_imp_norm + 1e-8)
    elif method == 'adaptive':
        # Adaptive weighting: dynamically adjust weights based on the variance of the two metrics
        cls_var = cls_attn_norm.var(dim=-1, keepdim=True)
        local_var = local_imp_norm.var(dim=-1, keepdim=True)
        # The smaller the variance, the more certain the metric is, and the greater the weight should be
        cls_weight = local_var / (cls_var + local_var)
        local_weight = cls_var / (cls_var + local_var)
        return cls_weight * cls_attn_norm + local_weight * local_imp_norm
    else:
        raise ValueError(f"Unknown method: {method}")
    
class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer # -2 22
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')


        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        if device_map == 'auto':
            self.clip_model = CLIPModel.from_pretrained(self.vision_tower_name)
            self.clip_model.to(self.vision_tower.device)
        else:
            self.clip_model = CLIPModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        
        self.text_tokenizer = CLIPTokenizer.from_pretrained(self.vision_tower_name)
        self.text_encoder = CLIPTextModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.text_encoder.requires_grad_(False)
    
        if device_map:
            self.text_encoders = self.text_encoder.to(self.vision_tower.device)

        self.is_loaded = True

    def feature_select(self, image_forward_outs, images=None, is_two_parts=False, get_last_layer_with_cls=False):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        # image_features = image_forward_outs.hidden_states[-1]
        if self.select_feature == 'patch':
            cls_token = image_features[:, 0:1]
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        if get_last_layer_with_cls:
            return image_features, image_forward_outs.last_hidden_state
        return image_features, cls_token
    
    
    def token_prune_merge_advanced_plus(self, images, if_adaptive=True, reduction_ratio = 1/8, token_num=-1):
        if_adaptive = True

        #set hooks for extracting desired layer's k and q
        hook_handle_k = self.vision_tower.vision_model.encoder.layers[23].self_attn.k_proj.register_forward_hook(hook_k)
        hook_handle_q = self.vision_tower.vision_model.encoder.layers[23].self_attn.q_proj.register_forward_hook(hook_q)

        #forward pass
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        cls_token_last_layer =image_forward_outs.hidden_states[self.select_layer][:, 0:1]
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        B, N, C = image_features.shape

        #extract desired layer's k and q and remove hooks; calculate attention
        desired_layer_k = outputs["desired_k"]
        desired_layer_q = outputs["desired_q"]

        hook_handle_k.remove()
        hook_handle_q.remove()

        attn = (desired_layer_q @ desired_layer_k.transpose(-2, -1)) * C ** -0.5
        attn = F.softmax(attn, dim=-1)

        cls_attn = attn[:, 0, 1:]
        if token_num == 64:
            compress_num = 13
            step_length = 33
            _, idx = torch.topk(cls_attn, compress_num, dim=1, largest=True)  # [B, left_tokens] , sorted=True
            reduction_ratio = compress_num / 576
        elif token_num == 128:
            compress_num = 40
            step_length = 18
            _, idx = torch.topk(cls_attn, compress_num, dim=1, largest=True)  # [B, left_tokens] , sorted=True
            reduction_ratio = compress_num / 576
        else:
            compress_num = 95
            step_length = 9
            _, idx = torch.topk(cls_attn, compress_num, dim=1, largest=True)  # [B, left_tokens] , sorted=True
            reduction_ratio = compress_num / 576
            
            
            
        if if_adaptive:
            # step_length = int(1/reduction_ratio)
            arithmetic_sequence = torch.arange(0, 575, int(step_length/3)).to(device=self.device)
            original_tensor_1d = idx.flatten().to(device=self.device)
            filtered_sequence = torch.tensor([x for x in arithmetic_sequence if x not in original_tensor_1d]).to(device=self.device)
            concatenated_tensor = torch.cat((idx, filtered_sequence.unsqueeze(0)), dim=1)
            idx = concatenated_tensor
        else:
            # this is for training
            step_length = int(1/reduction_ratio)
            new_idx = torch.zeros((idx.size(0), idx.size(1)*2), dtype=torch.long).to(device=self.device)
            for i in range(idx.size(0)):
                arithmetic_sequence = torch.arange(int(step_length/2), 575, int(step_length)).to(device=self.device)
                original_tensor_1d = idx[i].flatten().to(device=self.device)
                filtered_sequence = arithmetic_sequence
                # filtered_sequence = torch.tensor([x for x in arithmetic_sequence if x not in original_tensor_1d]).to(device=self.device)
                concatenated_tensor = torch.cat((original_tensor_1d, filtered_sequence), dim=0)
                new_idx[i] = concatenated_tensor
            idx = new_idx

        index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

        Key_wo_cls = desired_layer_k[:, 1:]  # [B, N-1, C]

        x_others = torch.gather(image_features, dim=1, index=index)  # [B, left_tokens, C]
        x_others_attn = torch.gather(cls_attn, dim=1, index=idx)  
        Key_others = torch.gather(Key_wo_cls, dim=1, index=index)  # [B, left_tokens, C]
        compl = complement_idx(idx, N)  # [B, N-1-left_tokens]
        non_topk = torch.gather(image_features, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]
        non_topk_Key = torch.gather(Key_wo_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))
        non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]

        Key_others_norm = F.normalize(Key_others, p=2, dim=-1)
        non_topk_Key_norm = F.normalize(non_topk_Key, p=2, dim=-1)
        B, left_tokens, C = x_others.size()
        updated_x_others = torch.zeros_like(x_others)

        for b in range(B):
            for i in range(left_tokens):
                key_others_norm = Key_others_norm[b,i,:].unsqueeze(0).unsqueeze(0)

                before_i_Key = Key_others_norm[b, :i, :].unsqueeze(0)  
                after_i_Key = Key_others_norm[b, i+1:, :].unsqueeze(0) 

                before_i_x_others = x_others[b, :i, :].unsqueeze(0)  
                after_i_x_others = x_others[b, i+1:, :].unsqueeze(0)   
                rest_x_others = torch.cat([before_i_x_others, after_i_x_others, non_topk[b,:,:].unsqueeze(0)], dim=1)   
                before_i_x_others_attn = x_others_attn[b, :i].unsqueeze(0)  
                after_i_x_others_attn = x_others_attn[b, i+1:].unsqueeze(0)  
                rest_x_others_attn = torch.cat([before_i_x_others_attn, after_i_x_others_attn, non_topk_attn[b,:].unsqueeze(0)], dim=1)  

                rest_Keys = torch.cat([before_i_Key, after_i_Key, non_topk_Key_norm[b,:,:].unsqueeze(0)], dim=1)
                cos_sim_matrix = torch.bmm(key_others_norm, rest_Keys.transpose(1, 2))

                _, cluster_indices = torch.topk(cos_sim_matrix, k=int(32), dim=2, largest=True)

                cluster_tokens = rest_x_others[:,cluster_indices.squeeze(),:]
                weights = rest_x_others_attn[:,cluster_indices.squeeze()].unsqueeze(-1)

                # update cluster centers
                weighted_avg = torch.sum(cluster_tokens * weights, dim=1) #/ torch.sum(weights)
                updated_center = x_others[b, i, :]  + weighted_avg 
                updated_x_others[b, i, :] = updated_center 
            
        extra_one_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
        updated_x_others = torch.cat([updated_x_others, extra_one_token],dim=1)
        image_features = updated_x_others
        return image_features.to(torch.float16)
    
    # the Text-Guided Vision Complement (TGVC) module
    # def text_guided_vision_complement(self, images, text_features, token_num=23, vtc_times=1):
    #     # Encode text and obtain text features
    #     text_inputs = self.text_tokenizer(original_qs, padding=True, truncation=True, return_tensors="pt").to(self.device)
    #     text_features = self.text_encoder(**text_inputs).last_hidden_state
    #     text_features_projection = self.clip_model.text_projection(text_features)

    #     image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True)

    #     _, idx = torch.topk(metric, token_num, dim=1, largest=True)  # bs, token_num, sorted=True
    #     select_idx = torch.sort(idx, dim=1)[0]  # bs, token_num
     
    #     inverse_mask = torch.zeros((batch_size, image_features.size(1)), dtype=torch.bool, device=image_features.device)
    #     # Process each batch separately
    #     for b in range(batch_size):
    #         inverse_mask[b].scatter_(0, select_idx[b], True)
    #     inverse_mask = ~inverse_mask  # (B, 576)

    #     selected_image_features = torch.stack([image_features[b, select_idx[b]] for b in range(batch_size)])  # (B, token_num, 1024)
    #     # print(f"the shape of selected_image_features is {selected_image_features.shape}")
    #     other_image_features = torch.stack([image_features[b, inverse_mask[b]] for b in range(batch_size)])  # (B, 576-token_num, 1024)

    #     all_image_features_projection = self.clip_model.visual_projection(image_features)  # 1024->768
    #     all_image_features_norm = F.normalize(all_image_features_projection, dim=-1)
    #     other_image_features_projection = self.clip_model.visual_projection(other_image_features)
    #     other_image_features_norm = F.normalize(other_image_features_projection, dim=-1)
        
    #     text_features_norm = F.normalize(text_features_projection, dim=-1)
    #     text_to_image_similarity = torch.matmul(text_features_norm, other_image_features_norm.transpose(-2, -1)) * self.clip_model.logit_scale.exp()

    #     token_scores = text_to_image_similarity.mean(dim=1)  # (B, 576-token_num, text_token_num) -> (B, 576-token_num)
       
    #     complement_num = 0

    #     if complement_num > 0:
    #         _, idx = torch.topk(token_scores, complement_num, dim=1, largest=True)
    #         other_idx = torch.sort(idx, dim=1)[0]
    #         target_hidden = torch.stack([other_image_features[b, other_idx[b]] for b in range(other_image_features.size(0))])
    #         hidden_to_merge = torch.stack([other_image_features[b, ~torch.isin(
    #             torch.arange(other_image_features.size(1), device=other_image_features.device), 
    #             other_idx[b]
    #         )] for b in range(other_image_features.size(0))])
            
    #         vtc_times = 3
    #         for iteration in range(vtc_times):
    #             target_hidden_projection = self.clip_model.visual_projection(target_hidden)
    #             hidden_to_merge_projection = self.clip_model.visual_projection(hidden_to_merge)
    #             current_target_norm = F.normalize(target_hidden_projection, dim=-1)
    #             current_image2text_similarity = torch.matmul(current_target_norm, text_features_norm.transpose(-2, -1)) * self.clip_model.logit_scale.exp()
    #             current_text2image_similarity = torch.matmul(text_features_norm, current_target_norm.transpose(-2, -1)) * self.clip_model.logit_scale.exp()
    #             hidden_to_merge_norm = F.normalize(hidden_to_merge_projection, dim=-1)
    #             hidden_image2text_similarity = torch.matmul(hidden_to_merge_norm, text_features_norm.transpose(-2, -1)) * self.clip_model.logit_scale.exp()
    #             metric_1 = torch.matmul(hidden_image2text_similarity, current_text2image_similarity)
    #             metric_normalized = metric_1 / metric_1.norm(dim=-1, keepdim=True)
    #             assign_one_hot = torch.zeros(hidden_to_merge.shape[0], hidden_to_merge.shape[1], complement_num, dtype=other_image_features.dtype, device=metric_normalized.device)
    #             assign_one_hot.scatter_(2, metric_normalized.argmax(dim=2).unsqueeze(-1), 1)
    #             counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
    #             aggregated_hidden = torch.matmul(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts
    #             target_hidden = target_hidden + aggregated_hidden
            
    #         complement_tokens = target_hidden
    #         final_image_features = torch.cat([selected_image_features, complement_tokens], dim=1).to(images.dtype)
    #     else:
    #         final_image_features = selected_image_features

    #     return final_image_features
    
    def token_prune_visiontrim(self, images, if_adaptive=True, csa=False, DVTS_token_num=48, TGVC_token_num=16, start_layer=None, dataset_name=None, original_qs=None, vtc_times=1):
        
        # Ensure text encoder and tokenizer are loaded
        if original_qs is not None and (self.text_encoder is None or self.text_tokenizer is None):
            raise RuntimeError("Text encoder and tokenizer not loaded. Call load_model() first.")
        if original_qs is not None:
            # Encode text and obtain text features
            text_inputs = self.text_tokenizer(original_qs, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                text_features = self.text_encoder(**text_inputs).last_hidden_state
                text_features_projection = self.clip_model.text_projection(text_features)
        
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True)
        
        # the Dominant Vision Token Selection (DVTS) module
        image_features, cls_token = self.feature_select(image_forward_outs)  # Output hidden_states of the second-to-last layer of ViT
        image_features = image_features.to(images.dtype)
        batch_size = image_features.size(0)

        local_token_importance = build_token_affinity(image_features)
        last_layer_attn = image_forward_outs.attentions[-2]  # bs, heads, 577, 577
        last_layer_attn = last_layer_attn.mean(dim=1)  # bs, 577, 577

        # 1. Extract attention matrix between image tokens
        image_attn = last_layer_attn[:, 1:, 1:]  # [bs, 576, 576]
        # Create mask to exclude diagonal (self-attention)
        bs, n_tokens = image_attn.shape[0], image_attn.shape[1]
        mask = torch.eye(n_tokens, dtype=torch.bool, device=image_attn.device)
        masked_attn = image_attn.masked_fill(mask, 0)
        # 2. Calculate average and total attention received by each image token
        avg_received_attn = masked_attn.sum(dim=1) / (n_tokens - 1)  # [bs, 576]
        sum_received_attn = masked_attn.sum(dim=1)  # [bs, 576]

        # Calculate average and total attention allocated by each image token to other tokens
        avg_sent_attn = masked_attn.sum(dim=2) / (n_tokens - 1)
        sum_sent_attn = masked_attn.sum(dim=2)
        cls_attn = last_layer_attn[:, 0, 1:]  # bs, 576
        cls_attn_multi_layers = [attn.mean(dim=1)[:, 0, 1:] for attn in image_forward_outs.attentions[start_layer:-1]]  # [(bs, 576)...]
        cls_attn_sum = torch.sum(torch.stack(cls_attn_multi_layers), dim=0)  # can be modified to operate min or max

        # Normalize
        cls_attn_norm = F.normalize(cls_attn, dim=-1)
        avg_received_attn_norm = F.normalize(avg_received_attn, dim=-1)
        metric = cls_attn
        # 'weighted_sum' 'geometric' 'max' 'harmonic' 'adaptive'
        metric = combine_importance(cls_attn, local_token_importance, method='adaptive', alpha=0.5)    
        
        # Weighted combination, adjustable weight
        alpha = 0.5  # Adjustable weight parameter
        metric = alpha * cls_attn_norm + (1 - alpha) * avg_received_attn_norm

        _, idx = torch.topk(metric, DVTS_token_num, dim=1, largest=True)  # bs, token_num, sorted=True
        select_idx = torch.sort(idx, dim=1)[0]  # bs, token_num
        inverse_mask = torch.zeros((batch_size, image_features.size(1)), dtype=torch.bool, device=image_features.device)
        # Process each batch separately
        for b in range(batch_size):
            inverse_mask[b].scatter_(0, select_idx[b], True)
        inverse_mask = ~inverse_mask  # (B, 576)
        selected_image_features = torch.stack([image_features[b, select_idx[b]] for b in range(batch_size)])  # (B, token_num, 1024)
        other_image_features = torch.stack([image_features[b, inverse_mask[b]] for b in range(batch_size)])  # (B, 576-token_num, 1024)
        all_image_features_projection = self.clip_model.visual_projection(image_features)  # 1024->768
        all_image_features_norm = F.normalize(all_image_features_projection, dim=-1)
        other_image_features_projection = self.clip_model.visual_projection(other_image_features)
        other_image_features_norm = F.normalize(other_image_features_projection, dim=-1)
        
        text_features_norm = F.normalize(text_features_projection, dim=-1)
        # Calculate similarity between text and remaining tokens
        image_to_text_similarity = torch.matmul(other_image_features_norm, text_features_norm.transpose(-2, -1)) * self.clip_model.logit_scale.exp()
        text_to_image_similarity = torch.matmul(text_features_norm, other_image_features_norm.transpose(-2, -1)) * self.clip_model.logit_scale.exp()
        token_scores = text_to_image_similarity.mean(dim=1)  # (B, 576-token_num, text_token_num) -> (B, 576-token_num)
        complement_num = TGVC_token_num
        
        # the Text-Guided Vision Complement (TGVC) module
        if complement_num > 0:
            _, idx = torch.topk(token_scores, complement_num, dim=1, largest=True)
            other_idx = torch.sort(idx, dim=1)[0]
            target_hidden = torch.stack([other_image_features[b, other_idx[b]] for b in range(other_image_features.size(0))])
            hidden_to_merge = torch.stack([other_image_features[b, ~torch.isin(
                torch.arange(other_image_features.size(1), device=other_image_features.device), 
                other_idx[b]
            )] for b in range(other_image_features.size(0))])
            
            vtc_times = 3
            for iteration in range(vtc_times):
                target_hidden_projection = self.clip_model.visual_projection(target_hidden)
                hidden_to_merge_projection = self.clip_model.visual_projection(hidden_to_merge)
                current_target_norm = F.normalize(target_hidden_projection, dim=-1)
                current_image2text_similarity = torch.matmul(current_target_norm, text_features_norm.transpose(-2, -1)) * self.clip_model.logit_scale.exp()
                current_text2image_similarity = torch.matmul(text_features_norm, current_target_norm.transpose(-2, -1)) * self.clip_model.logit_scale.exp()
                hidden_to_merge_norm = F.normalize(hidden_to_merge_projection, dim=-1)
                hidden_image2text_similarity = torch.matmul(hidden_to_merge_norm, text_features_norm.transpose(-2, -1)) * self.clip_model.logit_scale.exp()
                metric_1 = torch.matmul(hidden_image2text_similarity, current_text2image_similarity)
                metric_normalized = metric_1 / metric_1.norm(dim=-1, keepdim=True)
                assign_one_hot = torch.zeros(hidden_to_merge.shape[0], hidden_to_merge.shape[1], complement_num, dtype=other_image_features.dtype, device=metric_normalized.device)
                assign_one_hot.scatter_(2, metric_normalized.argmax(dim=2).unsqueeze(-1), 1)
                counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
                aggregated_hidden = torch.matmul(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts
                target_hidden = target_hidden + aggregated_hidden
            
            complement_tokens = target_hidden
            final_image_features = torch.cat([selected_image_features, complement_tokens], dim=1).to(images.dtype)
        else:
            final_image_features = selected_image_features

        return cls_attn, final_image_features
    
    @torch.no_grad()
    def forward(self, images, method="none", dataset_name="none", start_layer=None, get_last_layer_with_cls=False, DVTS_token_num=48, TGVC_token_num=16, original_qs=None):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            if method == "VisionTrim":
                cls_attn, image_features = self.token_prune_visiontrim(images, if_adaptive=True, DVTS_token_num=DVTS_token_num, TGVC_token_num=TGVC_token_num, start_layer=start_layer, dataset_name=dataset_name, original_qs=original_qs)
                return cls_attn, image_features
            elif method == "llava_prumerge":
                image_features = self.token_prune_merge_advanced_plus(images, if_adaptive=DVTS_token_num, reduction_ratio=1/8, token_num=DVTS_token_num)
            else:
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                image_features = self.feature_select(images=images, image_forward_outs=image_forward_outs).to(images.dtype)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
