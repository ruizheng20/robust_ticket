import torch
import numpy as np
from torch import nn

# from transformers.modeling_bert import BertSelfAttention, BertSelfOutput, BertIntermediate, BertOutput

from transformers.models.bert.modeling_bert import (
    BertForSequenceClassification,
    SequenceClassifierOutput,
    BertForMaskedLM,
    BertModel,
    BertOnlyMLMHead,
    MaskedLMOutput,
    BertEncoder,
    BertEmbeddings,
    BertSelfAttention,
    BertSelfOutput,
    BertIntermediate,
    BertOutput,
    BertPooler
)

from .masked_linear import MaskedLinear, BinaryMaskedLinear
# from .bert import WordLevelBert
# from .util import use_cuda


class MaskedBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, out_w_per_mask, in_w_per_mask, mask_p):
        super().__init__(config)

        self.replace_layers_with_masked(out_w_per_mask, in_w_per_mask, mask_p)
        self.r_, self.l_, self.b_ = -0.1, 1.1, 2 / 3

        # if use_cuda:
        #     self.cuda()

    def replace_layers_with_masked(self, out_w_per_mask, in_w_per_mask, mask_p, verbose=False):
        """
        Replaces layers with their masked versions.
        out_w_per_mask: the number of output dims covered by a single mask parameter
        in_w_per_mask: the same as above but for input dims

        ex: (1,1) for weight masking
            (768,1) for neuron masking
            (768, 768) for layer masking
        """

        def replace_layers(layer_names, parent_types, replacement):
            for module_name, module in self.named_modules():
                for layer_name in layer_names:
                    if hasattr(module, layer_name) and type(module) in parent_types:
                        layer = getattr(module, layer_name)
                        setattr(module, layer_name, replacement(layer))
                        if verbose:
                            print("Replaced {} in {}".format(layer_name, module_name))

        # replace_layers(('query', 'key', 'value', 'dense'),
        #                (BertSelfAttention, BertSelfOutput, BertIntermediate, BertOutput),
        #                lambda x: MaskedLinear.from_layer(x, out_w_per_mask, in_w_per_mask, mask_p))

        replace_layers(('query', 'key', 'value', 'dense', 'pooler'),
                       (BertSelfAttention, BertSelfOutput, BertIntermediate, BertOutput, BertPooler),
                       lambda x: MaskedLinear.from_layer(x, out_w_per_mask, in_w_per_mask, mask_p))

        # if use_cuda:
        #     self.cuda()

    def compute_total_regularizer(self):
        total, n = 0, 0
        for module in self.modules():
            if hasattr(module, 'regularizer'):
                total += module.regularizer()
                n += 1
        return total / n

    def compute_binary_pct(self):

        total, n = 0, 0
        for k, v in self.named_parameters():
            if 'mask_scores' in k:
                v = v.detach().cpu().numpy().flatten()
                v = 1 / (1 + np.exp(-v))  # sigmoid
                # total += np.sum(v < 0.01) + np.sum(v > 0.99)
                total += np.sum(v < (-self.r_)/(self.l_-self.r_)) + np.sum(v > (1-self.r_)/(self.l_-self.r_))
                n += v.size
        return total / n

    def compute_half_pct(self):
        total, n = 0, 0
        for k, v in self.named_parameters():
            if 'mask_scores' in k:
                v = v.detach().cpu().numpy().flatten()
                v = 1 / (1 + np.exp(-v))  # sigmoid
                total += np.sum(v < 0.5)
                n += v.size
        return total / n

    def get_sparsity(self, layer, head):  # both are 0-indexed
        """Returns fraction of non-zero entries in q, k, and v matrices of each head."""
        with torch.no_grad():
            layer = self.bert.encoder.layer[layer].attention.self
            num_heads, dim = self.bert.config.num_attention_heads, self.bert.config.hidden_size
            # each mask is (heads * head_size, dim)
            value_mask = layer.value.produce_mask_reshaped().reshape(num_heads, dim // num_heads, dim)[head]

            if hasattr(layer.query, 'produce_mask_reshaped'):
                query_mask = layer.query.produce_mask_reshaped().reshape(num_heads, dim // num_heads, dim)[head]
                key_mask = layer.key.produce_mask_reshaped().reshape(num_heads, dim // num_heads, dim)[head]
            else:
                query_mask = value_mask
                key_mask = value_mask

            return torch.mean(query_mask), torch.mean(key_mask), torch.mean(value_mask)

    def get_sparsity_head(self, layer, head):  # both are 0-indexed
        """Returns fraction of non-zero entries in q, k, and v matrices of each head."""
        with torch.no_grad():
            layer = self.bert.encoder.layer[layer].attention.self
            num_heads, dim = self.bert.config.num_attention_heads, self.bert.config.hidden_size
            # each mask is (heads * head_size, dim)
            value_mask = layer.value.produce_mask_reshaped().reshape(num_heads, dim // num_heads, dim)[head]

            if hasattr(layer.query, 'produce_mask_reshaped'):
                query_mask = layer.query.produce_mask_reshaped().reshape(num_heads, dim // num_heads, dim)[head]
                key_mask = layer.key.produce_mask_reshaped().reshape(num_heads, dim // num_heads, dim)[head]
            else:
                query_mask = value_mask
                key_mask = value_mask

            return (torch.mean(query_mask) + torch.mean(key_mask) + torch.mean(value_mask))/3

    def get_head_mask(self, layer, head):  # both are 0-indexed
        """Returns fraction of non-zero entries in q, k, and v matrices of each head."""
        with torch.no_grad():
            layer = self.bert.encoder.layer[layer].attention.self
            num_heads, dim = self.bert.config.num_attention_heads, self.bert.config.hidden_size
            # each mask is (heads * head_size, dim)
            value_mask = layer.value.produce_mask_reshaped().reshape(num_heads, dim // num_heads, dim)[head]

            if hasattr(layer.query, 'produce_mask_reshaped'):
                query_mask = layer.query.produce_mask_reshaped().reshape(num_heads, dim // num_heads, dim)[head]
                key_mask = layer.key.produce_mask_reshaped().reshape(num_heads, dim // num_heads, dim)[head]
            else:
                query_mask = value_mask
                key_mask = value_mask

            return query_mask, key_mask, value_mask

    def get_sparsity_dense(self, layer):  # both are 0-indexed
        """Returns fraction of non-zero entries in 3 dense matrices in the layer."""
        with torch.no_grad():
            layer = self.bert.encoder.layer[layer]
            mask1 = layer.attention.output.dense.produce_mask_reshaped()
            mask2 = layer.intermediate.dense.produce_mask_reshaped()
            mask3 = layer.output.dense.produce_mask_reshaped()

            return torch.mean(mask1), torch.mean(mask2), torch.mean(mask3)

    def get_sparsity_mlp(self, layer):  # both are 0-indexed
        """Returns fraction of non-zero entries in 3 dense matrices in the layer."""
        with torch.no_grad():
            layer = self.bert.encoder.layer[layer]
            mask1 = layer.attention.output.dense.produce_mask_reshaped()
            mask2 = layer.intermediate.dense.produce_mask_reshaped()
            mask3 = layer.output.dense.produce_mask_reshaped()

            return (torch.mean(mask1)+torch.mean(mask2)+torch.mean(mask3))/3

    def get_sparsity_pooler(self):  # both are 0-indexed
        """Returns fraction of non-zero entries in 3 dense matrices in the layer."""
        with torch.no_grad():
            layer = self.bert.pooler.dense
            mask = layer.produce_mask_reshaped()
            return torch.mean(mask)

    def get_sparsity_layer(self, layer):  # both are 0-indexed
        """Returns fraction of non-zero entries in the layer."""
        with torch.no_grad():
            layer = self.bert.encoder.layer[layer]
            mask1 = layer.attention.output.dense.produce_mask_reshaped()
            mask2 = layer.intermediate.dense.produce_mask_reshaped()
            mask3 = layer.output.dense.produce_mask_reshaped()
            value_mask = layer.attention.self.value.produce_mask_reshaped()
            query_mask = layer.attention.self.query.produce_mask_reshaped()
            key_mask = layer.attention.self.key.produce_mask_reshaped()

            sparsity = torch.sum(mask1) + torch.sum(mask2) + torch.sum(mask3) + torch.sum(value_mask) + torch.sum(
                query_mask) + torch.sum(key_mask)
            sparsity = sparsity / (
                        mask1.numel() + mask2.numel() + mask3.numel() + value_mask.numel() + query_mask.numel() + key_mask.numel())

            return sparsity

    def get_all_sparsity(self):
        n = 0
        layer_sparsity = 0
        for layer in range(12):
            layer_sparsity += self.get_sparsity_layer(layer)
            n += 1

        return layer_sparsity/n

    def get_sparsity_layer_attn(self, layer):  # both are 0-indexed
        """Returns fraction of non-zero entries in the layer."""
        with torch.no_grad():
            layer = self.bert.encoder.layer[layer]
            value_mask = layer.attention.self.value.produce_mask_reshaped()
            query_mask = layer.attention.self.query.produce_mask_reshaped()
            key_mask = layer.attention.self.key.produce_mask_reshaped()

            sparsity = torch.sum(value_mask) + torch.sum(query_mask) + torch.sum(key_mask)
            sparsity = sparsity / (value_mask.numel() + query_mask.numel() + key_mask.numel())

            return sparsity

    def reset_weights(self, encoder_only=True):
        for name, module in self.named_modules():
            if hasattr(module, 'reset_parameters') and ('encoder' in name or not encoder_only):
                module.reset_parameters()

    def freeze_bert(self, freeze=True):
        for name, param in self.named_parameters():
            if 'mask_scores' not in name:
                param.requires_grad = not freeze

    def unfreeze_bert(self):
        self.freeze_bert(freeze=False)

class BinaryBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, out_w_per_mask, in_w_per_mask, mask_p):
        super().__init__(config)

        self.replace_layers_with_masked(out_w_per_mask, in_w_per_mask, mask_p)
        self.r_, self.l_, self.b_ = -0.1, 1.1, 2 / 3

        # if use_cuda:
        #     self.cuda()

    def replace_layers_with_masked(self, out_w_per_mask, in_w_per_mask, mask_p, verbose=False):
        """
        Replaces layers with their masked versions.
        out_w_per_mask: the number of output dims covered by a single mask parameter
        in_w_per_mask: the same as above but for input dims

        ex: (1,1) for weight masking
            (768,1) for neuron masking
            (768, 768) for layer masking
        """

        def replace_layers(layer_names, parent_types, replacement):
            for module_name, module in self.named_modules():
                for layer_name in layer_names:
                    if hasattr(module, layer_name) and type(module) in parent_types:
                        layer = getattr(module, layer_name)
                        setattr(module, layer_name, replacement(layer))
                        if verbose:
                            print("Replaced {} in {}".format(layer_name, module_name))

        replace_layers(('query', 'key', 'value', 'dense', 'pooler'),
                       (BertSelfAttention, BertSelfOutput, BertIntermediate, BertOutput, BertPooler),
                       lambda x: BinaryMaskedLinear.from_layer(x, out_w_per_mask, in_w_per_mask, mask_p))

        # if use_cuda:
        #     self.cuda()

    def compute_total_regularizer(self):
        total, n = 0, 0
        for module in self.modules():
            if hasattr(module, 'regularizer'):
                total += module.regularizer()
                n += 1
        return total / n

    def compute_binary_pct(self):
        total, n = 0, 0
        for k, v in self.named_parameters():
            if 'mask_scores' in k:
                v = v.detach().cpu().numpy().flatten()
                v = 1 / (1 + np.exp(-v))  # sigmoid
                # total += np.sum(v < 0.01) + np.sum(v > 0.99)
                total += np.sum(v < (-self.r_)/(self.l_-self.r_)) + np.sum(v > (1-self.r_)/(self.l_-self.r_))
                n += v.size
        return total / n

    def compute_half_pct(self):
        total, n = 0, 0
        for k, v in self.named_parameters():
            if 'mask_scores' in k:
                v = v.detach().cpu().numpy().flatten()
                v = 1 / (1 + np.exp(-v))  # sigmoid
                total += np.sum(v < 0.5)
                n += v.size
        return total / n

    def get_sparsity(self, layer, head):  # both are 0-indexed
        """Returns fraction of non-zero entries in q, k, and v matrices of each head."""
        with torch.no_grad():
            layer = self.bert.encoder.layer[layer].attention.self
            num_heads, dim = self.bert.config.num_attention_heads, self.bert.config.hidden_size
            # each mask is (heads * head_size, dim)
            value_mask = layer.value.produce_mask_reshaped().reshape(num_heads, dim // num_heads, dim)[head]

            if hasattr(layer.query, 'produce_mask_reshaped'):
                query_mask = layer.query.produce_mask_reshaped().reshape(num_heads, dim // num_heads, dim)[head]
                key_mask = layer.key.produce_mask_reshaped().reshape(num_heads, dim // num_heads, dim)[head]
            else:
                query_mask = value_mask
                key_mask = value_mask

            return torch.mean(query_mask), torch.mean(key_mask), torch.mean(value_mask)

    def get_head_mask(self, layer, head):  # both are 0-indexed
        """Returns fraction of non-zero entries in q, k, and v matrices of each head."""
        with torch.no_grad():
            layer = self.bert.encoder.layer[layer].attention.self
            num_heads, dim = self.bert.config.num_attention_heads, self.bert.config.hidden_size
            # each mask is (heads * head_size, dim)
            value_mask = layer.value.produce_mask_reshaped().reshape(num_heads, dim // num_heads, dim)[head]

            if hasattr(layer.query, 'produce_mask_reshaped'):
                query_mask = layer.query.produce_mask_reshaped().reshape(num_heads, dim // num_heads, dim)[head]
                key_mask = layer.key.produce_mask_reshaped().reshape(num_heads, dim // num_heads, dim)[head]
            else:
                query_mask = value_mask
                key_mask = value_mask

            return query_mask, key_mask, value_mask

    def get_sparsity_dense(self, layer):  # both are 0-indexed
        """Returns fraction of non-zero entries in 3 dense matrices in the layer."""
        with torch.no_grad():
            layer = self.bert.encoder.layer[layer]
            mask1 = layer.attention.output.dense.produce_mask_reshaped()
            mask2 = layer.intermediate.dense.produce_mask_reshaped()
            mask3 = layer.output.dense.produce_mask_reshaped()

            return torch.mean(mask1), torch.mean(mask2), torch.mean(mask3)

    def get_sparsity_layer(self, layer):  # both are 0-indexed
        """Returns fraction of non-zero entries in the layer."""
        with torch.no_grad():
            layer = self.bert.encoder.layer[layer]
            mask1 = layer.attention.output.dense.produce_mask_reshaped()
            mask2 = layer.intermediate.dense.produce_mask_reshaped()
            mask3 = layer.output.dense.produce_mask_reshaped()
            value_mask = layer.attention.self.value.produce_mask_reshaped()
            query_mask = layer.attention.self.query.produce_mask_reshaped()
            key_mask = layer.attention.self.key.produce_mask_reshaped()

            sparsity = torch.sum(mask1) + torch.sum(mask2) + torch.sum(mask3) + torch.sum(value_mask) + torch.sum(
                query_mask) + torch.sum(key_mask)
            sparsity = sparsity / (
                        mask1.numel() + mask2.numel() + mask3.numel() + value_mask.numel() + query_mask.numel() + key_mask.numel())

            return sparsity

    def get_sparsity_layer_attn(self, layer):  # both are 0-indexed
        """Returns fraction of non-zero entries in the layer."""
        with torch.no_grad():
            layer = self.bert.encoder.layer[layer]
            value_mask = layer.attention.self.value.produce_mask_reshaped()
            query_mask = layer.attention.self.query.produce_mask_reshaped()
            key_mask = layer.attention.self.key.produce_mask_reshaped()

            sparsity = torch.sum(value_mask) + torch.sum(query_mask) + torch.sum(key_mask)
            sparsity = sparsity / (value_mask.numel() + query_mask.numel() + key_mask.numel())

            return sparsity

    def reset_weights(self, encoder_only=True):
        for name, module in self.named_modules():
            if hasattr(module, 'reset_parameters') and ('encoder' in name or not encoder_only):
                module.reset_parameters()

    def freeze_bert(self, freeze=True):
        for name, param in self.named_parameters():
            if 'mask_scores' not in name:
                param.requires_grad = not freeze

    def unfreeze_bert(self):
        self.freeze_bert(freeze=False)