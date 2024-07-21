import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple, Type

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, sigmoid_output: bool = False) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < self.num_layers - 1:
                x = F.relu(layer(x))
            else:
                x = layer(x)

        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) for _ in range(self.num_mask_tokens)]
        )
        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth)
        self.dense_embed_conv = nn.Conv2d(768, 256, kernel_size=1)
        self.token_proj = nn.Linear(transformer_dim, transformer_dim)  # Ensure tokens have the correct dimension
        self.src_proj = nn.Linear(transformer_dim, transformer_dim)  # Ensure src has the correct dimension

    def forward(
        self,
        image_embeddings: torch.Tensor,   
        image_pe: torch.Tensor,          
        sparse_prompt_embeddings: torch.Tensor, 
        dense_prompt_embeddings: torch.Tensor,  
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        if multimask_output:
            #mask_slice = slice(1, None)
            mask_slice = slice(1, 2)
        else:
            mask_slice = slice(0, 1)
            print("2")
        #print(f"mask_slice shape: {mask_slice.shape}")
        # print(f"mask_slice : {mask_slice}")
        # print(f"masks.shape:{masks.shape}")
        
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = sparse_prompt_embeddings.size(0)
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Project tokens to the correct dimension 
        tokens = self.token_proj(tokens) # by Ray

        dense_prompt_embeddings_resized = F.interpolate(dense_prompt_embeddings, size=(image_embeddings.shape[2], image_embeddings.shape[3]), mode="bilinear", align_corners=False)
        if dense_prompt_embeddings_resized.shape[1] != image_embeddings.shape[1]:
            dense_prompt_embeddings_resized = self.dense_embed_conv(dense_prompt_embeddings_resized)

        src = image_embeddings + dense_prompt_embeddings_resized

        # Resize image_pe to match src's dimensions
        image_pe_resized = F.interpolate(image_pe, size=(src.shape[2], src.shape[3]), mode='bilinear', align_corners=False) #by ray
        #pos_src = image_pe_resized.expand_as(src)
        pos_src = image_pe_resized.expand(batch_size, -1, -1, -1)
        #pos_src = image_pe.expand_as(src)


        # Flatten and project src to the correct dimension
        src = src.flatten(2).transpose(1, 2)  # [batch_size, seq_len, embed_dim]
        src = self.src_proj(src)

        # Ensure batch sizes match
        if src.size(0) != tokens.size(0):
            raise ValueError(f"Batch size of src ({src.size(0)}) and tokens ({tokens.size(0)}) must be equal")

        # Transpose to [seq_len, batch_size, embed_dim] for transformer
        src = src.transpose(0, 1)
        tokens = tokens.transpose(0, 1)

        hs = self.transformer(src, tokens)
        hs = hs.transpose(0, 1)  # back to [batch_size, seq_len, embed_dim]

        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        src = src.transpose(0, 1).reshape(batch_size, self.transformer_dim, *image_embeddings.shape[2:])
        upscaled_embedding = self.output_upscaling(src)

        
        # hs, src = self.transformer(src, pos_src, tokens)
        # iou_token_out = hs[:, 0, :]
        # mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # src = src.transpose(1, 2).view(src.size(0), self.transformer_dim, *image_embeddings.shape[2:])
        # upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)

        b, c, h, w = upscaled_embedding.shape  
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred