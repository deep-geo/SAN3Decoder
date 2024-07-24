from segment_anything.modeling import ImageEncoderViT, PromptEncoder, MaskDecoder
from segment_anything.modeling.sam import Sam
import torch.nn as nn
from torch.nn import LayerNorm


class SimpleTransformer(nn.Module):
    def __init__(self, transformer_dim):
        super(SimpleTransformer, self).__init__()
        self.transformer_layer = nn.Transformer(
            d_model=transformer_dim,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=2048,
            dropout=0.1,
        )

    #def forward(self, src, pos_src, tokens):
    def forward(self, src, tokens):
        return self.transformer_layer(src, tokens)


def create_three_decoder_sam(args):

    embed_dim = 256
    image_embedding_size = (args.image_size // 32, args.image_size // 32)  # Assuming image_encoder downscales by 32x
    input_image_size = (args.image_size, args.image_size)
    mask_in_chans = 16  # This is an example value; adjust according to your model requirements

    
    image_encoder = ImageEncoderViT(adapter_train=True)
    prompt_encoder = PromptEncoder(
        embed_dim=embed_dim,
        image_embedding_size=image_embedding_size,
        input_image_size=input_image_size,
        mask_in_chans=mask_in_chans,
    )  
    
    transformer_module = SimpleTransformer(256)
    # image_encoder = ImageEncoderViT(
    #     img_size=args.image_size,
    #     patch_size=16,
    #     embed_dim=768,
    #     depth=12,
    #     num_heads=12,
    #     mlp_ratio=4,
    #     qkv_bias=True,
    #     norm_layer=LayerNorm,
    # )
    
    # prompt_encoder = PromptEncoder(
    #     embed_dim=768,
    #     image_embedding_size=(args.image_size//16, args.image_size//16),
    #     input_image_size=(args.image_size, args.image_size),
    #     mask_in_chans=16,
    # )

    # transformer_module = SimpleTransformer(transformer_dim=768)
    
    segmentation_decoder = MaskDecoder(
        transformer_dim=256,
        transformer=transformer_module,  # Define or import the transformer used here
        num_multimask_outputs=1,
        activation=nn.GELU,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )
    
    normal_edge_decoder = MaskDecoder(
        transformer_dim=256,
        transformer=transformer_module,  # Define or import the transformer used here
        num_multimask_outputs=1,
        activation=nn.GELU,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )
    
    cluster_edge_decoder = MaskDecoder(
        transformer_dim=256,
        transformer=transformer_module,  # Define or import the transformer used here
        num_multimask_outputs=1,
        activation=nn.GELU,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )


    model = Sam(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        segmentation_decoder=segmentation_decoder,
        normal_edge_decoder=normal_edge_decoder,
        cluster_edge_decoder=cluster_edge_decoder,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    return model

sam_model_registry = {
    'three_decoder': create_three_decoder_sam,
    # Add other models if needed
}

# print("Registered models in sam_model_registry:")
# for key in sam_model_registry.keys():
#     print(key)