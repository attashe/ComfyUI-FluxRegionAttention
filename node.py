import torch
from xformers.ops import memory_efficient_attention as xattention
import numpy as np
from torch import Tensor
from comfy.ldm.modules import attention as comfy_attention
from comfy.ldm.flux import math as flux_math
from comfy.ldm.flux import layers as flux_layers
from comfy import model_management

from PIL import Image
from typing import List, Dict, Optional
from functools import partial
from einops import rearrange

import matplotlib.pyplot as plt

orig_attention = comfy_attention.optimized_attention


def xformers_attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor,
              attn_mask: Tensor = None, q_scale=None, k_scale=None, **kwargs) -> Tensor:
    q, k = flux_math.apply_rope(q, k, pe)

    # Permute the dimensions for q, k, v
    # From: [B, H, L, D] -> [B, L, H, D]
    q = rearrange(q, "B H L D -> B L H D")
    k = rearrange(k, "B H L D -> B L H D")
    v = rearrange(v, "B H L D -> B L H D")
    
    x = xattention(q, k, v, attn_bias=attn_mask)

    x = rearrange(x, "B L H D -> B L (H D)")

    return x


def prepare_attention_mask(lin_masks: List[Image.Image], reg_embeds: List[Tensor], 
                           Nx: int, emb_size: int, emb_len: int,):
    cross_mask = torch.zeros(emb_len + Nx, emb_len + Nx)
    q_scale = torch.ones(emb_len + Nx)
    k_scale = torch.ones(emb_len + Nx)

    n_regs = len(lin_masks)
    emb_cum_idx = 0

    # Mask main prompt to subprompts
    for j in range(n_regs):
        t1, t2 = emb_cum_idx + (j+1) * emb_size, emb_cum_idx + (j+2) * emb_size
        p1, p2 = emb_cum_idx, emb_cum_idx + emb_size
        print(t1, t2, p1, p2)
        
        cross_mask[t1 : t2, p1 : p2] = 1
        cross_mask[p1 : p2, t1 : t2] = 1
        
    emb_cum_idx += emb_size

    for i, (m, emb) in enumerate(zip(lin_masks, reg_embeds)):
        # mask text
        for j in range(1, n_regs - i):
            t1, t2 = emb_cum_idx + j * emb_size, emb_cum_idx + (j+1) * emb_size
            p1, p2 = emb_cum_idx, emb_cum_idx + emb_size
            print(t1, t2, p1, p2)
            
            cross_mask[t1 : t2, p1 : p2] = 1
            cross_mask[p1 : p2, t1 : t2] = 1
        
        scale = m.sum() / Nx
        print('m: ', m.shape, scale)
        if scale > 1e-5:
            q_scale[emb_cum_idx : emb_cum_idx+emb_size] = 1 / scale
            k_scale[emb_cum_idx : emb_cum_idx+emb_size] = 1 / scale
        
        # m (4096) -> (N_text * 256 + 4096)
        m = torch.cat([torch.ones(emb_size * (n_regs+1)), m])
        print(m.shape)
        
        mb = m > 0.5
        cross_mask[~mb, emb_cum_idx : emb_cum_idx + emb_size] = 1
        cross_mask[emb_cum_idx : emb_cum_idx + emb_size, ~mb] = 1
        emb_cum_idx += emb_size

    # Image Self-Attention attention between different areas blocking
    # Calculate pairwise masks between different areas with the kronecker product
    for i in range(n_regs):
        for j in range(i+1, n_regs):
            # We need to calculate two kr.prod for preserving the symmetry of the matrix
            kron1 = torch.kron(lin_masks[i].unsqueeze(0), lin_masks[j].unsqueeze(-1))
            kron2 = torch.kron(lin_masks[j].unsqueeze(0), lin_masks[i].unsqueeze(-1))
            # cross_mask[emb_cum_idx:, emb_cum_idx:] += kron1 + kron2
            
            # We need to select interesecting regions and set the rows and columns which are intersecting to 0
            
            # Get the intersecting regions
            intersect_idx = torch.logical_and(lin_masks[i] > 0.5, lin_masks[j] > 0.5)
            # Set the intersecting regions to 0
            kron_sum = kron1 + kron2
            kron_sum[intersect_idx, :] = 0
            kron_sum[:, intersect_idx] = 0
            
            # kron_sum[intersect_idx, intersect_idx] = 0

            # Add the kronecker product to the cross mask
            cross_mask[emb_cum_idx:, emb_cum_idx:] += kron_sum
    
    # Clean up the diagonal
    cross_mask.fill_diagonal_(0)
        
    q_scale = q_scale.reshape(1, 1, -1, 1).cuda()
    k_scale = k_scale.reshape(1, 1, -1, 1).cuda()
    
    return cross_mask, q_scale, k_scale


test_payload = {
    'prompt': {
        'positive': 'An italian cafe',
        'width': 1024,
        'height': 1024,
        'bboxes': [
            {
                'caption': 'An asian man with sombrero',
                'x': 100,
                'y': 200,
                'width': 300,
                'height': 700,
            },
            {
                'caption': 'A redhair sexual woman',
                'x': 500,
                'y': 200,
                'width': 300,
                'height': 700,
            }
        ],
    }
}


def process_payload(payload):
    bboxes = payload['prompt']['bboxes']
    masks = []
    subprompts = []
    
    for i, bbox in enumerate(bboxes):
        mask = Image.new('L', (payload['prompt']['width'], payload['prompt']['height']), 0)
        mask_arr = np.array(mask)
        
        # Draw the bounding box
        mask_arr[bbox['y']:bbox['y']+bbox['height'], bbox['x']:bbox['x']+bbox['width']] = 255
        mask = Image.fromarray(mask_arr)
        
        # Debug save the mask
        mask.save(f'mask_{i}.png')
        
        masks.append(mask)
        subprompts.append(bbox['caption'])

    return masks, subprompts


def generate_test_mask(masks, height, width):
    hH, hW = int(height) // 16, int(width) // 16
    print(height, width, '->', hH, hW)
    
    lin_masks = []
    for mask in masks:
        mask = mask.convert('L')
        mask = torch.tensor(np.array(mask)).unsqueeze(0).unsqueeze(0) / 255
        # Linearize mask
        mask = torch.nn.functional.interpolate(mask, (hH, hW), mode='nearest-exact').flatten()
        lin_masks.append(mask)

    return lin_masks, hH, hW

def generate_region_mask(region, width, height):
    if region.get('bbox') is not None:
        x1, y1, x2, y2 = region['bbox']
        mask = Image.new('L', (width, height), 0)
        mask_arr = np.array(mask)

        print(f'Generating masks with {width}x{height} and [{x1}, {y1}, {x2}, {y2}]')
        
        # Draw the bounding box
        mask_arr[int(y1*height):int(y2*height), int(x1*width):int(x2*width)] = 255
        mask = Image.fromarray(mask_arr)
        
        return mask
    elif region.get('mask') is not None:
        mask = region['mask']  # ComfyUI mask is tensor (bs x height x width)
        print('MASK: ', mask)
        mask = mask[0].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask = Image.fromarray(mask)
        mask = mask.resize((width, height))
        
        return mask
    else:
        raise Exception('Unknown region type')


class RegionAttention:
    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "condition")
    FUNCTION = "go"
    CATEGORY = "model_patches"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                # "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."}),
                "condition": ("CONDITIONING",),
                "samples": ("LATENT",),
                "region1": ("REGION",),
                "enabled": ("BOOLEAN", {"default": True},),
            },
            "optional": {
                "region2": ("REGION",),
                "region3": ("REGION",),
                "region4": ("REGION",),
            }
        }

    @classmethod
    def go(cls, *, model: object, condition, samples, region1, enabled: bool,
           region2=None, region3=None, region4=None):
        print(f'Region attention Node enabled: {enabled}, model: {model}')
        # masks, payload = process_payload(test_payload)

        latent = samples['samples']
        print('latent.shape', latent.shape)
        bs_l, n_ch, lH, lW = latent.shape
        text_emb = condition[0][0].clone()
        clip_emb = condition[0][1]['pooled_output'].clone()
        bs, emb_size, emb_dim = text_emb.shape
        iH, iW = lH * 8, lW * 8

        subprompts_embeds, masks = [region1['condition'][0][0],], [generate_region_mask(region1, iW, iH),]
        masks[-1].save(f'mask_1.png')
        if region2 is not None:
            print('append region2')
            sub_emb2 = region2['condition'][0][0]
            masks.append(generate_region_mask(region2, iW, iH))
            subprompts_embeds.append(sub_emb2)
            masks[-1].save(f'mask_2.png')
        if region3 is not None:
            print('append region3')
            sub_emb3 = region3['condition'][0][0]
            masks.append(generate_region_mask(region3, iW, iH))
            subprompts_embeds.append(sub_emb3)
        if region4 is not None:
            print('append region4')
            sub_emb4 = region4['condition'][0][0]
            masks.append(generate_region_mask(region4, iW, iH))
            subprompts_embeds.append(sub_emb4)

        lin_masks, hH, hW = generate_test_mask(masks, lH * 8, lW * 8)
        Nx = int(hH * hW)
        emb_len = (len(subprompts_embeds) + 1) * emb_size
        extended_condition = torch.cat([text_emb, *subprompts_embeds], dim=1) if enabled else text_emb
        
        attn_mask, q_scale, k_scale = prepare_attention_mask(lin_masks, subprompts_embeds, Nx, emb_size, emb_len)

        # Visualize and save the attention mask
        # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # ax.imshow(attn_mask.cpu().numpy())
        # plt.savefig('attention_mask_1.png')

        # Pad mask for xformers to reduce allocations during inference
        device = torch.device('cuda')
        attn_dtype = torch.bfloat16 if model_management.should_use_bf16(device=device) else torch.float16
        if attn_mask is not None:
            print(f'Aplying attention masks: {attn_mask.shape}')
            L, _ = attn_mask.shape
            H = 24  # 24 heads for FLUX models
            pad = 8 - L % 8
            
            # print(f'Attention mask memory padded by: {pad}')
            if pad != 8:
                # TODO: take dtype from memory_management computational_type
                mask_out = torch.empty([bs, H, L + pad, L + pad],
                                       dtype=torch.bfloat16, device=device)
                mask_out[:, :, :L, :L] = attn_mask
                # print(f'Attention mask memory padded to: {mask_out.shape}')
                attn_mask = mask_out[:, :, :L, :L]
            else:
                mask_out = torch.empty([bs, H, L, L],
                                       dtype=torch.bfloat16, device=device)
                mask_out[:, :, :, :] = attn_mask
                attn_mask = mask_out

        attn_mask_bool = attn_mask > 0.5
        attn_mask.masked_fill_(attn_mask_bool, float('-inf'))
        
        attn_mask_arg: Tensor = attn_mask if enabled else None

        # if attn_mask_arg is not None:
        #     fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        #     ax.imshow(attn_mask[0][0].float().cpu().numpy())
        #     plt.savefig('attention_mask_2.png')
        
        def region_attention(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
            print(q.shape, k.shape, v.shape)
            
            res = orig_attention(q, k, v, heads, mask=attn_mask, attn_precision=attn_precision, skip_reshape=skip_reshape)

            return res
            
        # comfy_attention.optimized_attention = orig_attention if not enabled else region_attention

        def override_attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
            
            q, k = flux_math.apply_rope(q, k, pe)
        
            heads = q.shape[1]
            x = region_attention(q, k, v, heads, skip_reshape=True)
            return x

        override_attention = partial(xformers_attention, attn_mask=attn_mask_arg)
        
        flux_math.attention = override_attention
        flux_layers.attention = override_attention

        del condition
        new_condition = [[
            extended_condition,
            {'pooled_output': clip_emb},
        ]]
        
        return (model, new_condition)


class FluxRegionMask:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "condition": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("REGION",)
    FUNCTION = "create_region"

    def create_region(self, mask, condition):
        return ({
            "condition": condition,
            "mask": mask,
        },)


class FluxRegionBBOX:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "x1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "y1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "x2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "y2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "condition": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("REGION",)
    FUNCTION = "create_region"

    def create_region(self, x1, y1, x2, y2, condition):
        return ({
            "condition": condition,
            "bbox": [x1, y1, x2, y2],
        },)


class CLIPDebug:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."}),
                "condition": ("CONDITIONING",), 
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
    FUNCTION = "debug"

    def debug(self, clip, condition):
        # print(clip)
        print('len(condition)', len(condition))
        print('len(condition[0]', len(condition[0]))
        print('type(condition[0][1])', type(condition[0][1]))
        print('condition[0][0].shape', condition[0][0].shape)
        print('list(condition[0][1].keys())', list(condition[0][1].keys()))
        print("condition[0][1]['pooled_output'].shape", condition[0][1]['pooled_output'].shape)
        
        return (condition,)

import numpy as np
from PIL import Image, ImageDraw

class RegionBbox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "image_height": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "x1": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0}),
                "y1": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0}),
                "x2": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "y2": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("BBOX",)
    FUNCTION = "create_bbox"

    def create_bbox(self, image_width, image_height, x1, y1, x2, y2):
        bbox = {
            "x1": int(x1 * image_width),
            "y1": int(y1 * image_height),
            "x2": int(x2 * image_width),
            "y2": int(y2 * image_height),
        }
        return (bbox,)

class VisualizeBBoxesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bboxes": ("BBOX",),
                "color": ("COLOR", {"default": "#FF0000"}),
                "width": ("INT", {"default": 2, "min": 1, "max": 10}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize_bboxes"

    def visualize_bboxes(self, image, bboxes, color, width):
        # Convert the PyTorch tensor to a PIL Image
        pil_image = Image.fromarray((image[0].permute(1, 2, 0) * 255).byte().cpu().numpy())
        draw = ImageDraw.Draw(pil_image)

        for bbox in bboxes:
            draw.rectangle([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]], outline=color, width=width)

        # Convert back to PyTorch tensor
        tensor_image = torch.from_numpy(np.array(pil_image)).float() / 255.0
        tensor_image = tensor_image.permute(2, 0, 1).unsqueeze(0)

        return (tensor_image,)

class BBoxToMaskNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "image_height": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "bboxes": ("BBOX",),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "create_mask"

    def create_mask(self, image_width, image_height, bbox):
        mask = torch.zeros((1, image_height, image_width))

        mask[0, bbox["y1"]:bbox["y2"], bbox["x1"]:bbox["x2"]] = 1.0

        return (mask,)
