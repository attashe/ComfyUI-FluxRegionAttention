from .node import RegionAttention, CLIPDebug, FluxRegionMask, FluxRegionBBOX

NODE_CLASS_MAPPINGS = {
    "RegionAttention": RegionAttention,
    "CLIPDebug": CLIPDebug,
    "FluxRegionMask": FluxRegionMask,
    "FluxRegionBBOX": FluxRegionBBOX,
    # "BoundingBoxNode": BoundingBoxNode,
    # "VisualizeBBoxesNode": VisualizeBBoxesNode,
    # "BBoxToMaskNode": BBoxToMaskNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RegionAttention": "Region Attention",
    "FluxRegionMask": "Region Mask",
    "FluxRegionBBOX": "Region bbox",
    # "RegionPack": "Region Packing",
    "CLIPDebug": "CLIP debug",
    # "VisualizeBBoxesNode": "Visualize Bounding Boxes",
    # "BBoxToMaskNode": "Bounding Boxes to Mask",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']