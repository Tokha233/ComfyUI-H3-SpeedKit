"""ComfyUI V3 entry point."""


async def comfy_entrypoint():
    from comfy_api.latest import ComfyExtension
    from .h3_speedkit.nodes import (H3SpeedKitEnvironment, H3SpeedKitOptimize,
        H3SpeedKitVideoVAELoader, H3SpeedKitDecode, H3SpeedKitSaveVideo, H3SpeedKitIndexedGate,
        H3SpeedKitDecodeAndSave)

    class H3SpeedKitExtension(ComfyExtension):
        async def get_node_list(self):
            return [H3SpeedKitEnvironment, H3SpeedKitOptimize,
                    H3SpeedKitVideoVAELoader, H3SpeedKitDecode, H3SpeedKitSaveVideo,
                    H3SpeedKitIndexedGate, H3SpeedKitDecodeAndSave]

    return H3SpeedKitExtension()
