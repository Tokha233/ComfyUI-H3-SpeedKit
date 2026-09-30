"""ComfyUI V3 nodes; heavy CUDA imports happen only on execution."""

import json
from comfy_api.latest import io
from .diagnostics import collect_environment


class H3SpeedKitEnvironment(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3SpeedKitEnvironment",
            display_name="H3 SpeedKit · Environment Report",
            category="H3 SpeedKit/Diagnostics",
            inputs=[io.Boolean.Input("probe_cuda", default=False)],
            outputs=[io.String.Output(display_name="environment_json")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, probe_cuda=False):
        report = json.dumps(collect_environment(probe_cuda), ensure_ascii=False, indent=2)
        return io.NodeOutput(report)


class H3SpeedKitOptimize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3SpeedKitOptimize", display_name="H3 SpeedKit · Optimize DiT (SM120)",
            category="H3 SpeedKit", inputs=[io.Model.Input("model"),
                io.Boolean.Input("enabled", default=True)], outputs=[io.Model.Output(), io.String.Output("report")])

    @classmethod
    def execute(cls, model, enabled=True):
        if not enabled:
            return io.NodeOutput(model.clone(), "H3 SpeedKit disabled")
        from .runtime import patch_model
        patched = patch_model(model)
        return io.NodeOutput(patched, "Kitchen 0.2.36 / SM120 backend installed. Execution logs report fallback reasons.")


class H3SpeedKitVideoVAELoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        import folder_paths
        return io.Schema(node_id="H3SpeedKitVideoVAELoader", display_name="H3 SpeedKit · Video VAE Loader",
            category="H3 SpeedKit", inputs=[io.Combo.Input("vae_name", options=folder_paths.get_filename_list("vae"))],
            outputs=[io.Vae.Output()])

    @classmethod
    def execute(cls, vae_name):
        import folder_paths
        from .video import load_video_vae
        return io.NodeOutput(load_video_vae(folder_paths.get_full_path_or_raise("vae", vae_name)))


class H3SpeedKitIndexedGate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3SpeedKitIndexedGate", display_name="H3 SpeedKit · Indexed Gate (Kitchen PR 219)",
            category="H3 SpeedKit/Experimental", inputs=[io.Model.Input("model"),
                io.Boolean.Input("enabled", default=True)], outputs=[io.Model.Output(), io.String.Output("report")])

    @classmethod
    def execute(cls, model, enabled=True):
        if not enabled:
            return io.NodeOutput(model.clone(), "Indexed gate disabled")
        from .indexed_gate import patch_model
        return io.NodeOutput(patch_model(model),
            "PR #219 outproj/FC2 consumer. First-use exact verification enabled; logs report fallbacks.")


class H3SpeedKitDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="H3SpeedKitDecode", display_name="H3 SpeedKit · Decode to RGB8",
            category="H3 SpeedKit", inputs=[io.Latent.Input("samples"), io.Vae.Input("vae"),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0)],
            outputs=[io.Custom("H3_RGB8").Output("frames")])

    @classmethod
    def execute(cls, samples, vae, fps=24.0):
        from .video import decode_samples
        return io.NodeOutput(decode_samples(samples, vae, fps))


class H3SpeedKitSaveVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="H3SpeedKitSaveVideo", display_name="H3 SpeedKit · Save Video",
            category="H3 SpeedKit", inputs=[io.Custom("H3_RGB8").Input("frames"),
                io.String.Input("filename_prefix", default="H3-SpeedKit"),
                io.Int.Input("crf", default=23, min=0, max=51), io.Audio.Input("audio", optional=True)],
            outputs=[io.String.Output("filename")], is_output_node=True)

    @classmethod
    def execute(cls, frames, filename_prefix="H3-SpeedKit", crf=23, audio=None):
        import folder_paths
        from pathlib import Path
        from .export import export_mp4
        folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), frames.pixels.shape[2], frames.pixels.shape[1])
        path = Path(folder) / f"{name}_{counter:05}_.mp4"
        return io.NodeOutput(export_mp4(frames, path, audio, crf=crf))
