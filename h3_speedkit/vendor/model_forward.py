# SPDX-License-Identifier: GPL-3.0-or-later
# Derived from ComfyUI MiniMaxH3Model._forward, file SHA 26ae72f5834cc9038a38600cd12c72d28da785cdaacd578d5b042b58ad48b09b.
# Only change: release eleven dead embedding temporaries before time embedding.
# This file is source, not runtime source rewriting.
import torch
import comfy
from comfy.ldm.minimax.model import (PackedLayout, time_shift_sigma, VISUAL_COND_TIMESTEP,
    AUDIO_COND_TIMESTEP, mask_row_values, patchify_video, pack_audio,
    rope_rotation_table, unpatchify_video, unpack_audio)

def _forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, denoise_mask=None, audio_denoise_mask=None, **kwargs):
    video_x, audio_x = (x[0], x[1])
    orig_t, orig_h, orig_w = (video_x.shape[2], video_x.shape[3], video_x.shape[4])
    video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, self.patch_size)
    if video_x.shape[0] != 1:
        raise ValueError('MiniMax H3 supports batch size 1')
    payload = minimax_payload or {}
    device = video_x.device
    dtype = context.dtype
    latent_t, lat_h, lat_w = (video_x.shape[2], video_x.shape[3], video_x.shape[4])
    audio_t = audio_x.shape[-1]
    text_len = context.shape[1]
    layout = payload.get('layout')
    if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
        layout = PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t, keyframes=payload.get('keyframes'), refs=payload.get('refs'))
    shift_v = float(transformer_options.get('minimax_h3_sigma_shift_video', self.sigma_shift_video))
    shift_a = float(transformer_options.get('minimax_h3_sigma_shift_audio', self.sigma_shift_audio))
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-06)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))
    vis_aug = float(payload.get('visual_cond_noise_aug', VISUAL_COND_TIMESTEP))
    aud_aug = float(payload.get('audio_cond_noise_aug', AUDIO_COND_TIMESTEP))
    seg_t = {'text': t_v, 'video': t_v, 'audio': t_a, 'cond': max(t_v, vis_aug), 'ref_img': max(t_v, vis_aug), 'cond_audio': max(t_a, aud_aug), 'ref_audio': max(t_a, aud_aug)}
    t_pin_v = max(t_v, VISUAL_COND_TIMESTEP)
    t_pin_a = max(t_a, AUDIO_COND_TIMESTEP)
    video_rows_t = None
    audio_rows_t = None
    if denoise_mask is not None:
        m = mask_row_values(denoise_mask[0, 0].to(torch.float32), latent_t, lat_h, lat_w)
        if m is not None:
            rows_t = (1.0 - m * sigma_v.to(m.device)).clamp(max=t_pin_v)
            if rows_t.unique().numel() == 1:
                seg_t['video'] = float(rows_t[0])
            else:
                video_rows_t = rows_t
    if audio_denoise_mask is not None:
        m = audio_denoise_mask[0, 0].to(torch.float32).reshape(-1)
        if not bool((m >= 1.0 - 0.001).all()):
            sigma_a = 1.0 - t_a
            rows_t = (1.0 - m * sigma_a).clamp(max=t_pin_a)
            if rows_t.unique().numel() == 1:
                seg_t['audio'] = float(rows_t[0])
            else:
                audio_rows_t = rows_t
    unique_t = sorted({t_v, t_a} | {seg_t[k] for _, _, k in layout.segments} | (set(video_rows_t.unique().tolist()) if video_rows_t is not None else set()) | (set(audio_rows_t.unique().tolist()) if audio_rows_t is not None else set()))
    t_row = {t: i for i, t in enumerate(unique_t)}
    seg_tag = {'text': 1, 'video': 0, 'audio': 2, 'cond': 0, 'ref_img': 0, 'cond_audio': 2, 'ref_audio': 2}

    def rows_to_mod_index(rows_t, tag):
        levels = rows_t.unique()
        base = torch.tensor([t_row[v] * 3 + tag for v in levels.tolist()], dtype=torch.long, device=rows_t.device)
        return base[torch.searchsorted(levels, rows_t)]
    text_tags = payload.get('text_token_tags')
    mod_segments = []
    for a, b, kind in layout.segments:
        row_base = t_row[seg_t[kind]] * 3
        if kind == 'text' and text_tags is not None:
            tags = text_tags.view(-1).tolist()
            run_start = 0
            for i in range(1, b - a + 1):
                if i == b - a or tags[i] != tags[run_start]:
                    mod_segments.append((a + run_start, a + i, row_base + int(tags[run_start])))
                    run_start = i
        elif kind == 'video' and video_rows_t is not None:
            mod_segments.append((a, b, rows_to_mod_index(video_rows_t, seg_tag[kind])))
        elif kind == 'audio' and audio_rows_t is not None:
            mod_segments.append((a, b, rows_to_mod_index(audio_rows_t, seg_tag[kind])))
        else:
            mod_segments.append((a, b, row_base + seg_tag[kind]))
    img_update = layout.img_update.to(device)
    audio_update = layout.audio_update.to(device)
    video_rows = patchify_video(video_x.to(torch.float32), self.patch_size)
    audio_rows = pack_audio(audio_x.to(torch.float32))
    cond_video_rows = self._cond_video_rows(payload, device)
    cond_audio_rows = self._cond_audio_rows(payload, device)
    all_video_rows = video_rows
    if cond_video_rows is not None:
        all_video_rows = torch.empty(img_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device)
        all_video_rows[~img_update] = cond_video_rows
        all_video_rows[img_update] = video_rows
    all_audio_rows = audio_rows
    if cond_audio_rows is not None:
        all_audio_rows = torch.empty(audio_update.shape[0], audio_rows.shape[1], dtype=torch.float32, device=device)
        all_audio_rows[~audio_update] = cond_audio_rows
        all_audio_rows[audio_update] = audio_rows
    video_embed = self.video_patch_proj(all_video_rows).to(dtype)
    audio_embed = self.audio_patch_proj(all_audio_rows).to(dtype)
    text_states = context[0]
    if text_states.shape[-1] != self.hidden_size:
        text_states = self.token_refiner(self.condition_proj(text_states), transformer_options=transformer_options)
    h = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
    voff = aoff = 0
    for a, b, kind in layout.segments:
        n = b - a
        if kind == 'text':
            h[a:b] = text_states
        elif kind in ('cond', 'ref_img', 'video'):
            h[a:b] = video_embed[voff:voff + n]
            voff += n
        else:
            h[a:b] = audio_embed[aoff:aoff + n]
            aoff += n
    del video_embed, audio_embed, all_video_rows, all_audio_rows, video_rows, audio_rows, cond_video_rows, cond_audio_rows, img_update, audio_update, text_states
    t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
    if self.use_adaln_curves:
        table = comfy.model_management.cast_to(self.adaln_t_table, device=device)
        pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
        t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
    else:
        t_emb = self.time_embedder(t_vals).to(dtype)
    rope_freqs = rope_rotation_table(self.rope_freqs(layout.position_ids, device), dtype)
    patches_replace = transformer_options.get('patches_replace', {})
    blocks_replace = patches_replace.get('dit', {})
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks), device, transformer_options)
    for i, block in enumerate(self.blocks):
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block)
        if ('double_block', i) in blocks_replace:

            def block_wrap(args):
                return {'img': block(args['img'], args['t_emb'], args['mod_segments'], args['rope_freqs'], transformer_options=args['transformer_options'])}
            h = blocks_replace['double_block', i]({'img': h, 't_emb': t_emb, 'mod_segments': mod_segments, 'rope_freqs': rope_freqs, 'transformer_options': transformer_options}, {'original_block': block_wrap})['img']
        else:
            h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
    if prefetch_queue is not None:
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)
    va, vb, _ = next((s for s in layout.segments if s[2] == 'video'))
    aa, ab, _ = next((s for s in layout.segments if s[2] == 'audio'))
    if video_rows_t is not None:
        video_seg = (va, vb, rows_to_mod_index(video_rows_t, 0) // 3)
    else:
        video_seg = (va, vb, t_row[seg_t['video']])
    if audio_rows_t is not None:
        audio_seg = (aa, ab, rows_to_mod_index(audio_rows_t, 0) // 3)
    else:
        audio_seg = (aa, ab, t_row[seg_t['audio']])
    v, a = self.final_layer(h, t_emb, video_seg, audio_seg, sigma_v, transformer_options.get('sample_sigmas'), (shift_v, shift_a))
    video_out = unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, self.latents_dim, self.patch_size)
    video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
    audio_out = unpack_audio(a)
    return [-video_out.to(video_x.dtype), -audio_out.to(audio_x.dtype)]
