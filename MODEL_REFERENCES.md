# Model files kept outside Git

Model parameters, checkpoints, adapters, and their local symlinks are excluded from Git and Git LFS. Download them from the pinned sources below, place them at the listed destination under assets/, and verify the SHA-256 checksum. The three local LoRA files and the small time embedding grid were hashed and match these values. The larger symlink targets have matching published sizes; their full hashes were not recalculated here.

## Pika H3 Flex adapters

Source: [Pika-Labs/Pika-H3-Flex](https://huggingface.co/Pika-Labs/Pika-H3-Flex/tree/3ff166447d5edb1406107ab8370c42000ce173e4/comfy/loras), revision 3ff166447d5edb1406107ab8370c42000ce173e4. License: MiniMax H3 Community License Agreement, with Pika NOTICE.

| Destination under assets/ | Source path | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| loras/pika_h3_flex_fl2va_comfy_bf16.safetensors | comfy/loras/pika_h3_flex_fl2va_comfy_bf16.safetensors | 1956172176 | 68783e6447696ec365d13c0554941e070bee4cfea13c37da52498ec4f9a53f89 |
| loras/pika_h3_flex_ref2va_comfy_bf16.safetensors | comfy/loras/pika_h3_flex_ref2va_comfy_bf16.safetensors | 1956172176 | e4d058fbd027377b925a0ee607b514ddd9037873fee3afa05d73b302e6f7f8e1 |

## MiniMax H3 base components

Source: [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/54472221d27c695fca9621b0c3ed52f06c4af707), revision 54472221d27c695fca9621b0c3ed52f06c4af707. License: [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE). Source paths equal destinations.

| Destination under assets/ | Bytes | SHA-256 |
| --- | ---: | --- |
| text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors | 15687142551 | 35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6 |
| vae/minimax_h3_video_vae_fp16.safetensors | 5207808496 | 7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522 |
| vae/minimax_h3_audio_vae_fp32.safetensors | 605254808 | 8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48 |
| diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors | 20970379616 | e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a |
| diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors | 20970379616 | 9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779 |

## Other adapters

| Destination under assets/ | Pinned source | Bytes | SHA-256 | License |
| --- | --- | ---: | --- | --- |
| loras/h3_character_swap_pro4500_1000.safetensors | [akatz-ai/MiniMax-H3-Character-Swap-LoRA](https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA/blob/96ea3407b230b6f7df71c58c70e31df96bd7e533/h3_character_swap_pro4500_1000.safetensors) | 155110320 | 4b2a3f420ae804c0aa3422761ff84dbd1bf52eef6900ffab6d2e66df63cb4e79 | MiniMax H3 Community License Agreement |
| loras/minimax_h3_turbo_v4_step600_ema.safetensors | [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/blob/43a74557ac3f6539db8e0f2a959d03feb7a81480/minimax_h3_turbo_v4_step600_ema.safetensors) | 779849816 | 5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3 | Apache 2.0 |
| loras/h3_silu_temb_grid.safetensors | [Larryvrh/ComfyUI-MiniMax-H3-Turbo](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/blob/4274783a23afcfdbea3b4876cb79effd6c510785/h3_silu_temb_grid.safetensors) | 5510600 | 30eb3c2cc7fb6b470d9717ff840d359313ac27cd64b705e32da1baa10f72d6a8 | Apache 2.0 |

## Retrieval

Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli). For each Hugging Face file, run:

    hf download REPOSITORY SOURCE_PATH --revision REVISION --local-dir DOWNLOAD_DIRECTORY

Copy the downloaded file from DOWNLOAD_DIRECTORY/SOURCE_PATH to its destination under assets/. Download the time embedding grid from the pinned GitHub file link. Run sha256sum on each destination and compare with this manifest. Hugging Face authentication may be required by the client or source access rules. Review each source's license and notices before use.
