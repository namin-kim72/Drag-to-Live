import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig
from accelerate import Accelerator
from tqdm.auto import tqdm
import numpy as np
import cv2
import torch.nn.functional as F

# 메모리 최적화 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==========================================
# 1. 긴급 속도 향상 설정
# ==========================================
DATASET_DIR = "dataset_drag2live"
OUTPUT_DIR = "output_drag_lora"
PRETRAINED_MODEL = "runwayml/stable-diffusion-v1-5"
MOTION_ADAPTER = "guoyww/animatediff-motion-adapter-v1-5-2"

RESOLUTION = 256
BATCH_SIZE = 1
# ★ 중요: 1로 줄여서 매 스텝 바로 업데이트 (속도 UP)
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-4

# ★ 중요: 600번만 해도 충분함 (약 30분~1시간 컷 예상)
TRAIN_STEPS = 600
CHECKPOINTING_STEPS = 200


# ==========================================
# 2. 데이터셋 클래스
# ==========================================
class DragDataset(Dataset):
    # ★ 중요: sample_n_frames를 8로 줄임 (메모리 2배 절약 -> 스와핑 방지)
    def __init__(self, dataset_dir, tokenizer, width=256, height=256, sample_n_frames=8):
        self.root = Path(dataset_dir)
        self.width = width
        self.height = height
        self.sample_n_frames = sample_n_frames
        self.tokenizer = tokenizer

        self.index_file = self.root / "index.jsonl"
        self.data = []
        if self.index_file.exists():
            with open(self.index_file, 'r', encoding='utf-8') as f:
                for line in f:
                    self.data.append(json.loads(line))

        print(f"Dataset Loaded: {len(self.data)} clips found.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        video_path = item['video']
        if not os.path.exists(video_path):
            video_path = str(self.root / "videos" / Path(item['video']).name)

        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (self.width, self.height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        # 프레임이 부족하면 복사, 넘치면 자름
        if len(frames) < self.sample_n_frames:
            frames = frames + [frames[-1]] * (self.sample_n_frames - len(frames))
        frames = frames[:self.sample_n_frames]

        pixel_values = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 127.5 - 1.0

        prompt = "timelapse clouds moving in the sky, cinematic, high quality"
        text_inputs = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs.input_ids[0]
        }


# ==========================================
# 3. 메인 학습 루프
# ==========================================
def main():
    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        mixed_precision="fp16"
    )

    motion_adapter = MotionAdapter.from_pretrained(MOTION_ADAPTER)
    pipe = AnimateDiffPipeline.from_pretrained(PRETRAINED_MODEL, motion_adapter=motion_adapter)

    unet = pipe.unet
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer

    # 메모리 최적화 켜기
    unet.enable_gradient_checkpointing()
    try:
        unet.enable_xformers_memory_efficient_attention()
    except:
        pass

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        bias="none",
    )
    unet.add_adapter(lora_config)

    trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)

    # 데이터셋 로드시 프레임 8로 설정
    dataset = DragDataset(DATASET_DIR, tokenizer, width=RESOLUTION, height=RESOLUTION, sample_n_frames=8)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    vae.to(accelerator.device, dtype=torch.float16)
    text_encoder.to(accelerator.device, dtype=torch.float16)

    print(f"***** Training Start (Total Steps: {TRAIN_STEPS}) *****")
    global_step = 0

    unet.train()
    progress_bar = tqdm(range(TRAIN_STEPS), disable=not accelerator.is_local_main_process)

    while global_step < TRAIN_STEPS:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=torch.float16)
                b, f, c, h, w = pixel_values.shape

                pixel_values = pixel_values.view(b * f, c, h, w)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents.view(b, f, 4, h // 8, w // 8)
                latents = latents * 0.18215

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, 1000, (b,), device=latents.device).long()

                from diffusers import DDPMScheduler
                noise_scheduler = DDPMScheduler.from_pretrained(PRETRAINED_MODEL, subfolder="scheduler")
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # (Batch, Channels, Frames, H, W)
                noisy_latents_input = noisy_latents.permute(0, 2, 1, 3, 4).contiguous()

                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                # 프레임 수(f)만큼 복제
                encoder_hidden_states = encoder_hidden_states.repeat_interleave(repeats=f, dim=0)

                model_pred = unet(noisy_latents_input, timesteps, encoder_hidden_states=encoder_hidden_states).sample

                # 원복
                model_pred = model_pred.permute(0, 2, 1, 3, 4).contiguous()

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % CHECKPOINTING_STEPS == 0:
                    save_path = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)

            if global_step >= TRAIN_STEPS:
                break

    unet = accelerator.unwrap_model(unet)
    unet.save_attn_procs(OUTPUT_DIR)
    print(f"🎉 DONE! Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()