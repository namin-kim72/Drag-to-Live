import torch
from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
from diffusers.utils import export_to_gif
from PIL import Image
import cv2
import numpy as np
import os

# ==========================================
# 1. 설정 (학습 때와 맞춰야 잘 나옵니다)
# ==========================================
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
MOTION_ADAPTER = "guoyww/animatediff-motion-adapter-v1-5-2"
LORA_PATH = "output_drag_lora"  # 방금 학습 끝난 폴더
TEST_IMAGE_PATH = "test_input.png"  # 준비한 이미지

# 학습 때 썼던 프롬프트 그대로 사용 (중요!)
PROMPT = "timelapse clouds moving in the sky, cinematic, high quality, 4k"
NEGATIVE_PROMPT = "bad quality, worst quality, blurry, low resolution, distortion, watermark"


# ==========================================
# 2. 색상 보정 저장 함수 (파란색 방지)
# ==========================================
def save_video_fixed(frames, path, fps=8):
    height, width, _ = np.array(frames[0]).shape
    # OpenCV 비디오 작성기
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame in frames:
        # PIL -> Numpy 변환
        img_np = np.array(frame)
        # RGB -> BGR 변환 (이게 핵심!)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        out.write(img_bgr)

    out.release()
    print(f"✨ 영상 저장 완료: {path}")


# ==========================================
# 3. 메인 실행 함수
# ==========================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 모델 로드
    print("Loading Base Model...")
    adapter = MotionAdapter.from_pretrained(MOTION_ADAPTER)
    pipe = AnimateDiffPipeline.from_pretrained(
        BASE_MODEL,
        motion_adapter=adapter,
        torch_dtype=torch.float16
    ).to(device)

    # 스케줄러 설정
    pipe.scheduler = DDIMScheduler.from_pretrained(
        BASE_MODEL,
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1
    )

    # 2. 학습한 LoRA 불러오기 (성적표 확인)
    print(f"Loading LoRA from {LORA_PATH}...")
    try:
        pipe.unet.load_attn_procs(LORA_PATH)
        print("✅ LoRA Load Success!")
    except Exception as e:
        print(f"❌ LoRA Load Failed: {e}")
        return

    # 3. 이미지 준비
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"⚠️ {TEST_IMAGE_PATH}가 없습니다! 검은 화면으로 테스트합니다.")
        input_image = Image.new('RGB', (256, 256), color='black')
    else:
        input_image = Image.open(TEST_IMAGE_PATH).convert("RGB")
        input_image = input_image.resize((256, 256))  # 학습 해상도 맞춤

    # 4. 영상 생성 (Inference)
    print("Generating Video... (약 1분 소요)")

    # 시드 고정 (매번 똑같이 잘 나오게 하기 위해)
    generator = torch.Generator(device=device).manual_seed(42)

    output = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_frames=16,  # 2초 영상
        guidance_scale=7.5,
        num_inference_steps=25,  # 25번만 그려도 충분
        generator=generator,
        width=256,
        height=256
    )

    frames = output.frames[0]

    # 5. 저장 (GIF + MP4)
    export_to_gif(frames, "final_result.gif")
    save_video_fixed(frames, "final_result.mp4", fps=8)

    print("🎉 모든 작업 완료! 'final_result.mp4'를 확인하세요.")


if __name__ == "__main__":
    main()