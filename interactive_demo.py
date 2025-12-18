import cv2
import numpy as np
import torch
from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
from PIL import Image
import os

# ==========================================
# 1. 설정
# ==========================================
IMAGE_PATH = "test_input.png"  # 피라미드 사진
OUTPUT_VIDEO = "demo_result.mp4"
OUTPUT_INPUT_IMG = "demo_input_with_arrow.jpg"  # PPT에 넣을 '입력' 이미지

# 모델 설정
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
MOTION_ADAPTER = "guoyww/animatediff-motion-adapter-v1-5-2"
LORA_PATH = "output_drag_lora"

# ==========================================
# 2. 드래그 인터페이스 (마우스로 그리기)
# ==========================================
drawing = False
ix, iy = -1, -1
img_display = None


def draw_arrow(event, x, y, flags, param):
    global ix, iy, drawing, img_display

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_img = img_display.copy()
            # 빨간색 화살표 그리기 (궤적 시각화)
            cv2.arrowedLine(temp_img, (ix, iy), (x, y), (0, 0, 255), 2, tipLength=0.3)
            cv2.imshow('Drag Your Cloud', temp_img)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # 최종 화살표 고정
        cv2.arrowedLine(img_display, (ix, iy), (x, y), (0, 0, 255), 2, tipLength=0.3)
        cv2.imshow('Drag Your Cloud', img_display)
        print(f"👉 궤적 입력됨: ({ix},{iy}) -> ({x},{y})")


# ==========================================
# 3. AI 영상 생성기
# ==========================================
def generate_video(pipe, prompt):
    print("⏳ AI가 구름을 생성하는 중... (약 1분)")
    generator = torch.Generator("cuda").manual_seed(42)
    output = pipe(
        prompt=prompt,
        negative_prompt="bad quality, low resolution",
        num_frames=16,
        guidance_scale=7.5,
        num_inference_steps=25,
        generator=generator,
        width=256, height=256
    )
    frames = output.frames[0]

    # 영상 저장 (색상 보정 포함)
    height, width, _ = np.array(frames[0]).shape
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), 8, (width, height))
    for frame in frames:
        img_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        out.write(img_bgr)
    out.release()
    print(f"✅ 영상 생성 완료: {OUTPUT_VIDEO}")


# ==========================================
# 4. 메인 실행
# ==========================================
def main():
    global img_display

    # 1. 이미지 로드 및 리사이즈
    if not os.path.exists(IMAGE_PATH):
        print("❌ test_input.jpg가 없습니다! 준비해주세요.")
        return

    original_img = cv2.imread(IMAGE_PATH)
    img_display = cv2.resize(original_img, (256, 256))

    # 2. 마우스 입력 받기
    cv2.namedWindow('Drag Your Cloud')
    cv2.setMouseCallback('Drag Your Cloud', draw_arrow)

    print("🎨 이미지 위에 마우스로 드래그하여 화살표를 그리세요.")
    print("   (다 그렸으면 'Enter'를 누르세요. 'q'는 종료)")

    while True:
        cv2.imshow('Drag Your Cloud', img_display)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter 키
            break
        elif key == ord('q'):
            return

    # 3. 입력 이미지 저장 (PPT용)
    cv2.imwrite(OUTPUT_INPUT_IMG, img_display)
    cv2.destroyAllWindows()
    print(f"📸 입력 궤적 이미지 저장됨: {OUTPUT_INPUT_IMG}")

    # 4. 모델 로드 및 생성
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = MotionAdapter.from_pretrained(MOTION_ADAPTER)
    pipe = AnimateDiffPipeline.from_pretrained(BASE_MODEL, motion_adapter=adapter, torch_dtype=torch.float16).to(device)
    pipe.scheduler = DDIMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler", clip_sample=False,
                                                   timestep_spacing="linspace", steps_offset=1)

    try:
        pipe.unet.load_attn_procs(LORA_PATH)
        print("✅ LoRA 적용 완료")
    except:
        print("⚠️ LoRA 파일이 없어서 기본 모델로 동작합니다.")

    # 생성 시작
    generate_video(pipe, "timelapse clouds moving over egyptian pyramids, desert, cinematic, high quality, 4k")


if __name__ == "__main__":
    main()