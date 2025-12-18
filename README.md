# Drag-to-Live: Controllable Cloud Animation on Edge Device

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/Diffusers-0.31.0-yellow)](https://huggingface.co/docs/diffusers/index)
[![Gradio](https://img.shields.io/badge/Demo-OpenCV-green)]()

## 📖 Introduction
최신 영상 생성 AI(Sora, Runway 등)는 퀄리티가 높지만, 사용자가 원하는 구체적인 움직임을 제어(Control)하기 어렵다는 한계가 있습니다. 또한, 기존 연구인 *Wan-Move* 등은 H100급의 고성능 GPU를 요구합니다.

**Drag-to-Live**는 이러한 문제를 해결하기 위해 고안된 **경량화 풍경 제어 모델**입니다.
사용자가 입력한 궤적(Trajectory)을 기반으로 정지된 이미지에 자연스러운 움직임을 부여하며, 최적화를 통해 **RTX 4060(8GB VRAM) 환경에서도 학습 및 구동**이 가능하도록 설계되었습니다.

## ✨ Key Features
* **👆 Intuitive Interface:** 이미지 위에 마우스로 드래그하여 움직임의 방향과 크기를 지정합니다.
* **⚡ Edge Optimization:** RTX 4060 8GB VRAM 환경에서도 학습 및 추론이 가능하도록 경량화되었습니다 (Gradient Checkpointing, fp16, 8-frame optimization).
* **🧠 Data-Driven Guidance:** CoTracker로 추출한 물리적 궤적을 LoRA(Low-Rank Adaptation)에 학습시켜 자연스러운 구름의 흐름을 구현합니다.
* **🎥 High Quality:** Stable Diffusion v1.5와 AnimateDiff를 기반으로 시네마틱한 타임랩스 영상을 생성합니다.

## 🏗️ System Architecture
<img width="4729" height="4465" alt="User Trajectory Cloud-2025-12-18-061955" src="https://github.com/user-attachments/assets/0fc1e0ea-faef-4c38-8c69-5bb860453889" />

본 프로젝트는 CoTracker(Trajectory Encoder)와 **AnimateDiff(Motion Generator)**, 그리고 LoRA(Style Controller)의 유기적인 결합으로 구성됩니다.
1.  **Input:** 정지 이미지 + 사용자 입력 궤적
2.  **Encoding:** CoTracker가 궤적을 좌표 데이터로 변환
3.  **Generation:** AnimateDiff가 시간 축을 생성하고, 학습된 LoRA가 구름의 물리적 움직임을 주입
4.  **Output:** 8~16 프레임의 고화질 타임랩스 영상

## 📦 Installation

### Prerequisites
* Windows 10/11 or Linux
* NVIDIA GPU (VRAM 8GB 이상 권장)
* Python 3.10+
* Anaconda (Recommended)

### Setup
```bash
# 1. Clone the repository
git clone [https://github.com/namin-kim72/Drag-to-Live.git](https://github.com/namin-kim72/Drag-to-Live.git)
cd Drag-to-Live

# 2. Create Conda environment
conda create -n drag2live python=3.10
conda activate drag2live

# 3. Install dependencies
# (CoTracker 설치를 위해 git이 필요합니다)
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Data Preparation (Optional)
직접 데이터셋을 구축하려면 `raw_cloud_video` 폴더에 MP4 파일들을 넣고 실행하세요. CoTracker가 궤적을 추출하여 학습 데이터를 생성합니다.
```bash
python build_dataset_drag2live.py
```

### 2. Training LoRA
구축된 데이터셋을 기반으로 LoRA 어댑터를 학습합니다. (RTX 4060 기준 약 30분~1시간 소요)
```Bash
python train_drag_lora.py
```

### 3. Interactive Demo (GUI)
학습된 모델을 사용하여 직접 화살표를 그리고 영상을 생성해봅니다.
```Bash
python interactive_demo.py
```
  사용법: 창이 뜨면 마우스로 원하는 구름의 이동 궤적을 그리세요. 입력을 마치면 Enter를 눌러 생성을 시작합니다.

### 📊 Results



https://github.com/user-attachments/assets/bebe15c9-77e8-4290-8e98-76d7513cf585



### 🛠️ Tech Stack

* **Core Model:** Stable Diffusion v1.5

* **Motion Module:** AnimateDiff

* **Trajectory Tracking:** CoTracker

* **Accelerator:** HuggingFace Accelerate & PEFT

* **Environment:** Python 3.10+, PyTorch 2.0+

### 👨‍💻 Author
* **Kim Nam-in** - Project Lead & Implementation - [IoT Track / Visual Intelligence Learning]

### 📜 References
1. Wan-Move: Motion-controllable Video Generation via Latent Trajectory Guidance (arXiv, 2025) 

2. AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning (ICLR, 2024) 

3. CoTracker: It is Better to Track Together (CVPR, 2024)
