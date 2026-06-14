# RaccoonBot OpenVLA Pipeline Cube Push Task

## 1. Project Overview

본 프로젝트는 기존 RaccoonBot OpenVLA pipeline을 확장하여, 기존 colored cylinder grasp task뿐만 아니라 cube object와 cube push task를 추가한 과제이다.

기존 예제는 `"grasp the red cylinder"`와 같은 cylinder grasp instruction만 지원하였다. 본 과제에서는 MuJoCo scene에 cube object를 추가하고, grasp task와 push task를 모두 포함하는 multi-object, multi-task dataset을 구성하였다.

## 2. Main Contribution

본 프로젝트의 주요 변경점은 다음과 같다.

- MuJoCo scene에 cube object 추가
- 기존 cylinder grasp task를 cube grasp task까지 확장
- cube push task 추가
- 4-object scene 구성
- RLDS / TFDS dataset 재구축
- OpenVLA LoRA fine-tuning 수행
- push inference 안정화를 위한 client-side action 보정 추가

## 3. Task Setup

최종 active object는 다음 4개로 구성하였다.

- green cylinder
- yellow cylinder
- red cube
- blue cube

최종 dataset은 다음 6개 task-target 조합으로 구성하였다.

| Task | Target | Episodes |
|---|---|---:|
| grasp | green cylinder | 50 |
| grasp | yellow cylinder | 50 |
| grasp | red cube | 50 |
| grasp | blue cube | 50 |
| push | red cube | 50 |
| push | blue cube | 50 |
| Total |  | 300 |

Cylinder push도 실험하였으나, cylinder는 굴림 및 접촉 특성이 불안정하여 안정적인 push demonstration을 확보하기 어려웠다. 따라서 최종 dataset에서는 push task를 cube object로 제한하였다.

## 4. New / Edited Files

| File | Description |
|---|---|
| `Mujoco/Raccoon_colored_cylinder_cube.xml` | cylinder와 cube가 포함되도록 수정한 MuJoCo scene |
| `Mujoco/raccoon_grasp_push_4objects_alltargets_dataset_fixed.py` | grasp/push demonstration dataset 생성 코드 |
| `local_client/openvla_grasp_push_4objects_custom_client_focus_pushboost.py` | grasp/push inference를 위한 수정 client |
| `logs/finetune_grasp_push_300ep_10000step_b2_ga8.log` | OpenVLA LoRA fine-tuning log |

## 5. Dataset Generation

Raw MuJoCo demonstration은 다음 명령어로 생성하였다.

```bash
cd /data/Raccoonbot_Openvla/Mujoco
python raccoon_grasp_push_4objects_alltargets_dataset_fixed.py
