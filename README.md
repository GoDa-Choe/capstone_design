# Occluded Point Clouds Classification via Point Clouds Completion 
#### Software Convergence Capstone design 2021-2

- **Myenggyu Choe (2017103762, software convergence Kyunghee Univs.)**
- Advisor Prof. Hyoseok Hwang(software convergence Kyunghee Univs.)


--------

## Overview
![image](https://user-images.githubusercontent.com/68394004/147044643-47fedd8b-bfbe-45e0-8f8b-17e65586d1d4.png)
실제 환경에서 라이다 등과 같은 3D 센서로 관측한 Point Clouds는 Occlusion 현상으로 인하여 불완전하다. 즉 실제 관측 가능 Point Clouds는 불완전한 포인트 클라우드로 분류할 수 있다.
이에 따른 학습 데이터셋과 테스트 데이터셋의 차이는 실제 성능 하락을 야기한다. 이번 캡스톤디자인에서는 생성에서 분류를 이어이지는 2단계 구조를 도입하여 Occluded Point Clouds의 Classification 정확도를 향상한다.

--------

## Architecture
![occluded_point_cloud_classification_network_architecture](https://user-images.githubusercontent.com/68394004/147044985-36ef4fdf-8e9b-4f8f-8daf-43fee78cbdcb.jpg)

생성에서 분류로 이어이지 2단계 구조를 도입하고 있다. Generator는 AutoEncoder 베이스의 PCN(Point Completion Network)를 활용하였으며 Classifier는 PointNet을 활용하였다.
PointNet은 complete point clouds로 사전 학습되었으며 Generator는 손실함수를 Cross Entropy와 Chamfer Distance로 나눠 학습하였다.

**Generator** : PCN  
**Classifier** : pre-trained PointNet

--------

## Experiments

**실험1**
![image](https://user-images.githubusercontent.com/68394004/147045946-dac0bda9-c148-4365-b0b8-afa4bd04a077.png)

**실험2**
![image](https://user-images.githubusercontent.com/68394004/147046002-b37fde71-5870-49bf-956b-da43f84949d0.png)

### Dataset
MVP(Multi-View Partial Dataset) & Partitioned MVP
![image](https://user-images.githubusercontent.com/68394004/147045896-ef220d69-1d05-4f8a-9d80-fcae5ad26ace.png)
![image](https://user-images.githubusercontent.com/68394004/147045905-5aa30bc3-5462-4b66-a882-13885965e893.png)

--------

## Results

![image](https://user-images.githubusercontent.com/68394004/147045857-862041ba-9e85-4a7c-8734-1734c3bbd66a.png)

![image](https://user-images.githubusercontent.com/68394004/147045359-b383ad06-6447-4a0c-94fb-933e48b26511.png)

--------

## Demo Video
https://user-images.githubusercontent.com/68394004/147043612-5f530eaf-7746-4775-9545-ba5f89041863.mp4
Link: [Demo Video](./소프트웨어융합캡스톤디자인(시연동영상_Point2Vision_최명규).mp4)

--------

## Commands

**train**
```
python3 train_file.py(src/train/*)

ex: python3 partitioned_mvp_occluded.py
```


**test**
```
python3 test_file.py(src/test/*)

ex: python3 partitioned_mvp_occluded_occluded.py
```
--------

### Report
[소프트웨어융합캡스톤디자인03(결과보고서_Point2Vision_최명규).pdf](https://github.com/GoDa-Choe/capstone_design/files/7760363/03._Point2Vision_.pdf)

### Presentation
[소프트웨어융합캡스톤디자인(최종발표_Point2Vision_최명규).pdf](https://github.com/GoDa-Choe/capstone_design/files/7760362/_Point2Vision_.pdf)

