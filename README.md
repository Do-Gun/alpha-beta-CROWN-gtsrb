본 프로젝트는 자율주행 차량에 사용될 수 있는 **GTSRB (German Traffic Sign Recognition Benchmark)** 데이터로 학습된 **CNN 모델의 안전성**을 [alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN) 프레임워크를 통해 검증합니다.


## 1.Setup

```bash
git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
cd alpha-beta-CROWN

conda env create -f complete_verifier/environment.yaml --name alpha-beta-crown
conda activate alpha-beta-crown
```
---

## 2. Custom Model Download
이 프로젝트에서 사용된 GTSRB CNN 모델에 대한 정보는 다음 레포지토리에서 확인할 수 있습니다:

https://github.com/Do-Gun/GTSRB-Marabou-Verification

```bash
git clone https://github.com/Do-Gun/alpha-beta-CROWN-gtsrb

mv alpha-beta-CROWN-gtsrb/gtsrb_custom_data.py complete_verifier/custom/
mv alpha-beta-CROWN-gtsrb/gtsrb_config.yaml complete_verifier/exp_configs/
mv alpha-beta-CROWN-gtsrb/model.pt complete_verifier/models/
```
---

## 3. Validation
```bash
cd complete_verifier

python abcrown.py --config exp_configs/gtsrb_config.yaml
```
