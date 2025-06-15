## 1.Setup

```bash
git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
cd alpha-beta-CROWN
conda env create -f complete_verifier/environment.yaml --name alpha-beta-crown
conda activate alpha-beta-crown


```
---

## 2. Custom Model Download

```bash
git clone https://github.com/Do-Gun/alpha-beta-CROWN-gtsrb

mv alpha-beta-CROWN-gtsrb/gtsrb_custom_data.py complete_verifier/custom/
mv alpha-beta-CROWN-gtsrb/gtsrb_config.yaml complete_verifier/exp_configs/
mv alpha-beta-CROWN-gtsrb/model.pt complete_verifier/models/
```
---

## 3. Validation
```bash
python abcrown.py --config exp_configs/gtsrb_config.yaml
```
