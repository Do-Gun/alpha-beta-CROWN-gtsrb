## 1.Setup

```bash
git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
cd alpha-beta-CROWN
conda env create -f complete_verifier/environment.yaml --name alpha-beta-crown
conda activate alpha-beta-crown

git clone https://github.com/Do-Gun/alpha-beta-CROWN-gtsrb

```
---

## Move Uploaded Files

```bash
mv ../alpha-beta/gtsrb_custom_data.py complete_verifier/custom/
mv ../alpha-beta/gtsrb_config.yaml complete_verifier/exp_configs/
mv ../alpha-beta/model.pt complete_verifier/models/
```
