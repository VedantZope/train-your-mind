# Work Log

## 2026-02-18
- Added new model options: `res_gn_gelu` and `cbam` (CBAM supports `--dilation`).  
- Added inference displacement export (`--save-displacement`) and Learn2Reg‑friendly naming `disp_{fixedId}_{movingId}.npz`.  
- Inference now skips TRE when keypoints are missing but still saves displacement fields.  
- Added dilation support to `res_gn_gelu` (uses `--dilation`).  
- Added multi‑block training support with `--num-blocks`, `--freeze-blocks`, and `--init-block1-ckpt` for cascade training.  
- Added feature-mode support in training/eval: `learned`, `mind`, and `mind_concat` (concatenates MIND + learned features before FireANTs).  
