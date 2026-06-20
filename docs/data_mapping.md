# Data Mapping

DermBench is distributed here as a text-only release.

The source text directory used to build this release was:

```text
DermCoT/Dermnet2/step4_txt/test
```

Each CoT text is aligned to a DermNet test image by path. The manifest field `dermnet_image_path` is the relative path under a local DermNet test image root, while `cot` is the relative path to the released CoT text in this repository.

Example:

```json
{
  "dermnet_image_path": "Acne and Rosacea Photos/07PerioralDermEye.jpg",
  "cot": "data/cot/Acne and Rosacea Photos/07PerioralDermEye.jpg.txt"
}
```

If your local image root is `/data/DermNet/test`, the expected image path is:

```text
/data/DermNet/test/Acne and Rosacea Photos/07PerioralDermEye.jpg
```

The local source contained 4002 paired test identifiers. The released manifest contains 4000 items to match the paper. Two problematic identifiers were excluded:

- `Acne and Rosacea Photos/Forest-2.jpg`: non-dermatology landscape image.
- `Atopic Dermatitis Photos/11IMG001.jpg`: missing disease label in the source label text.

