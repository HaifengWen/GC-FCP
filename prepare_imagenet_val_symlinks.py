from pathlib import Path
import os

val_dir = Path("data/imagenet/val")
label_txt = Path("data/imagenet/ImageNet_val_label.txt")
out_dir = Path("data/imagenet/val_by_synset")

out_dir.mkdir(parents=True, exist_ok=True)

missing = []
n = 0

with open(label_txt, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        filename, synset = line.split()
        dst_dir = out_dir / synset
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Source may be flat: val/ILSVRC2012_val_x.JPEG
        src = val_dir / filename

        # Or already organized: val/nXXXX/ILSVRC2012_val_x.JPEG
        if not src.exists():
            src = val_dir / synset / filename

        if not src.exists():
            missing.append((filename, synset))
            continue

        dst = dst_dir / filename
        if not dst.exists():
            os.symlink(src.resolve(), dst)

        n += 1

print("linked:", n)
print("missing:", len(missing))
if missing[:10]:
    print("first missing:", missing[:10])