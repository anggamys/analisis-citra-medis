# U-Net Nuclei Segmentation (BBBC038)

Segmentasi **binary semantic** nukleus pada dataset BBBC038 (Kaggle 2018 Data
Science Bowl). Seluruh nukleus digabung menjadi satu kelas foreground; tugas
ini tidak melakukan instance segmentation.

## Dataset

| Folder             | Isi                              | Peran                                                                          |
| ------------------ | -------------------------------- | ------------------------------------------------------------------------------ |
| `data-train/`      | 670 citra + 29.461 mask instance | train/validation                                                               |
| `data-test/`       | 65 citra stage1_test             | test resmi (65 `Public`)                                                       |
| `data-test-large/` | 3.019 citra stage2_test          | 106 `Private` untuk test eksternal; 2.913 `Ignored` tidak dipakai untuk metrik |

Ground truth test resmi tidak disimpan sebagai folder mask, melainkan sebagai
RLE di:

- `data/solutions/stage1_solution.csv`
- `data/solutions/stage2_solution_final.csv`

`scripts/prepare_test.py` men-decode RLE tersebut. Baris `Usage=Ignored` pada
stage2 **wajib dilewati** karena mask-nya berupa dummy `10x10`.

## Struktur

```text
unet-nuclei-segmentation/
├── configs/config.yaml
├── data/
│   ├── solutions/            # CSV ground truth resmi (tracked)
│   ├── splits/               # manifest CSV hasil prepare (tracked)
│   └── processed/            # mask gabungan + cache (ignored)
├── src/
│   ├── data/                 # loader, transform, RLE, manifest, split
│   ├── models/               # U-Net dan blok penyusun
│   ├── training/             # loss, optimizer, trainer
│   ├── evaluation/           # metrik dan evaluator
│   └── utils/                # config, seed, checkpoint, logger, visualisasi
├── scripts/
│   ├── prepare_data.py
│   ├── prepare_test.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
├── experiments/
└── requirements.txt
```

Tidak ada notebook (`.ipynb`). Eksperimen dijalankan lewat script agar
konsisten dan mudah diulang.

## Setup

Proyek ini memakai **satu virtual environment di root repo** (`.venv`),
dipakai bersama `preprocessing-image/` dan `segmentation/`. Setup diuji dengan
Python 3.14.10 dan PyTorch CPU.

Dari root repo:

```bash
python3 -m venv .venv
source .venv/bin/activate

# CPU-only (tanpa CUDA), jauh lebih ringan:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Dependensi klasik + U-Net (torch yang sudah terpasang dipakai ulang):
pip install -r requirements.txt
pip install -r unet-nuclei-segmentation/requirements.txt
```

Lalu jalankan script dari folder proyek:

```bash
cd unet-nuclei-segmentation
python scripts/train.py --smoke-test
```

Catatan: instalasi dependensi tidak dilakukan otomatis oleh repo ini. Untuk
GPU NVIDIA, lewati langkah CPU-only dan install PyTorch sesuai versi CUDA dari
situs resmi PyTorch agar build yang terpasang sesuai perangkat.

Di Colab/Kaggle, PyTorch bawaan sudah memakai CUDA. **Jangan** menjalankan
perintah `--index-url .../cpu` di sana karena akan mengganti torch CUDA dengan
build CPU. Cukup jalankan `pip install -r unet-nuclei-segmentation/requirements.txt`
dari root repo (atau `pip install -r requirements.txt` dari folder proyek);
torch yang sudah ada akan dipakai ulang.

## Alur penggunaan

### 1. Siapkan train/validation

```bash
python scripts/prepare_data.py --config configs/config.yaml
```

Script akan:

- menggabungkan seluruh mask instance menjadi satu mask biner 0/255,
- memvalidasi ukuran citra/mask,
- membagi 670 citra menjadi train/validation (default 85/15),
- menulis `data/splits/train.csv`, `data/splits/val.csv`, dan
  `data/processed/dataset_report.json`.

### 2. Siapkan test resmi

```bash
python scripts/prepare_test.py --config configs/config.yaml
```

Jika CSV solusi belum ada, tambahkan `--download`. Script akan men-decode RLE
dan menulis:

- `data/splits/test_stage1.csv` (65 citra),
- `data/splits/test_stage2_scored.csv` (106 citra),
- `data/splits/predict_stage1_all.csv` (65 citra),
- `data/splits/predict_stage2_all.csv` (3.019 citra).

### 3. Training

Default `configs/config.yaml` dioptimalkan untuk GPU (Colab T4/L4): U-Net
`base_channels: 64`, `batch_size: 16`, AMP aktif, 100 epoch dengan early
stopping. Device dipilih otomatis (`cuda` jika tersedia).

```bash
# Di Colab/GPU: langsung jalankan
python scripts/train.py --config configs/config.yaml

# Di laptop CPU: cukup smoke test, atau turunkan batch size
python scripts/train.py --smoke-test
python scripts/train.py --device cpu --batch-size 2 --base-channels 32 --epochs 5
```

Smoke test menjalankan 1 epoch pada 8 sampel dan otomatis memakai batch size 2.

Manifest train/validation bisa diarahkan langsung dari terminal:

```bash
python scripts/train.py \
  --train-manifest data/splits/train.csv \
  --val-manifest data/splits/val.csv

# atau arahkan seluruh direktori split
python scripts/train.py --splits-dir data/splits
```

Path relatif dihitung dari root proyek (`unet-nuclei-segmentation/`).

Resume:

```bash
python scripts/train.py --resume experiments/baseline/checkpoints/last.pt
```

### 4. Evaluasi

```bash
# test stage1 resmi
python scripts/evaluate.py --split stage1

# test eksternal stage2 (hanya 106 image Private)
python scripts/evaluate.py --split stage2

# pilih threshold di validation, lalu evaluasi test
python scripts/evaluate.py --split stage1 --tune-on-val
```

Threshold tidak boleh dipilih dari test set.

### 5. Prediksi tanpa ground truth

```bash
python scripts/predict.py \
  --manifest data/splits/predict_stage2_all.csv \
  --save-overlays
```

`predict_stage2_all.csv` memuat 3.019 citra (termasuk `Ignored`) karena
prediksi tidak membutuhkan label. Untuk metrik resmi, gunakan
`test_stage2_scored.csv` (106 citra).

## Konfigurasi

Semua parameter utama ada di `configs/config.yaml`. Default dioptimalkan untuk
GPU Colab (T4 16 GB / L4 24 GB): `base_channels: 64`, `batch_size: 16`,
`image_size: 256`, AMP aktif, dan `num_workers: 2`. Jika VRAM terbatas atau
memakai CPU, turunkan `base_channels` ke 32 dan/atau `batch_size` lewat CLI
(`--base-channels 32 --batch-size 2`).

Preprocessing citra:

- alpha channel dibuang,
- citra 16-bit dinormalisasi ke uint8 secara aman,
- aspek rasio dijaga: resize sisi terpanjang ke `image_size`, lalu pad ke
  persegi,
- mask selalu memakai interpolasi nearest,
- normalisasi memakai mean/std dari config,
- augmentasi hanya diterapkan pada training set.

## Protokol evaluasi

- Metrik: Dice, IoU, precision, recall.
- Dihitung **per image**, lalu dilaporkan mean ± std.
- Metrik dihitung pada kanvas model (`image_size`, dengan padding). Saat
  menyimpan mask prediksi, area padding dipotong dan mask dikembalikan ke
  ukuran asli citra.
- Stage1: 65 image `Public`.
- Stage2: hanya 106 image `Private`; 2.913 `Ignored` tidak dihitung.
- Threshold dipilih di validation.
- Test set tidak dipakai untuk memilih epoch, checkpoint, atau hyperparameter.

## Output eksperimen

```text
experiments/baseline/
├── config.yaml
├── environment.json
├── history.json
├── training_curves.png
├── checkpoints/{best.pt,last.pt}
├── logs/train_log.csv
├── evaluation/
│   ├── metrics_test_stage1.json
│   ├── per_image_test_stage1.csv
│   ├── predictions_test_stage1.png
│   └── failure_cases_test_stage1.png
└── predictions/
```

Checkpoint terbaik dipilih berdasarkan validation Dice.

## Testing

```bash
pytest -q
```

Tes mencakup RLE round-trip, split tanpa leakage, bentuk output U-Net, metrik
edge case, dan bentuk tensor Dataset.

## Catatan hardware

Laptop pengembangan memakai PyTorch CPU karena GPU lokal (MX110, 2 GB VRAM)
tidak memadai untuk training penuh. Konfigurasi default ditujukan untuk
Colab/Kaggle GPU (T4 16 GB atau lebih), dengan AMP dan batch size 16. Di CPU,
gunakan smoke test atau turunkan batch size; estimasi waktu training penuh di
CPU bisa sangat lama.
