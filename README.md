# Analisis Citra Medis — All Task Remote

Kumpulan tugas pengolahan citra medis: preprocessing, segmentasi klasik, dan
segmentasi nukleus dengan U-Net.

## Struktur

```
all-task-remote/
├── preprocessing-image/   # enhancement citra X-ray (jalan)
│   ├── data/              # input: 7 citra X-ray lutut
│   ├── src/               # 5 fungsi preprocessing
│   ├── output/            # hasil per citra (di-ignore git)
│   └── main.py            # batch semua citra di data/
├── segmentation/          # segmentasi USG infected (jalan)
│   ├── src/               # 5 fungsi segmentasi
│   ├── output/            # hasil per citra (di-ignore git)
│   └── main.py            # batch citra dari dataset/infected/
├── unet-nuclei-segmentation/  # U-Net segmentasi nukleus BBBC038
│   ├── configs/           # config eksperimen
│   ├── src/               # data, model, training, evaluation, utils
│   ├── scripts/           # prepare, train, evaluate, predict
│   ├── tests/             # unit test
│   └── README.md
├── requirements.txt       # dependensi global tugas klasik
└── README.md
```

## Setup

Semua tugas memakai **satu virtual environment** di root repo:

```bash
python3 -m venv .venv
source .venv/bin/activate

# Laptop CPU-only (lewati langkah ini di Colab/GPU):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Dependensi semua tugas:
pip install -r requirements.txt
pip install -r unet-nuclei-segmentation/requirements.txt
```

## Cara jalan

```bash
python preprocessing-image/main.py
python segmentation/main.py
```

Tugas U-Net memakai venv yang sama:

```bash
cd unet-nuclei-segmentation
python scripts/prepare_data.py
python scripts/prepare_test.py
python scripts/train.py
```

Detail alur, protokol evaluasi, dan struktur data ada di
`unet-nuclei-segmentation/README.md`.

Hasil tersimpan di `preprocessing-image/output/` dan `segmentation/output/`
dengan format `{nama_input}_{teknik}.png`.

## Dependensi

- `opencv-python-headless` — baca citra, CLAHE, median, Gaussian, threshold, k-means, watershed
- `numpy` — operasi array / histogram
- `matplotlib` — simpan plot perbandingan (backend `Agg`, tanpa display)
- `scikit-image` — active contour (snakes)
