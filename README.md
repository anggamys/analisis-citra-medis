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

```bash
pip install -r requirements.txt
```

## Cara jalan

```bash
python preprocessing-image/main.py
python segmentation/main.py
```

Tugas U-Net memakai virtual environment dan dependensi terpisah:

```bash
cd unet-nuclei-segmentation
pip install -r requirements.txt
python scripts/prepare_data.py
python scripts/prepare_test.py
python scripts/train.py
```

Detail alur, protokol evaluasi, dan struktur data ada di
`unet-nuclei-segmentation/README.md`.

Hasil tersimpan di `preprocessing-image/output/` dan `segmentation/output/`
dengan format `{nama_input}_{teknik}.png`.

## Dependensi

- `opencv-python` — baca citra, CLAHE, median, Gaussian, threshold, k-means, watershed
- `numpy` — operasi array / histogram
- `matplotlib` — simpan plot perbandingan (backend `Agg`, tanpa display)
- `scikit-image` — active contour (snakes)
