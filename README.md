# Analisis Citra Medis — All Task Remote

Kumpulan tugas pengolahan citra medis: preprocessing dan segmentation.

## Struktur

```
all-task-remote/
├── preprocessing-image/   # enhancement citra X-ray (jalan)
│   ├── data/              # input: 7 citra X-ray lutut
│   ├── src/               # 5 fungsi preprocessing
│   ├── output/            # hasil per citra (di-ignore git)
│   └── main.py            # batch semua citra di data/
├── segmentation/          # segmentasi tulang lutut (jalan)
│   ├── src/               # 6 fungsi segmentasi
│   ├── output/            # hasil per citra (di-ignore git)
│   └── main.py            # batch semua citra di preprocessing-image/data/
├── requirements.txt       # dependensi global
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

Hasil tersimpan di `preprocessing-image/output/` dan `segmentation/output/`
dengan format `{nama_input}_{teknik}.png`.

## Dependensi

- `opencv-python` — baca citra, CLAHE, median, Gaussian, threshold, k-means, watershed
- `numpy` — operasi array / histogram
- `matplotlib` — simpan plot perbandingan (backend `Agg`, tanpa display)
- `scikit-image` — active contour (snakes)
