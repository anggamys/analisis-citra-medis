# Segmentasi Citra USG Infected (Soal 3)

## Kasus

3 citra USG unik pertama dari `dataset/infected/` — sama dengan Soal 2
(`Image_001.jpg`, `Image_003.jpg`, `Image_005.jpg`; file dataset berpasangan
duplikat: 001=002, 003=004, dst.).

Masalah: pisahkan struktur anatomis (folikel hipoekoik gelap vs jaringan
echogenic terang) dari background agar region mencurigakan bisa diukur pada
analisis lanjut. Tantangan: speckle noise USG dan artefak garis tepi citra.

## Metode (sesuai spreadsheet tugas)

| Spreadsheet | Modul | Parameter diuji |
|-------------|-------|-----------------|
| Thresholding | `thresholding.py` | Otsu, fixed=127, adaptive-Gaussian |
| Growing | `growing.py` (region growing) | thresh 20, 40, 60 (seed piksel terang + blur 5×5) |
| Clustering (K-Means) | `clustering_kmeans.py` | k=2, 3, 4 |
| Clustering (FCM) | `clustering_fcm.py` (fuzzy c-means) | c=2, 3, 4 (m=2.0, seed fixed) |
| Watershed | `watershed.py` | dist_ratio 0.5, 0.7 |
| Active Contour (Snakes) | `snakes.py` | (α, β) = (0.015,10), (0.05,5), (0.01,20) |

## Output

Per citra input menghasilkan 7 file di `output/`:

```
{nama}_thresholding.png
{nama}_growing.png
{nama}_clustering_kmeans.png
{nama}_clustering_fcm.png
{nama}_watershed.png
{nama}_snakes.png
{nama}_summary.png
```

Contoh: `Image_003.jpg` → `Image_003_thresholding.png`, dst.
`*_summary.png` berisi Asli + 1 varian wakil tiap metode
(Otsu, growing t40, clustering k-means k2, clustering FCM c2,
watershed r0.7, snakes default).

## Cara jalan

```bash
# dari root repo
.venv/bin/python segmentation/main.py
```

Script memproses 3 sampel fixed di `dataset/infected/`,
menyimpan plot dengan backend `Agg` (tanpa display).

## Import sebagai modul

```python
from src import (
    thresholding,
    growing_segmentation,
    clustering_kmeans,
    clustering_fcm,
    watershed_segmentation,
    snakes_segmentation,
)
```
