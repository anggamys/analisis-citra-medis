# Segmentasi Citra X-ray Lutut

## Kasus

Citra X-ray sendi lutut grayscale yang sama dengan `preprocessing-image/data/`
(`0.jpg`, `2.jpg`, `3.jpg`, `4.jpeg`, `5.png`, `6.jpg`, `7.png`).

Masalah: pisahkan struktur tulang (femur–tibia) dari jaringan lunak/background
agar celah sendi dan batas tulang bisa diukur pada analisis lanjut.

## Metode vs masalah

| Teknik | Parameter diuji | Menjawab masalah |
|--------|-----------------|------------------|
| Thresholding | Otsu, fixed=127, adaptive-Gaussian | Pisah cepat tulang terang vs background |
| Region Growing | thresh 5, 10, 20 (seed tengah otomatis) | Tumbuhkan region konsisten intensitas dari seed |
| K-Means | k=2, 3, 4 | Kelompokkan piksel berdasar intensitas tanpa ambang manual |
| Watershed | dist_ratio 0.5, 0.7 | Pisah objek bersentuhan via marker + distance transform |
| Active Contour (Snakes) | (α, β) = (0.015,10), (0.05,5), (0.01,20) | Kontur deformable ikuti tepi tulang, init lingkaran otomatis |

## Output

Per citra input menghasilkan 6 file di `output/`:

```
{nama}_thresholding.png
{nama}_region_growing.png
{nama}_kmeans.png
{nama}_watershed.png
{nama}_active_contour.png
{nama}_summary.png
```

Contoh: `5.png` → `5_thresholding.png`, dst. `*_summary.png` berisi 1 varian
wakil tiap metode (Otsu, region t10, kmeans k2, watershed r0.7, snakes default).

## Cara jalan

```bash
# dari root repo
pip install -r requirements.txt
python segmentation/main.py
```

Script memindai semua `.png/.jpg/.jpeg` di `preprocessing-image/data/`,
memproses satu per satu, menyimpan plot dengan backend `Agg` (tanpa display).

## Import sebagai modul

```python
from src import (
    thresholding,
    region_growing,
    kmeans_segmentation,
    watershed_segmentation,
    active_contour_seg,
)
```
