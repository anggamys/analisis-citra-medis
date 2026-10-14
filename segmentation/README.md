# Segmentasi Citra USG Infected (Soal 3)

## Kasus

3 citra USG acak dari `dataset/infected/` (berbeda tiap run, diambil dari
isi unik via md5). Hanya file berpola `image<angka>.jpg`
(mis. `image11977.jpg`); varian rotate/augmentasi dikecualikan.
Nama terpilih selalu dicetak ke console, catat untuk laporan.
Untuk run reproduksibel: `SAMPLE_SEED=7 ...`.

Masalah: pisahkan struktur anatomis (folikel hipoekoik gelap vs jaringan
echogenic terang) dari background agar region mencurigakan bisa diukur pada
analisis lanjut. Tantangan: speckle noise USG dan artefak garis tepi citra.

## Metode (5 metode sesuai maksud Soal 3)

"Region Growing" di soal tertulis terpotong dua baris sehingga terbaca
seperti dua metode — sebenarnya satu metode. Sama untuk "Clustering
(K-Means)".

| Metode                  | Modul                  | Parameter diuji                                                       |
| ----------------------- | ---------------------- | --------------------------------------------------------------------- |
| Thresholding            | `thresholding.py`      | Otsu, fixed=127, adaptive-Gaussian                                    |
| Region Growing          | `growing.py`           | thresh 20, 40, 60 (seed terang+homogen, tepi & saturasi dikecualikan) |
| Clustering (K-Means)    | `clustering_kmeans.py` | k=2, 3, 4                                                             |
| Watershed               | `watershed.py`         | dist_ratio 0.5, 0.7                                                   |
| Active Contour (Snakes) | `snakes.py`            | (α, β) = (0.015,10), (0.05,5), (0.01,20)                              |

## Output

Per citra input menghasilkan 6 file di `output/`:

```
{nama}_thresholding.png
{nama}_growing.png
{nama}_clustering_kmeans.png
{nama}_watershed.png
{nama}_snakes.png
{nama}_summary.png
```

Contoh: `Image_003.jpg` → `Image_003_thresholding.png`, dst.
`*_summary.png` berisi Asli + 1 varian wakil tiap metode
(Otsu, growing t40, clustering k-means k2,
watershed r0.7, snakes default).

## Perbandingan untuk laporan

Tidak ada ground-truth mask di `dataset/`, jadi perbandingan dilakukan
kualitatif per citra:

| Metode                  | Objek terpisah? | Boundary jelas? | Noise ikut? | Over-segmentation? |
| ----------------------- | --------------- | --------------- | ----------- | ------------------ |
| Thresholding            |                 |                 |             |                    |
| Region Growing          |                 |                 |             |                    |
| Clustering (K-Means)    |                 |                 |             |                    |
| Watershed               |                 |                 |             |                    |
| Active Contour (Snakes) |                 |                 |             |                    |

## Cara jalan

```bash
# dari root repo
.venv/bin/python segmentation/main.py

# atau pin sampel agar sama antar run
SAMPLE_SEED=7 .venv/bin/python segmentation/main.py
```

Script mengambil 3 citra acak dari `dataset/infected/`,
menyimpan plot dengan backend `Agg` (tanpa display).

## Import sebagai modul

```python
from src import (
    thresholding,
    growing_segmentation,
    clustering_kmeans,
    watershed_segmentation,
    snakes_segmentation,
)
```
