# Preprocessing Citra X-ray Lutut

## Kasus

7 citra X-ray sendi lutut grayscale di `data/` (`0.jpg`, `2.jpg`, `3.jpg`, `4.jpeg`, `5.png`, `6.jpg`, `7.png`).

Masalah pada citra mentah:

- Kontras rendah, rentang intensitas menumpuk — batas tulang dan celah sendi (femur–tibia) kurang tegas.
- Detail trabekula / tepi tulang samar.
- Noise yang mengganggu saat kontras dinaikkan.

Tujuan: bandingkan 5 teknik enhancement + denoising untuk melihat mana yang paling memperjelas struktur tulang sebelum analisis lanjut.

## Metode vs masalah

| Teknik | Parameter diuji | Menjawab masalah |
|--------|-----------------|------------------|
| Histogram Equalization | — | Meratakan sebaran intensitas global |
| CLAHE | clip 1.0, 2.0, 4.0 | Peningkatan kontras lokal tanpa over-amplify noise |
| Gamma Correction | γ 0.5, 1.0, 2.0 | Koreksi kecerahan (γ<1 mencerahkan, γ>1 menggelapkan) |
| Median Filter | kernel 3, 5, 7 | Reduksi noise salt-and-pepper, jaga tepi |
| Gaussian Filter | kernel 3×3, 5×5, 7×7 | Penghalusan noise umum |

## Output

Per citra input menghasilkan 6 file di `output/`:

```
{nama}_histogram_equalization.png
{nama}_clahe_comparison.png
{nama}_gamma_correction.png
{nama}_median_filter.png
{nama}_gaussian_filter.png
{nama}_histograms_comparison.png
```

Contoh: `5.png` → `5_clahe_comparison.png`, dst.

## Cara jalan

```bash
# dari root repo
pip install -r requirements.txt
python preprocessing-image/main.py
```

Script memindai semua `.png/.jpg/.jpeg` di `data/`, memproses satu per satu, menyimpan plot dengan backend `Agg` (tanpa display).

## Soal 2: Enhancement citra infected + perbandingan kuantitatif

Script `soal2.py` memakai 3 citra unik pertama dari `dataset/infected/`
(`Image_001.jpg`, `Image_003.jpg`, `Image_005.jpg` — file dataset berpasangan
duplikat: 001=002, 003=004, dst., jadi dipilih yang unik).

Metode: HE, CLAHE (clip 2.0), Gamma Correction (0.5/1.0/2.0).
Metrik per citra per metode: mean, std (kontras RMS), entropi histogram.

```bash
.venv/bin/python preprocessing-image/soal2.py
```

Output di `output/soal2/`: `{nama}_before_after.png` (citra + histogram),
`comparison_all.png` (ringkasan), `metrics.csv` (18 baris).

Dokumentasi per metode (Metode, Code, Hasil, Analisis):

- [Histogram Equalization](docs/histogram-equalization.md)
- [CLAHE](docs/clahe.md)
- [Gamma Correction](docs/gamma-correction.md)
- [Perbandingan Hasil + Kesimpulan](docs/perbandingan-kesimpulan.md)

## Import sebagai modul

```python
from src import (
    histogram_equalization,
    clahe,
    gamma_correction,
    median_filter,
    gaussian_filter,
)
```
