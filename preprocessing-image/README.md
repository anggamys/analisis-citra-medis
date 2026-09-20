# Preprocessing Citra Medis

Implementasi teknik preprocessing citra untuk analisis citra medis.

## Struktur Folder

```
preprocessing-image/
├── data/
│   └── 616156.png
├── output/
│   ├── clahe_comparison.png
│   ├── gamma_correction.png
│   ├── gaussian_filter.png
│   ├── histogram_equalization.png
│   ├── histograms_comparison.png
│   └── median_filter.png
├── src/
│   ├── __init__.py
│   ├── histogram_equalization.py
│   ├── clahe.py
│   ├── gamma_correction.py
│   ├── median_filter.py
│   └── gaussian_filter.py
├── main.py
├── requirements.txt
└── README.md
```

## Penggunaan

```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan demonstrasi
python main.py
```

Semua hasil plot disimpan di folder `output/`.

## Import sebagai Modul

```python
from src import (
    histogram_equalization,
    clahe,
    gamma_correction,
    median_filter,
    gaussian_filter
)
```

## Teknik Preprocessing

| Teknik | Fungsi | Deskripsi |
|--------|--------|-----------|
| Histogram Equalization | `histogram_equalization(image)` | Menyebar distribusi intensitas agar kontras lebih merata |
| CLAHE | `clahe(image, clip_limit, tile_grid_size)` | HE lokal dengan batas clipping untuk hindari noise berlebih |
| Gamma Correction | `gamma_correction(image, gamma, c)` | Transformasi power-law untuk brighten/darken citra |
| Median Filter | `median_filter(image, kernel_size)` | Reduksi noise salt-and-pepper dengan ambil median tetangga |
| Gaussian Filter | `gaussian_filter(image, kernel_size, sigma)` | Penghalusan citra dengan konvolusi Gaussian kernel |
