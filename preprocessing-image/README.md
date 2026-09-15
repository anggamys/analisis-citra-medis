# Preprocessing Citra Medis

Implementasi teknik preprocessing citra untuk analisis citra medis.

## Struktur Folder

```
preprocessing-image/
├── src/
│   ├── __init__.py
│   ├── histogram_equalization.py
│   ├── clahe.py
│   ├── gamma_correction.py
│   ├── median_filter.py
│   └── gaussian_filter.py
├── main.py
├── test_preprocessing.py
├── requirements.txt
└── README.md
```

## Penggunaan

```bash
# Jalankan demonstrasi
python main.py
```

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
