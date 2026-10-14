# 7. Perbandingan Hasil

Nilai dari `output/soal2/metrics.csv` (citra `Image_003.jpg`):

| Metode | Kelebihan | Keterbatasan | Hasil Segmentasi |
|---|---|---|---|
| HE | Kontras global naik besar (std +54%), tanpa parameter | Noise ikut diperkuat, entropy turun, risiko washout | Cukup |
| CLAHE | Entropy tertinggi (7.611), kontras lokal alami | Perlu tuning clip/tile; efek tidak sedramatis HE | Baik |
| Gamma (0.5) | Mean tertinggi (109.24), murah, tanpa artefak spasial | Hanya geser kecerahan; γ=2.0 merusak (entropy 5.541) | Cukup |

> Berdasarkan hasil segmentasi, metode **CLAHE** memberikan hasil terbaik
> karena satu-satunya yang menaikkan entropy (detail bertambah, bukan
> hilang) sekaligus menjaga tampilan alami, sementara **HE** memiliki
> keterbatasan berupa noise yang ikut diperkuat dan **Gamma** hanya
> menggeser kecerahan tanpa menambah kontras lokal — sehingga CLAHE adalah
> preprocessing yang diteruskan ke tahap segmentasi.

# 8. Kesimpulan

Dari tiga metode enhancement pada citra USG tersebut, **CLAHE (clip 2.0)**
paling berhasil meningkatkan kualitas (entropy 7.111→7.611, folikel dan
tekstur jaringan paling jelas), sedangkan HE dan Gamma Correction hanya
mencapai perbaikan parsial karena noise yang terangkat dan hilangnya detail
pada γ ekstrem — sehingga CLAHE dipilih sebagai preprocessing sebelum
segmentasi pada Soal 3.
