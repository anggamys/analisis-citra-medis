# 7. Perbandingan Hasil

| Metode | Kelebihan | Keterbatasan | Hasil Segmentasi |
|---|---|---|---|
| Thresholding | Tercepat, Otsu otomatis tanpa parameter | Mask ikut mencakup non-target; fixed terlalu gelap, adaptive ber-noise | Cukup |
| Region Growing | Region dijamin terhubung; progresi thresh jelas | Sangat sensitif seed dan thresh; t40 masih parsial | Cukup |
| K-Means | `k=4` memisahkan folikel sebagai cluster sendiri | `k` manual; cluster tanpa makna anatomi bawaan | Baik |
| Watershed | Batas mengikuti struktur; over-segmentation terkendali via marker | Multi-region, bukan satu objek; frame tepi ikut jadi batas | Cukup |
| Active Contour (Snakes) | Boundary halus tertutup mengelilingi ROI ovarium; stabil di 3 varian | Tidak presisi per folikel; komputasi termahal | Baik |

> Berdasarkan hasil segmentasi, metode **K-Means (`k=4`)** memberikan hasil
> terbaik untuk target folikel karena satu cluster berkorespondensi langsung
> dengan gugusan folikel, sementara **Active Contour** terbaik untuk lokalisasi
> ROI ovarium karena boundary-nya halus dan tertutup. Metode lainnya
> (Thresholding, Region Growing, Watershed) memiliki keterbatasan berupa mask
> yang mencakup non-target atau region yang belum terisolasi sebagai satu
> objek — paling baik dipakai sebagai tahap awal, mis. Snakes untuk ROI
> lalu K-Means di dalamnya.

# 8. Kesimpulan

Dari lima metode klasik pada citra USG tersebut, **K-Means (`k=4`)** paling
berhasil memisahkan folikel dan **Active Contour** paling berhasil melokalisasi
ROI ovarium, sedangkan Thresholding, Region Growing, dan Watershed hanya
mencapai separasi parsial karena speckle noise dan intensitas yang tumpang
tindih — sehingga kombinasi metode (ROI lalu clustering di dalamnya) adalah
alur yang paling masuk akal untuk analisis lanjut.
