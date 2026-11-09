# BAB III – PENUTUP

Bab ini merangkum seluruh pekerjaan pada proyek segmentasi nukleus berbasis
U-Net (dataset BBBC038) dan memberikan saran pengembangan. Seluruh angka
yang dikutip berasal dari eksperimen `experiments/baseline` yang sudah
dibahas pada [BAB II](BAB-II-PEMBAHASAN.md).

---

## 3.1 Kesimpulan dan Saran

### 3.1.1 Kesimpulan

1. **Sistem berjalan end-to-end.** Seluruh pipeline — penggabungan 29.461
   mask instance menjadi 670 mask biner, pemisahan 570/100 train/val,
   preprocessing dan augmentasi, training, evaluasi, hingga prediksi 3.019
   citra stage2 — terimplementasi sebagai proyek Python modular tanpa
   notebook, dengan konfigurasi dan checkpoint yang dapat direproduksi.
2. **Model mencapai kualitas segmentasi yang baik.** Training U-Net
   (base 64, depth 4, 31,0 juta parameter) berhenti pada epoch ke-56 karena
   early stopping dengan validation Dice terbaik **0,8773** (epoch 41).
   Pada test resmi: **Dice 0,8323 ± 0,1188 (stage1, 65 citra)** dan
   **Dice 0,8453 ± 0,1496 (stage2 scored, 106 citra)**, dengan IoU
   masing-masing 0,7274 dan 0,7545.
3. **Generalisasi terjaga pada data eksternal.** Mean Dice stage2 (data
   eksternal yang tidak pernah dilihat model) tidak lebih rendah dari
   stage1, ditopang distribusi Dice yang left-skewed — mayoritas citra
   bernilai ≥ 0,8 (49/65 pada stage1; 86/106 pada stage2).
4. **Karakter model konservatif pada stage1** (precision 0,8916 >
   recall 0,8034) dan sedikit lebih agresif pada stage2 (recall 0,8701 >
   precision 0,8601), sebagian dipengaruhi perbedaan threshold (0,55 vs
   0,50).
5. **Kegagalan terkonsentrasi pada citra sulit.** Ekor rendah distribusi
   Dice (terburuk 0,2364 pada stage1 dan 0,0934 pada stage2) menunjukkan
   kesalahan utama berupa nukleus bersentuhan yang menyatu, nukleus kecil
   yang hilang, dan region palsu pada artefak — keterbatasan yang melekat
   pada pilihan tugas segmentasi semantik biner.

Secara keseluruhan, tujuan proyek tercapai: model menghasilkan mask nukleus
per piksel dengan kualitas baik pada mayoritas citra, dan seluruh proses
terdokumentasi dengan metrik, kurva, dan visualisasi yang dapat
direproduksi.

### 3.1.2 Saran

1. **Post-processing pemisahan instance.** Tambahkan watershed berbasis
   distance transform atau arsitektur instance-aware (Mask R-CNN,
   StarDist) untuk memisahkan nukleus yang bersentuhan, sehingga hasil
   dapat dipakai untuk penghitungan sel dan analisis morfologi per sel.
2. **Konsistensi threshold.** Bekukan satu threshold dari validation set
   yang sama untuk semua test set agar perbandingan antar-stage benar-benar
   setara; selisih 0,55 vs 0,50 saat ini kecil tetapi menyulitkan analisis.
3. **Test-time augmentation dan ensemble.** Rata-ratakan probabilitas dari
   flip/rotasi (dan beberapa checkpoint) untuk meredam kegagalan pada citra
   sulit tanpa melatih ulang model.
4. **Stabilitas estimasi.** Gunakan k-fold cross-validation dan perbesar
   validation set agar pemilihan checkpoint dan threshold tidak rentan
   terhadap fluktuasi metrik dari 100 citra validasi.
5. **Evaluasi pada resolusi asli.** Hitung metrik pada resolusi penuh
   (bukan kanvas 256 × 256) untuk mengukur kualitas batas objek sebenarnya,
   dan pertimbangkan pipeline 16-bit utuh untuk citra dengan rentang
   dinamis tinggi.
6. **Pengembangan arsitektur.** Eksplorasi backbone yang lebih kuat
   (ResNet, EfficientNet), attention gate pada skip connection, atau
   deep supervision untuk mengekang ekor rendah distribusi Dice.

Pengembangan lanjutan yang paling berdampak dan sesuai urutan prioritas
adalah poin 1–3: ketiganya langsung menyasar kelemahan yang terukur pada
hasil eksperimen (gabungan instance, inkonsistensi threshold, dan citra
sulit) dengan biaya implementasi yang relatif rendah.
