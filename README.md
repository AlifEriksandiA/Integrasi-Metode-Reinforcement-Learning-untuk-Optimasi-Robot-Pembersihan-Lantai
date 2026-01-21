# Smart Robot Cleaner dengan Q-Learning 🤖🧹

**Smart Robot Cleaner** adalah simulasi agen cerdas berbasis Python yang menggunakan algoritma **Reinforcement Learning (Q-Learning)**. Robot ini belajar secara mandiri bagaimana cara membersihkan seluruh lantai grid 10x10 sambil mengelola penggunaan baterai agar tidak mati di tengah jalan.

Proyek ini mendemonstrasikan penerapan AI sederhana dalam navigasi, pengambilan keputusan, dan manajemen sumber daya.

## 🌟 Fitur Utama

* **🧠 Q-Learning Agent**: Robot belajar dari pengalaman (*trial & error*) untuk memaksimalkan *reward*.
* **🔋 Manajemen Baterai**: Robot harus memutuskan kapan harus membersihkan dan kapan harus kembali ke *charging station* sebelum baterai habis.
* **🎮 Visualisasi Interaktif**: Dibangun menggunakan **Pygame** untuk melihat pergerakan robot secara real-time.
* **⚡ Mode Fast Train**: Fitur percepatan waktu untuk melatih robot ribuan episode dalam hitungan detik.
* **📈 Grafik Pembelajaran**: Visualisasi kurva pembelajaran (*learning curve*) menggunakan Matplotlib untuk melihat perkembangan kecerdasan robot.
* **💾 Save/Load Brain**: Otak robot (Q-Table) disimpan secara otomatis ke file `.pkl` sehingga latihan bisa dilanjutkan kapan saja.

## 🛠️ Prasyarat & Instalasi

Pastikan komputer Anda sudah terinstal **Python 3.x**.

Proyek ini membutuhkan beberapa library eksternal. Anda dapat menginstalnya dengan membuka terminal/CMD dan menjalankan perintah berikut:

```bash
pip install pygame numpy matplotlib

```

## 🚀 Cara Menjalankan Program

1. Pastikan Anda telah menyimpan kode program utama dengan nama file, misalnya `main.py`.
2. Buka terminal atau command prompt.
3. Arahkan ke folder tempat Anda menyimpan file tersebut.
4. Jalankan program dengan perintah:

```bash
python main.py

```

## 🎮 Panduan Penggunaan (GUI)

Setelah program berjalan, Anda akan melihat simulasi grid dan panel kontrol di sebelah kanan. Berikut fungsi tombol-tombolnya:

### 1. Tombol Hijau: [START TRAIN / RESUME]

* **Fungsi**: Memulai mode pelatihan cepat (*Fast Training*).
* **Cara Kerja**: Robot akan berlatih di latar belakang dengan kecepatan tinggi (tanpa animasi lambat).
* **Tips**: Gunakan ini di awal untuk membuat robot pintar dengan cepat. Tekan lagi untuk melanjutkan jika pelatihan berhenti.

### 2. Tombol Biru: [WATCH 1 EP]

* **Fungsi**: Mode demonstrasi/uji coba.
* **Cara Kerja**: Anda bisa menonton robot menyelesaikan 1 episode (misi) secara real-time dengan kecepatan normal.
* **Tips**: Gunakan ini setelah melakukan *Fast Train* untuk melihat seberapa pintar robot sekarang.

### 3. Tombol Kuning: [Lihat Grafik]

* **Fungsi**: Menampilkan statistik perkembangan.
* **Cara Kerja**: Membuka jendela pop-up grafik total *reward* per episode.
* **Analisis**: Grafik yang menanjak naik menandakan robot semakin pintar dalam mengambil keputusan.

## 🧠 Konsep AI (Under The Hood)

Robot ini menggunakan tabel Q-Learning untuk menentukan aksi terbaik.

### 1. State (Keadaan)

Robot mengamati lingkungan berdasarkan 4 parameter utama:

* **Posisi Robot**: Koordinat grid `(x, y)`.
* **Level Baterai**: Disederhanakan menjadi 3 level (`Low`, `Medium`, `High`).
* **Status Charging**: Sedang mengecas atau tidak.
* **Sensor Sekitar**: Informasi 4 arah (atas, bawah, kiri, kanan) apakah ada tembok atau lantai kotor.

### 2. Actions (Aksi)

Robot memiliki 5 pilihan aksi yang bisa diambil:

* `Up`, `Down`, `Left`, `Right` (Bergerak).
* `Charge` (Mengisi daya di station).

### 3. Reward System (Sistem Hadiah)

* **+25 Poin**: Membersihkan lantai kotor.
* **+100 Poin**: Berhasil cas pertama kali (insentif agar robot mau mengecas).
* **+1000 Poin**: Misi Selesai (Semua bersih & robot selamat).
* **-100 Poin**: Mati (Baterai habis).
* **-5 Poin**: Menabrak tembok.

## 📂 Struktur File

* `main.py`: Kode utama program yang berisi environment, agen, dan visualisasi.
* `robot_brain_smart.pkl`: File biner tempat menyimpan Q-Table (terbuat otomatis setelah latihan). Hapus file ini jika ingin mereset otak robot dari nol.

## 👤 Author
1. Alif Eriksandi Agustino
2. Muhammad Farkhan Fadillah
3. Arbi Yusuf Ramanda
4. Maulana Ihsan Maggio

* Teknik Komputer - Universitas Brawijaya
