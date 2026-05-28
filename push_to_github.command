#!/bin/bash
cd "$(dirname "$0")"

echo "======================================"
echo "  Push ke GitHub — Boxing RF Thesis"
echo "======================================"
echo ""
echo "Menambahkan file..."
git add .gitignore requirements.txt

echo "Membuat commit..."
git commit -m "Setup Streamlit Cloud deployment: update requirements & add .gitignore"

echo ""
echo "Push ke GitHub..."
git push origin main

echo ""
echo "======================================"
echo "  ✅ Selesai! Cek GitHub kamu:"
echo "  https://github.com/jpereiraboavida/boxing-rf-thesis"
echo "======================================"
echo ""
read -p "Tekan Enter untuk tutup..."
