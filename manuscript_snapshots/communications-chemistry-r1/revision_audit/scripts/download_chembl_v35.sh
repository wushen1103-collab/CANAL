#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$ROOT/data/raw/chembl_35"
URL="https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_35/chembl_35_sqlite.tar.gz"
TAR="$RAW_DIR/chembl_35_sqlite.tar.gz"

mkdir -p "$RAW_DIR" "$ROOT/logs"

echo "[download] $URL"
if command -v aria2c >/dev/null 2>&1; then
  aria2c -x 8 -s 8 -c -d "$RAW_DIR" -o "$(basename "$TAR")" "$URL" 2>&1 | tee "$ROOT/logs/download_chembl_v35.log"
else
  curl -L -C - --retry 8 --retry-delay 10 -o "$TAR" "$URL" 2>&1 | tee "$ROOT/logs/download_chembl_v35.log"
fi

sha256sum "$TAR" | tee "$RAW_DIR/chembl_35_sqlite.tar.gz.sha256"

if [ ! -f "$RAW_DIR/chembl_35/chembl_35.db" ] && ! find "$RAW_DIR" -name 'chembl_35.db' | grep -q .; then
  echo "[extract] $TAR"
  tar -xzf "$TAR" -C "$RAW_DIR"
fi

DB_PATH="$(find "$RAW_DIR" -name 'chembl_35.db' | head -1)"
if [ -z "$DB_PATH" ]; then
  echo "Could not locate chembl_35.db after extraction" >&2
  exit 1
fi
echo "$DB_PATH" | tee "$RAW_DIR/DB_PATH.txt"
