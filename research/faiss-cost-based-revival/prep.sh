#!/bin/bash
# Downloads and prepares the datasets of the PR thread with the thread's own protocol.
set -uxo pipefail
cd ~/work/faiss-pr5915
mkdir -p raw data weiss
# The preparation script of the PR thread (dedup, 20,000 held-out rows, z-score).
[ -s weiss/datasets.py ] || curl -sSL --fail -o weiss/datasets.py \
  https://raw.githubusercontent.com/eyal-weiss/emeans-net-gain-revival/de3febc057c307379e485602d52582ec836c0de9/research/datasets.py
cd raw
get() { [ -s "$2" ] || curl -sSL --fail -o "$2" "$1"; }
get https://ndownloader.figshare.com/files/5976045 kddcup.data.gz
get https://archive.ics.uci.edu/static/public/116/us+census+data+1990.zip "us+census+data+1990.zip"
get https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip "individual+household+electric+power+consumption.zip"
get https://archive.ics.uci.edu/static/public/148/statlog+shuttle.zip "statlog+shuttle.zip"
get https://download.geonames.org/export/dump/allCountries.zip allCountries.zip
ls -la
cd ..
for name in shuttle_z power_z geonames census_z kdd_z; do
  [ -f data/$name/meta.json ] || .pixi/envs/default/bin/python -I weiss/datasets.py prepare $name --raw raw --out data
done
echo PREP_DONE
