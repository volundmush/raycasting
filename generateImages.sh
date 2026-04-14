#!/usr/bin/env bash

set -euo pipefail

modes=(basic reflection refraction glossy softshadow all)
views=(front left right top)

for mode in "${modes[@]}"; do
  for view in "${views[@]}"; do
    python3 rayTrace.py --mode "$mode" --view "$view" &
  done
done

wait
