#!/usr/bin/env bash
export LD_LIBRARY_PATH="/mnt/scratch/fbsnpat/envs/biophysics-research-agent/lib:${LD_LIBRARY_PATH:-}"
exec /mnt/scratch/fbsnpat/bot/VLAB2/HDOCKlite-v1.1/hdock "$@"
