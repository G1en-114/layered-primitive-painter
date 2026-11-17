#!/bin/bash
# SCNet: CLIP B/16 embeddings of every prompt's source image, from decomposer reconstructions (no source pixels here)
R=/root/private_data; S=$R/seq/v3_mix
cd $R/lpp_dev
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 LPP_RASTER=native LPP_EXT_BUILD=$R/ext_build_v2
python -u tools/source_embeddings.py --seq $S/face_t0 $S/face_t2 $S/anime_t0 $S/anime_t1 $S/anime_t2 $S/anime_t3 \
  --text-encoder $R/models/chinese-clip-vit-base-patch16 --out $R/source_emb_b16.npz > $R/source_emb.log 2>&1
