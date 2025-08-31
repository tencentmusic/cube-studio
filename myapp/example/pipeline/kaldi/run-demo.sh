#!/usr/bin/env bash


. ./cmd.sh
. ./path.sh

stage=1

nj=10

test_sets="thchs"

. utils/parse_options.sh

if [ $stage -le 1 ]; then
  # make features
  mfccdir=mfcc_demo
  #corpora="aidatatang aishell magicdata primewords stcmds thchs"
  corpora="thchs"
  for c in $corpora; do
    
      steps/make_mfcc_pitch_online.sh --cmd "$train_cmd" --nj 20 \
        data/$c/train exp/make_mfcc/$c/train_demo $mfccdir/$c || exit 1
      steps/compute_cmvn_stats.sh data/$c/train \
        exp/make_mfcc/$c/train_demo $mfccdir/$c || exit 1
    
  done
fi


echo all train test done!
exit 0
