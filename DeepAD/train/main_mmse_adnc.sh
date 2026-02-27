dataset="/media/mprl2/Hard Disk/zwl/ADR2/gazedata240507"
for i in $(seq 1 100)
do
  for questionnaire in "mmse" 
  do
    for n_fold in 0 1 2 3 4 5 6 7 8 9
    do
      python mainADNC.py \
      --dataset "$dataset" \
      --datacsv "adnc_10f4" \
      --questionnaire "$questionnaire" \
      --n_fold "$n_fold"  \
      --train-batch 20 \
      --test-batch 20  \
      --gpu-id "0"  \
      --arch sss\
      --epochs 100
    done
  done
done