# caffee_BD_cs519

Before using this project on boundary detection, please set your GPU available.

SETUP DATA

In a new terminal, go to directory: caffe_BD/data/BSDS500/
Convert BSDS500 data
./convert_ground_truth_mat2image.py
./generate_lst.py
TRAIN

Go to directory: caffe_BD/
Train model:
./examples/BSDS500/BSDS500_train.sh
TEST

Go to directory: caffe_BD/
Test an image using trained model:
python BSDS500_test.py
