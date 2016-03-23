Readme

Before using this project on boundary detection, please set your GPU available.

[SETUP DATA]/t
1. In a new terminal, go to directory: caffe_BD/data/BSDS500/
2. Convert BSDS500 data
  ./convert_ground_truth_mat2image.py
  ./generate_lst.py

[TRAIN]
3. Go to directory: caffe_BD/
4. Train model:
  ./examples/BSDS500/BSDS500_train.sh

[TEST]
5. Go to directory: caffe_BD/
6. Test an image using trained model:
  python BSDS500_test.py
