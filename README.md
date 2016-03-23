# caffe_BD
Before using this project on boundary detection, please set your GPU available.

####SETUP DATA
0. In a new terminal, go to directory: caffe_BD/data/BSDS500/
0. Convert BSDS500 data<br />
  ./convert_ground_truth_mat2image.py<br />
  ./generate_lst.py

####TRAIN
0. Go to directory: caffe_BD/
0. Train model:<br />
  ./examples/BSDS500/BSDS500_train.sh

####TEST
0. Go to directory: caffe_BD/
0. Test an image using trained model:<br />
  python BSDS500_test.py
