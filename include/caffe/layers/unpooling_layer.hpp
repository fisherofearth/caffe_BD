#ifndef CAFFE_UNPOOLING_LAYER_HPP_
#define CAFFE_UNPOOLING_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/neuron_layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

/*
 * @brief Does unpooling operation on the network like Zeiler's paper in ECCV 2014
 * TODO(mariolew) Through documentation on the useage of unpooling layer. 
 */
 template <typename Dtype>
 class UnpoolingLayer : public Layer <Dtype> {
     public:
     explicit UnpoolingLayer(const LayerParameter &param)
        : Layer<Dtype>(param) {}
     virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
     virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);
     virtual inline const char* type() const { return "Unpooling";  }
     //virtual inline int MinBottomBlobs() const { return 1;  }
     //virtual inline int MaxBottomBlobs() const { return 2;  }
     virtual inline int ExactNumBottomBlobs() const { return 2; }
     virtual inline int ExactNumTopBlobs() const { return 1;  }
     
     protected:
     virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);
    // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    //                          const vector<Blob<Dtype>*>& top);
     //virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      //                         const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    //    NOT_IMPLEMENTED;                               
   // }
     virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
     
    // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    //                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

     int kernel_h_, kernel_w_;
     int stride_h_, stride_w_;
     int pad_h_, pad_w_;
     int channels_;
     int height_, width_;
     int unpooled_height_, unpooled_width_;
     bool global_pooling_;
  
 };


}  // namespace caffe

#endif  // CAFFE_UNPOOLING_LAYER_HPP_
