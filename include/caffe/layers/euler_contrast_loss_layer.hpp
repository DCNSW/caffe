#ifndef CAFFE_EULER_CONTRAST_LOSS_LAYER_HPP_
#define CAFFE_EULER_CONTRAST_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class EulerContrastLossLayer : public LossLayer<Dtype> {
 public:
  explicit EulerContrastLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "EulerContrastLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 3; }
  
 protected:
  void UpdateLambda();
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int output_num_;
  int feat_dim_;
  int batch_size_;
  
  int axis_;
  float alpha_;
  int start_iter_;
  int end_iter_;
  int iter_;
  float power_;
  int step_;
  float eps_;
  float scale_min_;
  
  EulerContrastLossParameter_ChangeType change_type_;
  
  Blob<Dtype> distance_; // Shape --> (Batch_Size, Output_Num)
  Blob<Dtype> loss_; // Shape --> (Batch_Size)
  Blob<Dtype> center_; // Shape --> (Output_Num, Feat_Dim)
  Blob<Dtype> feat_temp_; // Shape --> (Batch_Size, Output_Num, Feat_Dim)
};

}  // namespace caffe

#endif  // CAFFE_EULER_CONTRAST_LOSS_LAYER_HPP_
