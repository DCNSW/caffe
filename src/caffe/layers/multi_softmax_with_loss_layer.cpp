#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/multi_softmax_with_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultiSoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  LossLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void MultiSoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "Multi_softmax_loss_layer: "
      << "output of inner_product_layer must match number of labels.";
  blob_num_ = bottom[0]->shape(0);
  channel_num_ = bottom[0]->shape(1);
}

template <typename Dtype>
void MultiSoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const Dtype* data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  Dtype loss = 0;
  for (int i = 0; i < blob_num_; i++) {
    for (int j = 0; j < channel_num_; j++) {
      // data -> 0~1
      /***
      Dtype temp = Dtype(0.0);
      Dtype prob = data[i * channel_num_ + j];
      temp = Dtype(1) + exp(Dtype(0.0) - prob);
      prob = Dtype(1) / temp;
      if (label[i * channel_num_ + j] == Dtype(1.0)){
          loss -= log(std::max(prob,
                               Dtype(FLT_MIN)));
      } else {
          //loss -= log(std::max(Dtype(1.0) - data[i * channel_num_ + j],
          //                     Dtype(FLT_MIN)));
          loss -= log(std::max(Dtype(1.0) - prob,
                               Dtype(FLT_MIN)));
      }
      ***/
      Dtype label_value = (label[i * channel_num_ + j] + Dtype(1.0)) / Dtype(2.0);
      loss -= data[i * channel_num_ + j] * (label_value - (data[i * channel_num_ + j] >= 0)) - 
        log(1 + exp(data[i * channel_num_ + j] - 2 * (data[i * channel_num_ + j] >= 0) * data[i * channel_num_ + j]));
      //++count
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / Dtype(blob_num_ * channel_num_);
}

template <typename Dtype>
void MultiSoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* data = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    for (int i = 0; i < blob_num_; i++){
      for (int j = 0; j < channel_num_; j++){
        //Dtype temp = Dtype(0.0);
        Dtype prob = data[i * channel_num_ + j];
        //temp = Dtype(1.0) + exp(Dtype(0.0) - prob);
        //prob = Dtype(1.0) / temp;
        /***
        bottom_diff[i * channel_num_ + j] = label_value - (tanh(prob / Dtype(2.0)) + Dtype(1.0)) / Dtype(2.0);
        caffe_mul<Dtype>(1, prob, Dtype(-1.0), temp);
        caffe_exp<Dtype>(1, temp, prob);
        caffe_add<Dtype>(1, prob, Dtype(1.0), temp);
        caffe_div<Dtype>(1, Dtype(1.0), temp, prob);
        ***/
        Dtype label_value = (label[i * channel_num_ + j] + Dtype(1.0)) / Dtype(2.0);
        bottom_diff[i * channel_num_ + j] = (tanh(prob / Dtype(2.0)) + Dtype(1.0)) / Dtype(2.0) - label_value;
      }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / Dtype(blob_num_ * channel_num_);
    caffe_scal(blob_num_ * channel_num_, loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(MultiSoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(MultiSoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(MultiSoftmaxWithLoss);

}  // namespace caffe
