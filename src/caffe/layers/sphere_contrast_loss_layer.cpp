#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/sphere_contrast_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SphereContrastLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  this->transpose_ = this->layer_param_.sphere_contrast_loss_param().transpose();
  this->alpha_ = this->layer_param_.sphere_contrast_loss_param().alpha();
  this->start_iter_ = this->layer_param_.sphere_contrast_loss_param().start_iter();
  this->end_iter_ = this->layer_param_.sphere_contrast_loss_param().end_iter();
  this->power_ = this->layer_param_.sphere_contrast_loss_param().power();
  this->step_ = this->layer_param_.sphere_contrast_loss_param().step();
  this->eps_ = this->layer_param_.sphere_contrast_loss_param().eps();
  this->change_type_ = this->layer_param_.sphere_contrast_loss_param().change_type();
  this->iter_ = 0;

  const vector<int> bottom_shape = bottom[0]->shape(); // weight shape
  CHECK_EQ(bottom_shape.size(), 2) << "bottom[0] must be two dimension.";
  if (this->transpose_) {
    this->output_num_ = bottom_shape[1];
    this->feat_dim_ = bottom_shape[0];
  } else {
    this->output_num_ = bottom_shape[0];
    this->feat_dim_ = bottom_shape[1];
  }
  this->batch_size_ = bottom[1]->shape(0);
  
  this->blobs_.resize(1);
  this->blobs_[0].reset(new Blob<Dtype>({1}));
  this->blobs_[0]->mutable_cpu_data()[0] = 0.;
  this->param_propagate_down_.resize(this->blobs_.size(), false);
}

template <typename Dtype>
void SphereContrastLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);

  vector<int> top_shape = {1};
  top[1]->Reshape(top_shape);
  top[1]->ShareData(*(this->blobs_[0]));
  
  vector<int> distance_shape = {this->batch_size_, this->output_num_};
  this->distance_.Reshape(distance_shape); // Shape --> (Batch_Size, Output_Num)

  vector<int> loss_shape = {this->batch_size_};
  this->loss_.Reshape(loss_shape); // Shape --> (Batch_Size)
  caffe_set<Dtype>(this->loss_.count(), (Dtype)1., this->loss_.mutable_cpu_diff());

  // vector<int> grad_shape = {this->batch_size_, this->output_num_, this->feat_dim_};
  // this->grad_.Reshape(grad_shape);
}

template <typename Dtype>
void SphereContrastLossLayer<Dtype>::UpdateLambda() {
  if (this->change_type_ == SphereContrastLossParameter_ChangeType_LINEAR) {
    const Dtype margin = 1. / (this->end_iter_ - this->start_iter_) * this->step_;
    if (this->iter_ <= this->start_iter_) {
      this->blobs_[0]->mutable_cpu_data()[0] = 0.;
    } else if (this->iter_ <= this->end_iter_) {
      this->blobs_[0]->mutable_cpu_data()[0] = (this->iter_ - this->start_iter_) / this->step_ * margin; 
    } else {
      this->blobs_[0]->mutable_cpu_data()[0] = 1.;
    }
  } else if (this->change_type_ == SphereContrastLossParameter_ChangeType_EXPONENT) {
    if (this->iter_ <= this->start_iter_) {
      this->blobs_[0]->mutable_cpu_data()[0] = 0.;
    } else if (this->iter_ <= this->end_iter_) {
      const int exponent = (this->iter_ - this->start_iter_) / this->step_;
      this->blobs_[0]->mutable_cpu_data()[0] = 1. - pow(this->power_, exponent);
    } else {
      this->blobs_[0]->mutable_cpu_data()[0] = 1.; 
    }
  } else {
    LOG(FATAL) << "Unknown change type.";
  }
  this->iter_ += 1;
}

template <typename Dtype>
void SphereContrastLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void SphereContrastLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(SphereContrastLossLayer);
#endif

INSTANTIATE_CLASS(SphereContrastLossLayer);
REGISTER_LAYER_CLASS(SphereContrastLoss);

}  // namespace caffe
