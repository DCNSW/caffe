#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/euler_contrast_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EulerContrastLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  
  this->axis_ = this->layer_param_.euler_contrast_loss_param().axis();
  this->alpha_ = this->layer_param_.euler_contrast_loss_param().alpha();
  this->start_iter_ = this->layer_param_.euler_contrast_loss_param().start_iter();
  this->end_iter_ = this->layer_param_.euler_contrast_loss_param().end_iter();
  this->power_ = this->layer_param_.euler_contrast_loss_param().power();
  this->step_ = this->layer_param_.euler_contrast_loss_param().step();
  this->eps_ = this->layer_param_.euler_contrast_loss_param().eps();
  this->change_type_ = this->layer_param_.euler_contrast_loss_param().change_type();
  this->scale_min_ = this->layer_param_.euler_contrast_loss_param().scale_min();
  this->iter_ = 0;

  this->output_num_ = bottom[0]->count(0, this->axis_);
  this->feat_dim_ = bottom[0]->count(this->axis_);
  this->batch_size_ = bottom[1]->count(0, this->axis_);
  
  this->blobs_.resize(2);
  this->param_propagate_down_.resize(this->blobs_.size());
  
  // For scale parameter
  this->blobs_[0].reset(new Blob<Dtype>({this->output_num_}));
  
  // Fill the scale
  shared_ptr<Filler<Dtype> > scale_filler(GetFiller<Dtype>(
    this->layer_param_.euler_contrast_loss_param().scale_filler()));
  scale_filler->Fill(this->blobs_[0].get());
  caffe_scal<Dtype>(this->blobs_[0]->count(), sqrt(this->feat_dim_), this->blobs_[0]->mutable_cpu_data());

  this->param_propagate_down_[0] = true;

  // For lambda
  this->blobs_[1].reset(new Blob<Dtype>({1}));
  this->blobs_[1]->mutable_cpu_data()[0] = 0.;
  this->param_propagate_down_[1] = false;
}

template <typename Dtype>
void EulerContrastLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);

  // Mean scale top
  top[1]->Reshape({1});

  // Lambda top
  top[2]->Reshape({1});
  top[2]->ShareData(*(this->blobs_[1]));
  
  vector<int> distance_shape = {this->batch_size_, this->output_num_};
  this->distance_.Reshape(distance_shape); // Shape --> (Batch_Size, Output_Num)

  vector<int> loss_shape = {this->batch_size_};
  this->loss_.Reshape(loss_shape); // Shape --> (Batch_Size)

  vector<int> center_shape = {this->output_num_, this->feat_dim_};
  this->center_.Reshape(center_shape);

  vector<int> feat_temp_shape = {this->batch_size_, this->output_num_, this->feat_dim_};
  this->feat_temp_.Reshape(feat_temp_shape);
}

template <typename Dtype>
void EulerContrastLossLayer<Dtype>::UpdateLambda() {
  if (this->change_type_ == EulerContrastLossParameter_ChangeType_LINEAR) {
    const Dtype margin = 1. / (this->end_iter_ - this->start_iter_) * this->step_;
    if (this->iter_ <= this->start_iter_) {
      this->blobs_[1]->mutable_cpu_data()[0] = 0.;
    } else if (this->iter_ <= this->end_iter_) {
      this->blobs_[1]->mutable_cpu_data()[0] = (this->iter_ - this->start_iter_) / this->step_ * margin; 
    } else {
      this->blobs_[1]->mutable_cpu_data()[0] = 1.;
    }
  } else if (this->change_type_ == EulerContrastLossParameter_ChangeType_EXPONENT) {
    if (this->iter_ <= this->start_iter_) {
      this->blobs_[1]->mutable_cpu_data()[0] = 0.;
    } else if (this->iter_ <= this->end_iter_) {
      const int exponent = (this->iter_ - this->start_iter_) / this->step_;
      this->blobs_[1]->mutable_cpu_data()[0] = 1. - pow(this->power_, exponent);
    } else {
      this->blobs_[1]->mutable_cpu_data()[0] = 1.; 
    }
  } else {
    LOG(FATAL) << "Unknown change type.";
  }
  this->iter_ += 1;
}

template <typename Dtype>
void EulerContrastLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void EulerContrastLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(EulerContrastLossLayer);
#endif

INSTANTIATE_CLASS(EulerContrastLossLayer);
REGISTER_LAYER_CLASS(EulerContrastLoss);

}  // namespace caffe
