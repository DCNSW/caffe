#include <algorithm>
#include <cfloat>
#include <math.h>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/euler_focus_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EulerFocusLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  this->axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.euler_focus_loss_param().axis());
  this->alpha_ = this->layer_param_.euler_focus_loss_param().alpha();
  this->scale_min_ = this->layer_param_.euler_focus_loss_param().scale_min();

  this->start_iter_ = this->layer_param_.euler_focus_loss_param().start_iter();
  this->end_iter_ = this->layer_param_.euler_focus_loss_param().end_iter();
  this->power_ = this->layer_param_.euler_focus_loss_param().power();
  this->step_ = this->layer_param_.euler_focus_loss_param().step();
  this->change_type_ = this->layer_param_.euler_focus_loss_param().change_type();
  this->iter_ = 0;

  CHECK(this->end_iter_ > this->start_iter_) << "The end iter must be bigger than the start iter.";

  this->output_num_ = bottom[0]->count(0, this->axis_);
  this->feat_dim_ = bottom[0]->count(this->axis_);
  this->batch_size_ = bottom[1]->count(0, this->axis_);

  CHECK_EQ(this->feat_dim_, bottom[1]->count(this->axis_))
    << "bottom[0] and bottom[1] must have the same feat_dim!";

  this->blobs_.resize(2);
  this->param_propagate_down_.resize(this->blobs_.size());
  
  // For scale parameter
  this->blobs_[0].reset(new Blob<Dtype>({this->output_num_}));
  
  // Fill the scale
  shared_ptr<Filler<Dtype> > scale_filler(GetFiller<Dtype>(
    this->layer_param_.euler_focus_loss_param().scale_filler()));
  scale_filler->Fill(this->blobs_[0].get());
  caffe_scal<Dtype>(this->blobs_[0]->count(), sqrt(this->feat_dim_), this->blobs_[0]->mutable_cpu_data());

  this->param_propagate_down_[0] = true;

  // For lambda
  this->blobs_[1].reset(new Blob<Dtype>({1}));
  this->blobs_[1]->mutable_cpu_data()[0] = 0.;
  this->param_propagate_down_[1] = false;
}

template <typename Dtype>
void EulerFocusLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  LossLayer<Dtype>::Reshape(bottom, top);

  // Mean scale top
  top[1]->Reshape({1});

  // Lambda top
  top[2]->Reshape({1});
  top[2]->ShareData(*(this->blobs_[1]));

  vector<int> distance_shape = {this->batch_size_};
  this->distance_.Reshape(distance_shape);

  vector<int> center_shape = {this->output_num_, this->feat_dim_};
  this->center_.Reshape(center_shape);
}

template <typename Dtype>
void EulerFocusLossLayer<Dtype>::UpdateLambda() {
  if (this->change_type_ == EulerFocusLossParameter_ChangeType_LINEAR) {
    const Dtype margin = 1. / (this->end_iter_ - this->start_iter_) * this->step_;
    if (this->iter_ <= this->start_iter_) {
      this->blobs_[1]->mutable_cpu_data()[0] = 0.;
    } else if (this->iter_ <= this->end_iter_) {
      this->blobs_[1]->mutable_cpu_data()[0] = (this->iter_ - this->start_iter_) / this->step_ * margin; 
    } else {
      this->blobs_[1]->mutable_cpu_data()[0] = 1.;
    }
  } else if (this->change_type_ == EulerFocusLossParameter_ChangeType_EXPONENT) {
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
void EulerFocusLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void EulerFocusLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(EulerFocusLossLayer);
#endif

INSTANTIATE_CLASS(EulerFocusLossLayer);
REGISTER_LAYER_CLASS(EulerFocusLoss);

}  // namespace caffe
