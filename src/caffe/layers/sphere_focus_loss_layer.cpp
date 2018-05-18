#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/sphere_focus_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SphereFocusLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  this->axis_ = this->layer_param_.sphere_focus_loss_param().axis();
  this->alpha_ = this->layer_param_.sphere_focus_loss_param().alpha();
  this->start_iter_ = this->layer_param_.sphere_focus_loss_param().start_iter();
  this->end_iter_ = this->layer_param_.sphere_focus_loss_param().end_iter();
  this->power_ = this->layer_param_.sphere_focus_loss_param().power();
  this->step_ = this->layer_param_.sphere_focus_loss_param().step();
  this->change_type_ = this->layer_param_.sphere_focus_loss_param().change_type();
  this->iter_ = 0;

  CHECK(this->end_iter_ > this->start_iter_) << "The end iter must be bigger than the start iter.";
  
  this->output_num_ = bottom[0]->count(0, this->axis_);
  this->feat_dim_ = bottom[0]->count(this->axis_);
  this->batch_size_ = bottom[1]->count(0, this->axis_);

  CHECK_EQ(this->feat_dim_, bottom[1]->count(this->axis_)) 
    << "bottom[0] and bottom[1] must have the same feat_dim!";

  this->blobs_.resize(1);
  this->param_propagate_down_.resize(this->blobs_.size());

  // For lambda
  this->blobs_[0].reset(new Blob<Dtype>({1}));
  this->blobs_[0]->mutable_cpu_data()[0] = 0.;
  this->param_propagate_down_[0] = false;
}

template <typename Dtype>
void SphereFocusLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);

  vector<int> top_shape = {1};
  top[1]->Reshape(top_shape);
  top[1]->ShareData(*(this->blobs_[0]));

  vector<int> distance_shape = {this->batch_size_};
  this->distance_.Reshape(distance_shape);
}

template <typename Dtype>
void SphereFocusLossLayer<Dtype>::UpdateLambda() {
  if (this->change_type_ == SphereFocusLossParameter_ChangeType_LINEAR) {
    const Dtype margin = 1. / (this->end_iter_ - this->start_iter_) * this->step_;
    if (this->iter_ <= this->start_iter_) {
      this->blobs_[0]->mutable_cpu_data()[0] = 0.;
    } else if (this->iter_ <= this->end_iter_) {
      this->blobs_[0]->mutable_cpu_data()[0] = (this->iter_ - this->start_iter_) / this->step_ * margin; 
    } else {
      this->blobs_[0]->mutable_cpu_data()[0] = 1.;
    }
  } else if (this->change_type_ == SphereFocusLossParameter_ChangeType_EXPONENT) {
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
void SphereFocusLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void SphereFocusLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(SphereFocusLossLayer);
#endif

INSTANTIATE_CLASS(SphereFocusLossLayer);
REGISTER_LAYER_CLASS(SphereFocusLoss);

}  // namespace caffe
