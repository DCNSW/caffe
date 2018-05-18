#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/modified_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ModifiedInnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.modified_inner_product_param().num_output();
  this->bias_term_ = this->layer_param_.modified_inner_product_param().bias_term();
  this->transpose_ = this->layer_param_.modified_inner_product_param().transpose();
  this->alpha_ = this->layer_param_.modified_inner_product_param().alpha();
  this->N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(this->layer_param_.modified_inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  this->K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    if (this->transpose_) {
      weight_shape[0] = this->K_;
      weight_shape[1] = this->N_;
    } else {
      weight_shape[0] = this->N_;
      weight_shape[1] = this->K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (this->bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  this->output_weight_ = this->layer_param_.modified_inner_product_param().output_weight();
  if (this->output_weight_) {
    CHECK_EQ(top.size(), 2) << "Top size must be 2.";
  } else {
    CHECK_EQ(top.size(), 1) << "Top size must be 1.";
  }
}

template <typename Dtype>
void ModifiedInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
    this->layer_param_.modified_inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K) << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  this->M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = this->N_;
  top[0]->Reshape(top_shape);
  if (this->output_weight_) {
    top[1]->ReshapeLike(*(this->blobs_[0]));
    top[1]->ShareData(*(this->blobs_[0])); // Output weight parameters for gap loss or cluster loss
    // top[1]->ShareDiff(*(this->blobs_[0]));
  }
  // Set up the bias multiplier
  if (this->bias_term_) {
    vector<int> bias_shape(1, this->M_);
    this->bias_multiplier_.Reshape(bias_shape);
    caffe_set(this->M_, Dtype(1), this->bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void ModifiedInnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
      this->M_, this->N_, this->K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  if (this->bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->M_, this->N_, 1, (Dtype)1.,
        this->bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void ModifiedInnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    if (this->transpose_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          this->K_, this->N_, this->M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          this->N_, this->K_, this->M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }
    // Added by Wang
    if (this->output_weight_) {
      caffe_axpy<Dtype>(this->blobs_[0]->count(), this->alpha_, 
        top[1]->cpu_diff(), this->blobs_[0]->mutable_cpu_diff());
    }
  }
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, this->M_, this->N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    if (this->transpose_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          this->M_, this->K_, this->N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          this->M_, this->K_, this->N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ModifiedInnerProductLayer);
#endif

INSTANTIATE_CLASS(ModifiedInnerProductLayer);
REGISTER_LAYER_CLASS(ModifiedInnerProduct);

}  // namespace caffe
