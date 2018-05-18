#include <algorithm>
#include <math.h>
#include <vector>
#include <iostream>

#include "caffe/layers/euler_focus_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

using namespace std;

namespace caffe {

template <typename Dtype>
__global__ void ClipScale(const int nthreads, const Dtype scale_min, Dtype* scale_data) {
  CUDA_KERNEL_LOOP(index, nthreads) { // nthreads = Output_Num
    const int class_id = index;
    scale_data[class_id] = max(scale_data[class_id], scale_min);
  }
}

template <typename Dtype>
__global__ void ComputeCenter(const int nthreads, const int feat_dim,
  const Dtype* weight_data, const Dtype* scale_data, Dtype* center_data) {
  CUDA_KERNEL_LOOP(index, nthreads) { // nthreads = Output_Num * Feat_Dim
    const int class_id = index / feat_dim;
    center_data[index] = scale_data[class_id] * weight_data[index];
  }
}

template <typename Dtype>
__global__ void ComputeDistance(const int nthreads, const int feat_dim,
  const Dtype* center_data, const Dtype* feat_data, const Dtype* label_data,
  Dtype* distance_data) {
  CUDA_KERNEL_LOOP(index, nthreads) { // nthreads = Batch_Size
    const int batch_id = index;
    const int label_id = label_data[batch_id]; 
    const Dtype* center_data_temp = center_data + label_id * feat_dim;
    const Dtype* feat_data_temp = feat_data + batch_id * feat_dim;
    distance_data[batch_id] = 0.;
    for (int feat_id = 0; feat_id < feat_dim; feat_id++) {
      distance_data[batch_id] += (feat_data_temp[feat_id] - center_data_temp[feat_id]) * 
        (feat_data_temp[feat_id] - center_data_temp[feat_id]);
    }
  }
}

template <typename Dtype>
void EulerFocusLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight_data = bottom[0]->gpu_data();
  const Dtype* feat_data = bottom[1]->gpu_data();
  const Dtype* label_data = bottom[2]->gpu_data();
  
  Dtype* scale_data = this->blobs_[0]->mutable_gpu_data();
  Dtype* distance_data = this->distance_.mutable_gpu_data();
  Dtype* center_data = this->center_.mutable_gpu_data();

  // Clip scale
  const int scale_nthreads = this->blobs_[0]->count();
  ClipScale<Dtype><<<CAFFE_GET_BLOCKS(scale_nthreads), CAFFE_CUDA_NUM_THREADS>>>(
    scale_nthreads, this->scale_min_, scale_data);

  // Compute center
  const int center_nthreads = this->center_.count();
  ComputeCenter<Dtype><<<CAFFE_GET_BLOCKS(center_nthreads), CAFFE_CUDA_NUM_THREADS>>>(
    center_nthreads, this->feat_dim_, weight_data, scale_data, center_data);

  // Compute distance
  const int distance_nthreads = this->batch_size_;
  ComputeDistance<Dtype><<<CAFFE_GET_BLOCKS(distance_nthreads), CAFFE_CUDA_NUM_THREADS>>>(
    distance_nthreads, this->feat_dim_, center_data, feat_data, label_data, distance_data);
  
  Dtype loss = 0;
  caffe_gpu_asum<Dtype>(this->batch_size_, distance_data, &loss);
  top[0]->mutable_cpu_data()[0] = 0.5 * loss / this->batch_size_;

  // Compute mean scale
  Dtype sum_scale = 0;
  caffe_gpu_asum<Dtype>(this->blobs_[0]->count(), this->blobs_[0]->gpu_data(), &sum_scale);
  top[1]->mutable_cpu_data()[0] = sum_scale / this->blobs_[0]->count();
}

template <typename Dtype>
__global__ void ComputeCenterGrad(const int nthreads, const int feat_dim, 
  const int batch_size, const Dtype* center_data, const Dtype* feat_data, 
  const Dtype* label_data, Dtype* center_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) { // nthreads = Output_Num * Feat_Dim
    const int class_id = index / feat_dim;
    const int feat_id = index % feat_dim;
    center_diff[index] = 0;
    Dtype count = 0;
    for (int batch_id = 0; batch_id < batch_size; batch_id++) {
      if (label_data[batch_id] == class_id) {
        center_diff[index] += (center_data[index] - feat_data[batch_id * feat_dim + feat_id]);
        count += 1;
      }
    }
    center_diff[index] = center_diff[index] / (count + 1);
  }
}

template <typename Dtype>
__global__ void ComputeScaleGrad(const int nthreads, const int feat_dim,
  const Dtype* center_diff, const Dtype* weight_data, Dtype* scale_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) { // nthreads = Output_Num
    const int class_id = index;
    scale_diff[class_id] = 0.;
    for (int feat_id = 0; feat_id < feat_dim; feat_id++) {
      scale_diff[class_id] += center_diff[class_id * feat_dim + feat_id] * 
        weight_data[class_id * feat_dim + feat_id];
    }
  }
}

template <typename Dtype>
__global__ void ComputeWeightGrad(const int nthreads, const int feat_dim,
  const Dtype* center_diff, const Dtype* scale_data, Dtype* weight_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) { // nthreads = Output_Num * Feat_Dim
    const int class_id = index / feat_dim;
    weight_diff[index] = scale_data[class_id] * center_diff[index];
  }
}

template <typename Dtype>
__global__ void ComputeFeatGrad(const int nthreads, const int feat_dim,
  const Dtype* center_data, const Dtype* feat_data, const Dtype* label_data,
  Dtype* feat_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) { // nthreads = Batch_Size * Feat_Dim
    const int batch_id = index / feat_dim;
    const int feat_id = index % feat_dim;
    const int label_id = label_data[batch_id];
    feat_diff[index] = feat_data[index] - center_data[label_id * feat_dim + feat_id];
  }
}

template <typename Dtype>
void EulerFocusLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  // Update Lambda
  this->UpdateLambda();
  const Dtype lambda = this->blobs_[1]->cpu_data()[0];
  const Dtype loss_weight = top[0]->cpu_diff()[0];

  const Dtype* weight_data = bottom[0]->gpu_data();
  const Dtype* feat_data = bottom[1]->gpu_data();
  const Dtype* label_data = bottom[2]->gpu_data();
  const Dtype* center_data = this->center_.gpu_data();
  Dtype* center_diff = this->center_.mutable_gpu_diff();
  
  // Compute center diff
  const int center_nthreads = this->center_.count();
  ComputeCenterGrad<Dtype><<<CAFFE_GET_BLOCKS(center_nthreads), CAFFE_CUDA_NUM_THREADS>>>(
    center_nthreads, this->feat_dim_, this->batch_size_, center_data,
    feat_data, label_data, center_diff);

  // Compute scale diff
  if (this->param_propagate_down_[0]) {
    Dtype* scale_diff = this->blobs_[0]->mutable_cpu_diff();
    const int scale_nthreads = this->output_num_;
    ComputeScaleGrad<Dtype><<<CAFFE_GET_BLOCKS(scale_nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      scale_nthreads, this->feat_dim_, center_diff, weight_data, scale_diff);
    
    const Dtype scale = this->alpha_ * loss_weight * lambda;
    caffe_gpu_scal<Dtype>(this->blobs_[0]->count(), scale, scale_diff);
  }

  // Compute normalized weight diff
  if (propagate_down[0]) {
    const Dtype* scale_data = this->blobs_[0]->gpu_data();
    Dtype* weight_diff = bottom[0]->mutable_gpu_diff();

    const int weight_nthreads = bottom[0]->count();
    ComputeWeightGrad<Dtype><<<CAFFE_GET_BLOCKS(weight_nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      weight_nthreads, this->feat_dim_, center_diff, scale_data, weight_diff);

    const Dtype scale = this->alpha_ * loss_weight * lambda;
    caffe_gpu_scal<Dtype>(bottom[0]->count(), scale, weight_diff);
  }

  // Compute feature diff
  if (propagate_down[1]) {
    Dtype* feat_diff = bottom[1]->mutable_gpu_diff();
    const int feat_nthreads = bottom[1]->count();
    ComputeFeatGrad<Dtype><<<CAFFE_GET_BLOCKS(feat_nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      feat_nthreads, this->feat_dim_, center_data, feat_data, label_data, feat_diff);

    const Dtype scale = loss_weight * lambda / this->batch_size_;
    caffe_gpu_scal<Dtype>(bottom[1]->count(), scale, feat_diff);
  }

  if (propagate_down[2]) {
    LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs.";
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EulerFocusLossLayer);

}  // namespace caffe
