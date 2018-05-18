#include <algorithm>
#include <math.h>
#include <vector>
#include <iostream>

#include "caffe/layers/sphere_focus_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

using namespace std;

namespace caffe {

template <typename Dtype>
__global__ void ComputeDistance(const int nthreads, const int feat_dim,
  const Dtype* weight_data, const Dtype* feat_data, const Dtype* label_data,
  Dtype* distance_data) {
  CUDA_KERNEL_LOOP(index, nthreads) { // nthreads = Batch_Size
    const int batch_id = index;
    const int label_id = label_data[batch_id]; 
    
    const Dtype* weight_data_temp = weight_data + label_id * feat_dim;
    const Dtype* feat_data_temp = feat_data + batch_id * feat_dim;
    
    distance_data[batch_id] = 0;
    for (int feat_id = 0; feat_id < feat_dim; feat_id++) {
      distance_data[batch_id] += (feat_data_temp[feat_id] - weight_data_temp[feat_id]) * 
        (feat_data_temp[feat_id] - weight_data_temp[feat_id]);
    }
  }
}

template <typename Dtype>
void SphereFocusLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight_data = bottom[0]->gpu_data();
  const Dtype* feat_data = bottom[1]->gpu_data();
  const Dtype* label_data = bottom[2]->gpu_data();
  Dtype* distance_data = this->distance_.mutable_gpu_data();

  const int distance_nthreads = this->distance_.count();
  ComputeDistance<Dtype><<<CAFFE_GET_BLOCKS(distance_nthreads), CAFFE_CUDA_NUM_THREADS>>>(
    distance_nthreads, this->feat_dim_, weight_data, feat_data, label_data, distance_data);
  
  Dtype loss = 0;
  caffe_gpu_asum<Dtype>(this->distance_.count(), distance_data, &loss);
  top[0]->mutable_cpu_data()[0] = 0.5 * loss / this->batch_size_;
}

template <typename Dtype>
__global__ void ComputeWeightGrad(const int nthreads, const int feat_dim, 
  const int batch_size, const Dtype* weight_data, const Dtype* feat_data, 
  const Dtype* label_data, Dtype* weight_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) { // nthreads = Output_Num * Feat_Dim
    const int class_id = index / feat_dim;
    const int feat_id = index % feat_dim;
    
    weight_diff[index] = 0;
    Dtype count = 0;
    for (int batch_id = 0; batch_id < batch_size; batch_id++) {
      if (label_data[batch_id] == class_id) {
        weight_diff[index] += (weight_data[index] - feat_data[batch_id * feat_dim + feat_id]);
        count++;
      }
    }
    weight_diff[index] = weight_diff[index] / (count + 1);
  }
}

template <typename Dtype>
__global__ void ComputeFeatGrad(const int nthreads, const int feat_dim,
  const Dtype* weight_data, const Dtype* feat_data, const Dtype* label_data,
  Dtype* feat_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) { // nthreads = Batch_Size * Feat_Dim
    const int batch_id = index / feat_dim;
    const int feat_id = index % feat_dim;
    const int label_id = label_data[batch_id];
    feat_diff[index] = (feat_data[index] - weight_data[label_id * feat_dim + feat_id]);
  }
}

template <typename Dtype>
void SphereFocusLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  // Update Lambda
  this->UpdateLambda();
  const Dtype lambda = this->blobs_[0]->cpu_data()[0];
  const Dtype loss_weight = top[0]->cpu_diff()[0];

  const Dtype* weight_data = bottom[0]->gpu_data();
  const Dtype* feat_data = bottom[1]->gpu_data();
  const Dtype* label_data = bottom[2]->gpu_data();
  
  // Compute normalized weight diff
  if (propagate_down[0]) {
    Dtype* weight_diff = bottom[0]->mutable_gpu_diff();
    
    const int weight_nthreads = bottom[0]->count();
    ComputeWeightGrad<Dtype><<<CAFFE_GET_BLOCKS(weight_nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      weight_nthreads, this->feat_dim_, this->batch_size_, weight_data,
      feat_data, label_data, weight_diff);
    
    const Dtype scale  = lambda * this->alpha_ * loss_weight;
    caffe_gpu_scal<Dtype>(bottom[0]->count(), scale, weight_diff);
  }

  // Compute normalized feature diff
  if (propagate_down[1]) {
    Dtype* feat_diff = bottom[1]->mutable_gpu_diff();
    
    const int feat_nthreads = bottom[1]->count();
    ComputeFeatGrad<Dtype><<<CAFFE_GET_BLOCKS(feat_nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      feat_nthreads, this->feat_dim_, weight_data, feat_data, label_data, feat_diff);
    
    const Dtype scale = lambda * loss_weight / this->batch_size_;
    caffe_gpu_scal<Dtype>( bottom[1]->count(), scale, feat_diff);
  }

  if (propagate_down[2]) {
    LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs.";
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SphereFocusLossLayer);

}  // namespace caffe
