#include <algorithm>
#include <math.h>
#include <vector>
#include <iostream>

#include "caffe/layers/euler_contrast_loss_layer.hpp"
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
__global__ void ComputeDistance(const int nthreads, const int feat_dim, const int output_num,
  const Dtype* center_data, const Dtype* feat_data, Dtype* distance_data) {
  CUDA_KERNEL_LOOP(index, nthreads) { // nthreads = Batch_size * Output_Num
    const int batch_id = index / output_num;
    const int class_id = index % output_num;
    const Dtype* center_data_temp = center_data + class_id * feat_dim;
    const Dtype* feat_data_temp = feat_data + batch_id * feat_dim;
    
    distance_data[index] = 0.;
    for (int feat_id = 0; feat_id < feat_dim; feat_id++) {
      distance_data[index] += (center_data_temp[feat_id] - feat_data_temp[feat_id]) * 
        (center_data_temp[feat_id] - feat_data_temp[feat_id]);
    }
  }
}

template <typename Dtype>
__global__ void ComputeLoss(const int nthreads, const int output_num, const Dtype eps,
  const Dtype* distance_data, const Dtype* label_data, Dtype* loss_data) {
  CUDA_KERNEL_LOOP(index, nthreads) { // nthreads = Batch_Size
    const int batch_id = index;
    const int label_id = label_data[batch_id];
    const Dtype numerator = distance_data[batch_id * output_num + label_id];
    
    Dtype denominator = 0.;
    for (int class_id = 0; class_id < output_num; class_id++) {
      denominator += (class_id != label_id) ? distance_data[batch_id * output_num + class_id] : 0;
    }

    denominator = denominator / (output_num - 1); // Normalize sum
    denominator += eps; // Escape from dividing zero
    loss_data[batch_id] = numerator / denominator;
  }
}

template <typename Dtype>
void EulerContrastLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  const Dtype* weight_data = bottom[0]->gpu_data();
  const Dtype* feat_data = bottom[1]->gpu_data();
  const Dtype* label_data = bottom[2]->gpu_data();
  
  Dtype* scale_data = this->blobs_[0]->mutable_gpu_data();
  Dtype* distance_data = this->distance_.mutable_gpu_data();
  Dtype* center_data = this->center_.mutable_gpu_data();
  Dtype* loss_data = this->loss_.mutable_gpu_data();

  // Clip scale
  const int scale_nthreads = this->blobs_[0]->count();
  ClipScale<Dtype><<<CAFFE_GET_BLOCKS(scale_nthreads), CAFFE_CUDA_NUM_THREADS>>>(
    scale_nthreads, this->scale_min_, scale_data);

  // Compute center
  const int center_nthreads = this->center_.count();
  ComputeCenter<Dtype><<<CAFFE_GET_BLOCKS(center_nthreads), CAFFE_CUDA_NUM_THREADS>>>(
    center_nthreads, this->feat_dim_, weight_data, scale_data, center_data);

  // Compute distance between different pairs of feature and center vectors,
  // so we have Batch_size * Output_Num kinds of pairs.
  const int distance_nthreads = this->distance_.count();
  ComputeDistance<Dtype><<<CAFFE_GET_BLOCKS(distance_nthreads), CAFFE_CUDA_NUM_THREADS>>>(
    distance_nthreads, this->feat_dim_, this->output_num_, center_data, feat_data, distance_data);
  
  // Compute each feature loss towards different clusters
  const int loss_nthreads = this->loss_.count();
  ComputeLoss<Dtype><<<CAFFE_GET_BLOCKS(loss_nthreads), CAFFE_CUDA_NUM_THREADS>>>(
    loss_nthreads, this->output_num_, this->eps_, distance_data, label_data, loss_data);
  
  // Compute loss
  Dtype loss = 0;
  caffe_gpu_asum<Dtype>(this->loss_.count(), loss_data, &loss);
  top[0]->mutable_cpu_data()[0] = 0.5 * loss / this->batch_size_;

  // Compute mean scale
  Dtype sum_scale = 0;
  caffe_gpu_asum<Dtype>(this->blobs_[0]->count(), this->blobs_[0]->gpu_data(), &sum_scale);
  top[1]->mutable_cpu_data()[0] = sum_scale / this->blobs_[0]->count();
}

template <typename Dtype>
__global__ void ComputeCenterGrad(const int nthreads, const int batch_size, const int output_num, const int feat_dim,
  const Dtype eps, const Dtype* center_data, const Dtype* feat_data, const Dtype* label_data,
  const Dtype* distance_data, Dtype* center_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) { // nthreads = Output_Num * Feat_Dim
    const int center_id = index / feat_dim;
    const int feat_id = index % feat_dim;
    
    // Compute equal part and unequal part
    int equal_count = 0; int unequal_count = 0;
    Dtype equal = 0; Dtype unequal = 0;
    
    for (int batch_id = 0; batch_id < batch_size; batch_id++) {
      const int label_id = label_data[batch_id];
      
      Dtype denominator = 0;
      for (int class_id = 0; class_id < output_num; class_id++) {
        denominator += (class_id != label_id) ? distance_data[batch_id * output_num + class_id] : 0;
      }
      denominator = denominator / (output_num - 1) + eps;

      if (label_id == center_id) {
        equal += (center_data[index] - feat_data[batch_id * feat_dim + feat_id]) / denominator;
        equal_count++;
      } else {
        unequal += ((center_data[index] - feat_data[batch_id * feat_dim + feat_id]) * 
          distance_data[batch_id * output_num + label_id] / (output_num - 1)) / (denominator * denominator);
        unequal_count++;
      }
    }
    equal = equal / (equal_count + 1);
    unequal = unequal / (unequal_count + 1);
    center_diff[index] = equal - unequal;
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
__global__ void ComputeFeatTempGrad(const int nthreads, const int feat_dim, const int output_num,
  const Dtype eps, const Dtype* center_data, const Dtype* feat_data, const Dtype* label_data, 
  const Dtype* distance_data, Dtype* feat_temp_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) { // nthreads = Batch_Size * Feat_Dim 
    const int batch_id = index / feat_dim;
    const int feat_id = index % feat_dim;
    const int label_id = label_data[batch_id];
    
    Dtype denominator = 0.;
    for (int class_id = 0; class_id < output_num; class_id++) {
      denominator += (class_id != label_id) ? distance_data[batch_id * output_num + class_id] : 0;
    }
    denominator = denominator / (output_num - 1) + eps;

    for (int class_id = 0; class_id < output_num; class_id++) {
      // feat_temp_ => (Batch_size, Output_Num, Feat_Dim)
      Dtype* feat_temp_diff_temp = feat_temp_diff + batch_id * output_num * feat_dim + class_id * feat_dim;
      feat_temp_diff_temp[feat_id] = feat_data[batch_id * feat_dim + feat_id] - center_data[class_id * feat_dim + feat_id];
      if (class_id == label_id) {
        feat_temp_diff_temp[feat_id] = feat_temp_diff_temp[feat_id] / denominator;
      } else {
        feat_temp_diff_temp[feat_id] *= (distance_data[batch_id * output_num + label_id] / (output_num - 1));
        feat_temp_diff_temp[feat_id] *= (-1.0 / (denominator * denominator));
      }
    }
  }
}

template <typename Dtype>
__global__ void ComputeFeatGrad(const int nthreads, const int feat_dim, const int output_num,
  const Dtype* weight_data, const Dtype* label_data, Dtype* feat_temp_diff, Dtype* feat_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) { // nthreads = Batch_Size
    const int batch_id = index;
    const int label_id = label_data[batch_id];
    const Dtype* weight_data_temp = weight_data + label_id * feat_dim;
    
    for (int class_id = 0; class_id < output_num; class_id++) {
      if (class_id == label_id) { continue; }
      
      Dtype* feat_temp_diff_temp = feat_temp_diff + batch_id * output_num * feat_dim + class_id * feat_dim;
      
      // Compute inner product
      Dtype weight_inner_product = 0.;
      for (int feat_id = 0; feat_id < feat_dim; feat_id++) {
        weight_inner_product += -1 * weight_data_temp[feat_id] * feat_temp_diff_temp[feat_id];
      }

      // Compute final feature diff
      for (int feat_id = 0; feat_id < feat_dim; feat_id++) {
        if (weight_inner_product <= 0) {
          feat_temp_diff_temp[feat_id] += -2.0 * weight_inner_product * weight_data_temp[feat_id];
        }
      }
    }

    for (int feat_id = 0; feat_id < feat_dim; feat_id++) {
      feat_diff[batch_id * feat_dim + feat_id] = 0;
      for (int class_id = 0; class_id < output_num; class_id++) {
        feat_diff[batch_id * feat_dim + feat_id] +=
          feat_temp_diff[batch_id * output_num * feat_dim + class_id * feat_dim + feat_id];
      }
    }
  }
}

template <typename Dtype>
void EulerContrastLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  // Update Lambda
  this->UpdateLambda();
  const Dtype lambda = this->blobs_[1]->cpu_data()[0];
  const Dtype loss_weight = top[0]->cpu_diff()[0];

  const Dtype* weight_data = bottom[0]->gpu_data();
  const Dtype* feat_data = bottom[1]->gpu_data();
  const Dtype* label_data = bottom[2]->gpu_data();
  const Dtype* distance_data = this->distance_.gpu_data();
  const Dtype* center_data = this->center_.gpu_data();
  Dtype* center_diff = this->center_.mutable_gpu_diff();

  // Compute center diff
  const int center_nthreads = this->center_.count();
   ComputeCenterGrad<Dtype><<<CAFFE_GET_BLOCKS(center_nthreads), CAFFE_CUDA_NUM_THREADS>>>(
    center_nthreads, this->batch_size_, this->output_num_, this->feat_dim_, this->eps_, 
    center_data, feat_data, label_data, distance_data, center_diff);

  // Compute diff for scale parameter
  if (this->param_propagate_down_[0]) {
    Dtype* scale_diff = this->blobs_[0]->mutable_cpu_diff();
    const int scale_nthreads = this->output_num_;
    ComputeScaleGrad<Dtype><<<CAFFE_GET_BLOCKS(scale_nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      scale_nthreads, this->feat_dim_, center_diff, weight_data, scale_diff);
    
    const Dtype scale = this->alpha_ * loss_weight * lambda;
    caffe_gpu_scal<Dtype>(this->blobs_[0]->count(), scale, scale_diff);
  }

  // Compute diff for weight
  if (propagate_down[0]) {
    const Dtype* scale_data = this->blobs_[0]->gpu_data();
    Dtype* weight_diff = bottom[0]->mutable_gpu_diff();

    const int weight_nthreads = bottom[0]->count();
    ComputeWeightGrad<Dtype><<<CAFFE_GET_BLOCKS(weight_nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      weight_nthreads, this->feat_dim_, center_diff, scale_data, weight_diff);

    const Dtype scale = this->alpha_ * loss_weight * lambda;
    caffe_gpu_scal<Dtype>(bottom[0]->count(), scale, weight_diff);
  }

  // Compute diff for feature
  if (propagate_down[1]) {
    Dtype* feat_temp_diff = this->feat_temp_.mutable_gpu_data();
    Dtype* feat_diff = bottom[1]->mutable_gpu_diff();
    
    // Compute two parts diff
    const int feat_temp_nthreads = bottom[1]->count();
    ComputeFeatTempGrad<Dtype><<<CAFFE_GET_BLOCKS(feat_temp_nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      feat_temp_nthreads, this->feat_dim_, this->output_num_, this->eps_, 
      center_data, feat_data, label_data, distance_data, feat_temp_diff);

    const int feat_nthreads = this->batch_size_;
    ComputeFeatGrad<Dtype><<<CAFFE_GET_BLOCKS(feat_nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      feat_nthreads, this->feat_dim_, this->output_num_, 
      weight_data, label_data, feat_temp_diff, feat_diff);
    
    const Dtype scale = loss_weight * lambda / this->batch_size_;
    caffe_gpu_scal<Dtype>(bottom[1]->count(), scale, feat_diff);
  }

  if (propagate_down[2]) {
    LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs.";
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EulerContrastLossLayer);

}  // namespace caffe
