#include <algorithm>
#include <math.h>
#include <vector>

#include "caffe/layers/sphere_contrast_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ComputeDistance(const int nthreads, const int feat_dim, const int output_num,
  const Dtype* weight_data, const Dtype* feat_data, Dtype* distance) {
  CUDA_KERNEL_LOOP(index, nthreads) { // nthreads = Batch_size * Output_Num
    const int batch_id = index / output_num;
    const int cluster_id = index % output_num;
    const Dtype* weight_data_temp = weight_data + cluster_id * feat_dim;
    const Dtype* feat_data_temp = feat_data + batch_id * feat_dim;
    Dtype result = 0.;
    for (int i = 0; i < feat_dim; i++) {
      result += (weight_data_temp[i] - feat_data_temp[i]) * (weight_data_temp[i] - feat_data_temp[i]);
    }
    distance[index] = result;
  }
}

template <typename Dtype>
__global__ void ComputeLoss(const int nthreads, const int output_num, const Dtype eps,
  const Dtype* distance_data, const Dtype* label_data, Dtype* loss_data) {
  CUDA_KERNEL_LOOP(index, nthreads) { // nthreads = Batch_Size
    const int batch_id = index;
    const int cluster_id = label_data[batch_id];
    const Dtype numerator = distance_data[batch_id * output_num + cluster_id];
    Dtype denominator = 0.;
    for (int i = 0; i < output_num; i++) {
      denominator += (i != cluster_id) ? distance_data[batch_id * output_num + i] : 0;
    }
    denominator = denominator / (output_num - 1); // Normalize sum
    denominator += eps; // Escape from dividing zero
    loss_data[batch_id] = numerator / denominator;
  }
}

template <typename Dtype>
void SphereContrastLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight_data = bottom[0]->gpu_data();
  const Dtype* feat_data = bottom[1]->gpu_data();
  const Dtype* label_data = bottom[2]->gpu_data();
  Dtype* distance_data = this->distance_.mutable_gpu_data();
  Dtype* loss_data = this->loss_.mutable_gpu_data();

  // Compute distance between different pairs of feature and cluster vectors,
  // so we have Batch_size * Output_Num kinds of pairs.
  const int nthreads1 = this->distance_.count();
  ComputeDistance<Dtype><<<CAFFE_GET_BLOCKS(nthreads1), CAFFE_CUDA_NUM_THREADS>>>(
    nthreads1, this->feat_dim_, this->output_num_, weight_data, feat_data, distance_data);
  
  // Compute each feature loss towards different clusters
  const int nthreads2 = this->batch_size_;
  ComputeLoss<Dtype><<<CAFFE_GET_BLOCKS(nthreads2), CAFFE_CUDA_NUM_THREADS>>>(
    nthreads2, this->output_num_, this->eps_, distance_data, label_data, loss_data);
  
  Dtype loss = 0;
  caffe_gpu_asum<Dtype>(this->batch_size_, loss_data, &loss);
  top[0]->mutable_cpu_data()[0] = 0.5 * loss / this->batch_size_;
}

template <typename Dtype>
__global__ void ComputeWeightGrad(const int nthreads, const int output_num, const int feat_dim, const int batch_size,
  const Dtype eps, const Dtype* weight_data, const Dtype* feat_data, const Dtype* label_data,
  const Dtype* distance_data, Dtype* weight_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) { // nthreads = Output_Num * Feat_Dim
    const int cluster_id = index / feat_dim;
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

      if (label_id == cluster_id) {
        equal += (weight_data[index] - feat_data[batch_id * feat_dim + feat_id]) / denominator;
        equal_count++;
      } else {
        unequal += ((weight_data[index] - feat_data[batch_id * feat_dim + feat_id]) * 
          distance_data[batch_id * output_num + label_id] / (output_num - 1)) / (denominator * denominator);
        unequal_count++;
      }
    }
    equal = equal / (equal_count + 1);
    unequal = unequal / (unequal_count + 1);
    weight_diff[index] = equal - unequal;
  }
}

template <typename Dtype>
__global__ void ComputeFeatGrad(const int nthreads, const int feat_dim, const int output_num,
  const Dtype eps, const Dtype* weight_data, const Dtype* feat_data, const Dtype* distance_data, 
  const Dtype* label_data, Dtype* feat_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) { // nthreads = batch_size * feat_dim
    const int batch_id = index / feat_dim;
    const int feat_id = index % feat_dim;
    const int label_id = label_data[batch_id]; 
    Dtype denominator = 0.;
    // Compute left part
    for (int i = 0; i < output_num; i++) {
      denominator += (i != label_id) ? distance_data[batch_id * output_num + i] : 0;
    }
    denominator = denominator / (output_num - 1) + eps;
     // denominator = denominator + eps;
    Dtype left = feat_data[batch_id * feat_dim + feat_id] - weight_data[label_id * feat_dim + feat_id];
    left = left / denominator;
    // Compute right part
    Dtype numerator = 0.;
    for (int i = 0; i < output_num; i++) {
      numerator += (i != label_id) ? (feat_data[batch_id * feat_dim + feat_id] - weight_data[i * feat_dim + feat_id]) : 0;
    }
    numerator = numerator * distance_data[batch_id * output_num + label_id] / (output_num - 1);
    // numerator = numerator * distance_data[batch_id * output_num + label_id];
    const Dtype right = numerator / (denominator * denominator);
    feat_diff[index] = left - right;
  }
}

template <typename Dtype>
void SphereContrastLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  // Update Lambda
  this->UpdateLambda();
  const Dtype lambda = this->blobs_[0]->cpu_data()[0];
  const Dtype loss_weight = top[0]->cpu_diff()[0];
  
  const Dtype* weight_data = bottom[0]->gpu_data();
  const Dtype* feat_data = bottom[1]->gpu_data();
  const Dtype* label_data = bottom[2]->gpu_data();
  const Dtype* distance_data = this->distance_.gpu_data();
 

  if (propagate_down[0]) {
    const Dtype* vec_one = this->loss_.gpu_diff();
    Dtype* weight_diff = bottom[0]->mutable_gpu_diff();

    const int weight_nthreads = bottom[0]->count();
    ComputeWeightGrad<Dtype><<<CAFFE_GET_BLOCKS(weight_nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      weight_nthreads, this->output_num_, this->feat_dim_, this->batch_size_, this->eps_, 
      weight_data, feat_data, label_data, distance_data, weight_diff);

    const Dtype scale = lambda * this->alpha_ * loss_weight;
    caffe_gpu_scal<Dtype>(bottom[0]->count(), scale, weight_diff);
  }
  if (propagate_down[1]) {
    Dtype* feat_diff = bottom[1]->mutable_gpu_diff();
    
    const int feat_nthreads = bottom[1]->count();
    ComputeFeatGrad<Dtype><<<CAFFE_GET_BLOCKS(feat_nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      feat_nthreads, this->feat_dim_, this->output_num_, this->eps_, 
      weight_data, feat_data, distance_data, label_data, feat_diff);
    
    const Dtype scale = lambda * loss_weight / this->batch_size_;
    caffe_gpu_scal<Dtype>(bottom[1]->count(), scale, feat_diff);
  }
  if (propagate_down[2]) {
    LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs.";
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SphereContrastLossLayer);

}  // namespace caffe
