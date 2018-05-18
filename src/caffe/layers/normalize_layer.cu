#include <iostream>
#include <vector>

#include "caffe/layers/normalize_layer.hpp"
#include "caffe/util/math_functions.hpp"

using namespace std;

namespace caffe {

template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* temp_data = this->temp_.mutable_gpu_data();
  Dtype* normal_data = this->normal_.mutable_gpu_data(); 
  // Element-wise power temp_data = bottom_data^2
  caffe_gpu_powx<Dtype>(bottom[0]->count(), bottom_data, 2, temp_data);
  // Mathematical Formula: normal_data = temp_data * vec_one_data
  // normal_data --> (Feat_Num, 1)
  // temp_data --> (Feat_Num, Feat_Dim)
  // vect_one_data --> (Feat_Dim, 1)
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
    this->feat_num_, 1, this->feat_dim_,
    (Dtype)1., temp_data, this->vec_one_.gpu_data(),
    (Dtype)0., normal_data);
  // Add a small bias to escape dividing zero
  caffe_gpu_add_scalar<Dtype>(this->normal_.count(), this->eps_, normal_data);
  // Get the length of feature vectors
  caffe_gpu_sqrt<Dtype>(this->normal_.count(), normal_data, normal_data);
  // Mathematical Formula: temp_data = normal_data * vec_one_data
  // temp_data --> (Feat_Num, Feat_Dim)
  // normal_data --> (Feat_Num, 1)
  // vec_one_data --> (1, Feat_Dim)
  // Boradcast normal_data for element-wise dividing
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
    this->feat_num_, this->feat_dim_, 1,
    (Dtype)1., normal_data, this->vec_one_.gpu_data(),
    (Dtype)0., temp_data);
  // Element-wise dividing for normalization
  caffe_gpu_div<Dtype>(bottom[0]->count(), bottom_data, temp_data, top_data);
  // cout << "normal_data[0] = " << this->normal_.cpu_data()[0] << endl;
  // cout << "bottom_data[0] = " << bottom[0]->cpu_data()[0] << endl;
  // cout << "feat_dim = " << this->feat_dim_ << endl;
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* temp_data = this->temp_.gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* temp_diff = this->temp_.mutable_gpu_diff();
  // Element-wise multiple 
  caffe_gpu_mul<Dtype>(bottom[0]->count(), top_diff, top_data, temp_diff);
  // Mathematical Formula: normal_diff = temp_diff * vec_one_data
  // normal_diff --> (Feat_Num, 1)
  // temp_diff --> (Feat_Num, Feat_Dim)
  // vec_one_data --> (Feat_Dim, 1)
  // Sum slice-wise value to compute inner product between top_data and top_diff
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
    this->feat_num_, 1, this->feat_dim_,
    (Dtype)1., temp_diff, this->vec_one_.gpu_data(),
    (Dtype)0., this->normal_.mutable_gpu_diff());
  // Mathematical Formula: temp_diff = normal_diff * vec_one_data
  // temp_diff --> (Feat_Num, Feat_Dim)
  // normal_diff --> (Feat_Num, 1)
  // vec_one_data --> (1, Feat_Dim)
  // Boradcast for element-wise mutltipling
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
    this->feat_num_, this->feat_dim_, 1,
    (Dtype)1., this->normal_.gpu_diff(), this->vec_one_.gpu_data(),
    (Dtype)0., temp_diff);
  // Element-wise multiple
  caffe_gpu_mul<Dtype>(bottom[0]->count(), top_data, temp_diff, temp_diff);
  caffe_gpu_sub<Dtype>(bottom[0]->count(), top_diff, temp_diff, temp_diff);
  caffe_gpu_div<Dtype>(bottom[0]->count(), temp_diff, temp_data, bottom_diff);
  
  // cout << "bottom_diff[0] = " << bottom[0]->cpu_diff()[0] << endl;
  // cout << "top_diff[0] = " << top[0]->cpu_diff()[0] << endl;
  // abort();
  // caffe_gpu_set<Dtype>(bottom[0]->count(), 0, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(NormalizeLayer);

}  // namespace caffe
