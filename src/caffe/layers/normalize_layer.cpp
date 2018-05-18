#include <vector>

#include "caffe/layers/normalize_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NormalizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  this->eps_ = this->layer_param_.normalize_param().eps();
  this->axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.normalize_param().axis());
  this->feat_num_ = bottom[0]->count(0, this->axis_);
  this->feat_dim_ = bottom[0]->count(this->axis_);
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);

  vector<int> vec_one_shape = {this->feat_dim_};
  this->vec_one_.Reshape(vec_one_shape);
  caffe_set<Dtype>(this->vec_one_.count(), (Dtype)1., this->vec_one_.mutable_cpu_data());

  vector<int> normal_shape = {this->feat_num_};
  this->normal_.Reshape(normal_shape);

  vector<int> temp_shape = {this->feat_num_, this->feat_dim_};
  this->temp_.Reshape(temp_shape);
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(NormalizeLayer);
#endif

INSTANTIATE_CLASS(NormalizeLayer);
REGISTER_LAYER_CLASS(Normalize);

}  // namespace caffe
