#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/multi_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultiAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  /***
  top_k_ = this->layer_param_.accuracy_param().top_k();

  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }
  ***/
}

template <typename Dtype>
void MultiAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  /***
  CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
      << "top_k must be less than or equal to the number of classes.";
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  ***/
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "Multi_accuracy_layer: "
      << "output of inner_product_layer must match number of labels.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
  /***
  if (top.size() > 1) {
    // Per-class accuracy is a vector; 1 axes.
    vector<int> top_shape_per_class(1);
    top_shape_per_class[0] = bottom[0]->shape(label_axis_);
    top[1]->Reshape(top_shape_per_class);
    nums_buffer_.Reshape(top_shape_per_class);
  }
  ***/
}

template <typename Dtype>
void MultiAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int num_blobs = bottom[0]->shape(0);
  const int num_labels = bottom[0]->shape(1);
  int count = 0;
  for (int i = 0; i < num_blobs; ++i) {
    for (int j = 0; j < num_labels; ++j) {
      const Dtype data_value = bottom_data[i * num_labels + j];
      const int label_value = static_cast<int>(bottom_label[i * num_labels + j]);

      if ((data_value >= 0) && (label_value == 1))
        ++accuracy;
      if ((data_value < 0) && (label_value == -1))
        ++accuracy;
      ++count;
      /***
      Dtype temp = bottom_data[i * num_labels + j];
      Dtype prob = 0;
      caffe_mul(1, temp, Dtype(-1.0), temp);
      caffe_exp(1, temp, temp);
      caffe_add(1, temp, Dtype(1.0), temp);
      caffe_div(1, Dtype(1.0), temp, prob);
      //const int label_value = 
      //    static_cast<int>(bottom_label[i * num_labels + j]);
      if ((std::max(bottom_data[i * num_labels + j], Dtype(0.5)) == Dtype(0.5)) && (label_value == -1))
            ++accuracy;
      if ((std::max(bottom_data[i * num_labels + j], Dtype(0.5)) == bottom_data[i * num_labels + j]) && (label_value == 1))
            ++accuracy;
      ++count;
      ***/
    }
  }
  /***
  vector<Dtype> maxval(top_k_+1);
  vector<int> max_id(top_k_+1);
  if (top.size() > 1) {
    caffe_set(nums_buffer_.count(), Dtype(0), nums_buffer_.mutable_cpu_data());
    caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
  }
  int count = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      const int label_value =
          static_cast<int>(bottom_label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      if (top.size() > 1) ++nums_buffer_.mutable_cpu_data()[label_value];
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_labels);
      // Top-k accuracy
      std::vector<std::pair<Dtype, int> > bottom_data_vector;
      for (int k = 0; k < num_labels; ++k) {
        bottom_data_vector.push_back(std::make_pair(
            bottom_data[i * dim + k * inner_num_ + j], k));
      }
      std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
      // check if true label is in top k predictions
      for (int k = 0; k < top_k_; k++) {
        if (bottom_data_vector[k].second == label_value) {
          ++accuracy;
          if (top.size() > 1) ++top[1]->mutable_cpu_data()[label_value];
          break;
        }
      }
      ++count;
    }
  }
  ***/

  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = accuracy / count;
  /***
  if (top.size() > 1) {
    for (int i = 0; i < top[1]->count(); ++i) {
      top[1]->mutable_cpu_data()[i] =
          nums_buffer_.cpu_data()[i] == 0 ? 0
          : top[1]->cpu_data()[i] / nums_buffer_.cpu_data()[i];
    }
  }
  ***/
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(MultiAccuracyLayer);
REGISTER_LAYER_CLASS(MultiAccuracy);

}  // namespace caffe
