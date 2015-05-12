#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

// HYQ entire file adapted from base_data_layer.cu

template <typename Dtype>
void ImageDataMultLabelLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BasePrefetchingDataLayer<Dtype>::JoinPrefetchThread();

  top[0]->Reshape(this->prefetch_data_.num(), this->prefetch_data_.channels(),
      this->prefetch_data_.height(), this->prefetch_data_.width());

  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
      top[0]->mutable_gpu_data());

  if (this->output_labels_) {
    //caffe_copy(prefetch_label_.count(), 
    //           prefetch_label_.cpu_data(),
    //           top[1]->mutable_gpu_data());
    for(int label_id = 0; label_id < num_labels_; label_id++){
      caffe_copy(prefetch_labels_[label_id]->count(), 
                 prefetch_labels_[label_id]->cpu_data(),
                 top[1+label_id]->mutable_gpu_data());
    }
  }
  BasePrefetchingDataLayer<Dtype>::CreatePrefetchThread();
}

INSTANTIATE_LAYER_GPU_FORWARD(ImageDataMultLabelLayer);

}// namespace caffe
