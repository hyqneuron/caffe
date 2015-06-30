#include <opencv2/core/core.hpp>
#include "opencv2/opencv.hpp"

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {


// HYQ: entire file adapted from image_data_layer.cpp

template <typename Dtype>
ImageDataMultLabelLayer<Dtype>::~ImageDataMultLabelLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void ImageDataMultLabelLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_mult_label_param().new_height();
  const int new_width  = this->layer_param_.image_data_mult_label_param().new_width();
  const bool is_color  = this->layer_param_.image_data_mult_label_param().is_color();
  string root_folder = this->layer_param_.image_data_mult_label_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";

  // Read the file with filenames and labels
  num_labels_ = top.size()-1;
  const string& source = this->layer_param_.image_data_mult_label_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  // if prefix exists, we prepend it to all filenames
  string prefix = "";
  if(this->layer_param_.image_data_mult_label_param().has_image_location_prefix())
    prefix = this->layer_param_.image_data_mult_label_param().image_location_prefix();
  //int label;
  //while (infile >> filename >> label) {
  while (infile >> filename ) {
    vector<int> labels;
    labels.resize(num_labels_);
    // if using bounding box, first 4 numbers are bounding coordinates
    // left,top,width,height
    if(this->layer_param_.image_data_mult_label_param().use_bbox()){
      float left,top,width,height;
      infile >> left;
      infile >> top;
      infile >> width;
      infile >> height;
      bboxes_.push_back(std::make_tuple(left,top,width,height));
    }
    // now, the normal labels
    for(int label_id = 0; label_id<num_labels_; label_id++){
      int label;
      infile >> label;
      labels[label_id]=label;
    }
    lines_.push_back(std::make_pair(prefix+filename, labels));
  }

  if (this->layer_param_.image_data_mult_label_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";
  LOG(INFO) << "Last sample being: " << lines_.back().first;
  for(int label_id = 0; label_id<num_labels_; label_id++)
    LOG(INFO) << "Label: " << lines_.back().second[label_id];

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_mult_label_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_mult_label_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  LOG(INFO) << "##############################################";
  LOG(INFO) << "##############################################";
  LOG(INFO) << "size of top: " << top.size();
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  const int channels = cv_img.channels();
  const int height = cv_img.rows;
  const int width = cv_img.cols;
  // image
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.image_data_mult_label_param().batch_size();
  if (crop_size > 0) {
    top[0]->Reshape(batch_size, channels, crop_size, crop_size);
    this->prefetch_data_.Reshape(batch_size, channels, crop_size, crop_size);
    this->transformed_data_.Reshape(1, channels, crop_size, crop_size);
  } else {
    top[0]->Reshape(batch_size, channels, height, width);
    this->prefetch_data_.Reshape(batch_size, channels, height, width);
    this->transformed_data_.Reshape(1, channels, height, width);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  LOG(INFO) << "number of labels: " << num_labels_;
  vector<int> label_shape(1, batch_size);
  // we reshape each top blob
  // also build one prefetch buffer in prefetch_labels_ for each label
  prefetch_labels_.resize(num_labels_);
  for(int label_id = 0; label_id < num_labels_; label_id++){
    top[label_id+1]->Reshape(label_shape);
    //Blob<Dtype> prefetch_label_blob;
    prefetch_labels_[label_id].reset(new Blob<Dtype>());
    prefetch_labels_[label_id]->Reshape(label_shape);
  }
  // prefetch_label is no used by us
  this->prefetch_label_.Reshape(label_shape);
}

template <typename Dtype>
void ImageDataMultLabelLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// HYQ adapted from BasePrefetchingDataLayer
template <typename Dtype>
void ImageDataMultLabelLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  this->prefetch_data_.mutable_cpu_data();
  if (this->output_labels_) {
    this->prefetch_label_.mutable_cpu_data();
    // HYQ seems below is not absolutely necessary. added to be safe
    for(int label_id = 0; label_id < num_labels_; label_id++)
        prefetch_labels_[label_id]->mutable_cpu_data();
  }
  DLOG(INFO) << "Initializing prefetch";
  this->CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

// HYQ adapted from BasePrefetchingDataLayer
template <typename Dtype>
void ImageDataMultLabelLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  BasePrefetchingDataLayer<Dtype>::JoinPrefetchThread();
  DLOG(INFO) << "Thread joined";
  // Reshape to loaded data.
  top[0]->Reshape(this->prefetch_data_.num(), this->prefetch_data_.channels(),
      this->prefetch_data_.height(), this->prefetch_data_.width());
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    // HYQ
    //caffe_copy(prefetch_label_.count(), 
    //           prefetch_label_.cpu_data(),
    //           top[1]->mutable_cpu_data());
    for(int label_id = 0; label_id < num_labels_; label_id++){
      caffe_copy(prefetch_labels_[label_id]->count(), 
                 prefetch_labels_[label_id]->cpu_data(),
                 top[1+label_id]->mutable_cpu_data());
    }
  }
  // Start a new prefetch thread
  DLOG(INFO) << "CreatePrefetchThread";
  BasePrefetchingDataLayer<Dtype>::CreatePrefetchThread();
}
// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ImageDataMultLabelLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataMultLabelParameter image_data_mult_label_param = this->layer_param_.image_data_mult_label_param();
  const int batch_size = image_data_mult_label_param.batch_size();
  const int new_height = image_data_mult_label_param.new_height();
  const int new_width = image_data_mult_label_param.new_width();
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const bool is_color = image_data_mult_label_param.is_color();
  string root_folder = image_data_mult_label_param.root_folder();

  // Reshape on single input batches for inputs of varying dimension.
  if (batch_size == 1 && crop_size == 0 && new_height == 0 && new_width == 0) {
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        0, 0, is_color);
    this->prefetch_data_.Reshape(1, cv_img.channels(),
        cv_img.rows, cv_img.cols);
    this->transformed_data_.Reshape(1, cv_img.channels(),
        cv_img.rows, cv_img.cols);
  }

  Dtype* prefetch_data = this->prefetch_data_.mutable_cpu_data();
  //Dtype* prefetch_label= this->prefetch_label_.mutable_cpu_data();
  vector<Dtype*> prefetch_labels;
  for(int label_id=0; label_id<num_labels_; label_id++)
      prefetch_labels.push_back(this->prefetch_labels_[label_id]->mutable_cpu_data());

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    read_time += timer.MicroSeconds();
    timer.Start();

    // if using dynamic bounding box, get the crop first
    if(this->layer_param_.image_data_mult_label_param().use_bbox()){
      // if we're dealing with a bbox'ed image, here's what we do:
      // 1. read in image (already done before)
      // 2. crop to bounding box + 0.1
      // 3. resize cropped region to 100x100
      // 4. transformer crops again to 82x82
      float left,top,width,height;
      float right,bottom;
      std::tuple<float,float,float,float> bbox = bboxes_[lines_id_];
      // 2.1 making the bounding box 1.1 times its original size
      width  = std::get<2>(bbox);
      height = std::get<3>(bbox);
      left = std::get<0>(bbox);//-0.05*width;
      top  = std::get<1>(bbox);//-0.05*height;
      //width  += 0.1*width;
      //height += 0.1*height;
      right = left + width;
      bottom= top  + height;
      // 2.2 correct bounding box if it is out-of-bounds
      if(left<0) left = 0;
      if(right>1)right= 1;
      if(top<0) top=0;
      if(bottom>1) bottom=1;
      width = right-left;
      height= bottom-top;

      // 2.3 compute integer bbox coordinates
      int i_left, i_top, i_width, i_height;
      i_left = cv_img.cols * left;
      i_top  = cv_img.rows * top;
      i_width  = cv_img.cols * width;
      i_height = cv_img.rows * height;

      int pid = lines_[lines_id_].second[0];

      LOG(INFO)<< "ID=" << pid << " "
               << left << " "
               << top << " "
               << width << " "
               << height << " " ;
      LOG(INFO)<< "ID=" << pid << " "
               << i_left << " " 
               << i_top << " " 
               << i_width << " " 
               << i_height<< " "  
               << cv_img.rows<< " "  
               << cv_img.cols;

      // 2.4 crop bounding box
      cv::Mat img_bbox;
      cv::Rect roi(i_left, i_top, i_width, i_height);
      img_bbox = cv_img(roi);

      // 3. resize to 100x100
      cv::Mat img_resized;
      int resized_size = 100; //TODO use variable size
      cv::resize(img_bbox, img_resized, cv::Size(resized_size, resized_size)); 

      cv_img = img_resized;
      // we assume lines_ contain [(pid, class)]
      cv::imwrite(std::to_string(lines_[lines_id_].second[0])+".jpg", cv_img);
    }
    // Apply transformations (mirror, crop...) to the image
    int offset = this->prefetch_data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();
    //LOG(INFO) << "lines_id_: "<< lines_id_;
    //LOG(INFO) << "file: "<< lines_[lines_id_].first;

    //prefetch_label[item_id] = lines_[lines_id_].second;
    vector<int> labels = lines_[lines_id_].second;
    for(int label_id=0; label_id<num_labels_; label_id++)
        prefetch_labels[label_id][item_id] = labels[label_id];
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      LOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_mult_label_param().shuffle()) {
        LOG(INFO) << "Re-shuffling training samples";
        ShuffleImages();
      }
    }
  }
  // FIXME hack to get first batch of resized jpgs
  CHECK(false);
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(ImageDataMultLabelLayer, Forward);
#endif
INSTANTIATE_CLASS(ImageDataMultLabelLayer);
REGISTER_LAYER_CLASS(ImageDataMultLabel);

}  // namespace caffe
