// Copyright 2021 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LIDAR_CENTERPOINT__PREPROCESS__POINTCLOUD_DENSIFICATION_HPP_
#define LIDAR_CENTERPOINT__PREPROCESS__POINTCLOUD_DENSIFICATION_HPP_

#include <list>
#include <string>
#include <utility>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

namespace centerpoint
{
class DensificationParam
{
public:
  DensificationParam(const unsigned int num_past_frames)
  : pointcloud_cache_size_(num_past_frames + /*current frame*/ 1)
  {
  }

  unsigned int pointcloud_cache_size() const { return pointcloud_cache_size_; }

private:
  unsigned int pointcloud_cache_size_{1};
};

struct PointCloudWithTransformTime
{
  std::vector<Eigen::Vector4f> pointcloud;
  Eigen::Isometry3f affine_past2world;
  double timestamp;
};

class PointCloudDensification
{
public:
  explicit PointCloudDensification(const DensificationParam & param);

  bool enqueuePointCloud(
    const std::vector<Eigen::Vector4f> & input_pointcloud, const Eigen::Isometry3f & tf, const double & timestamp);

  double getCurrentTimestamp() const { return current_timestamp_; }
  Eigen::Isometry3f getAffineWorldToCurrent() const { return affine_world2current_; }
  std::list<PointCloudWithTransformTime>::iterator getPointCloudCacheIter()
  {
    return pointcloud_cache_.begin();
  }
  bool isCacheEnd(std::list<PointCloudWithTransformTime>::iterator iter)
  {
    return iter == pointcloud_cache_.end();
  }
  unsigned int pointcloud_cache_size() const { return param_.pointcloud_cache_size(); }

private:
  void enqueue(const std::vector<Eigen::Vector4f> & msg, const Eigen::Isometry3f & affine, const double & timestamp);
  void dequeue();

  DensificationParam param_;
  double current_timestamp_{0.0};
  Eigen::Isometry3f affine_world2current_;
  std::list<PointCloudWithTransformTime> pointcloud_cache_;
};

}  // namespace centerpoint

#endif  // LIDAR_CENTERPOINT__PREPROCESS__POINTCLOUD_DENSIFICATION_HPP_
