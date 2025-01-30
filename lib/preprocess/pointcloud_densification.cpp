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

#include "lidar_centerpoint/preprocess/pointcloud_densification.hpp"

#include <string>
#include <utility>

namespace centerpoint
{
PointCloudDensification::PointCloudDensification(const DensificationParam & param) : param_(param)
{
}

bool PointCloudDensification::enqueuePointCloud(
  const std::vector<Eigen::Vector4f> & pointcloud, const Eigen::Isometry3f & tf, const double & timestamp)
{
  if (param_.pointcloud_cache_size() > 1) {   
    enqueue(pointcloud, tf, timestamp);
  } else {
    enqueue(pointcloud, Eigen::Isometry3f::Identity(), timestamp);
  }

  dequeue();

  return true;
}

void PointCloudDensification::enqueue(
  const std::vector<Eigen::Vector4f> & msg,
  const Eigen::Isometry3f & affine_world2current,
  const double & timestamp)
{
  affine_world2current_ = affine_world2current;
  current_timestamp_ = timestamp;
  PointCloudWithTransformTime pointcloud = {msg, affine_world2current.inverse()};
  pointcloud_cache_.push_front(pointcloud);
}

void PointCloudDensification::dequeue()
{
  if (pointcloud_cache_.size() > param_.pointcloud_cache_size()) {
    pointcloud_cache_.pop_back();
  }
}

}  // namespace centerpoint
