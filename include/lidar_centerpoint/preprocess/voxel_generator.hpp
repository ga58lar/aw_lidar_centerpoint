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

#ifndef LIDAR_CENTERPOINT__PREPROCESS__VOXEL_GENERATOR_HPP_
#define LIDAR_CENTERPOINT__PREPROCESS__VOXEL_GENERATOR_HPP_

#include <lidar_centerpoint/centerpoint_config.hpp>
#include <lidar_centerpoint/preprocess/pointcloud_densification.hpp>

#include <memory>
#include <vector>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

namespace centerpoint
{
class VoxelGeneratorTemplate
{
public:
  explicit VoxelGeneratorTemplate(
    const DensificationParam & param, const CenterPointConfig & config);

  virtual std::size_t generateSweepPoints(std::vector<float> & points) = 0;

  bool enqueuePointCloud(
    const std::vector<Eigen::Vector4d> & input_pointcloud, const Eigen::Isometry3d & tf, const double & timestamp);

protected:
  std::unique_ptr<PointCloudDensification> pd_ptr_{nullptr};

  CenterPointConfig config_;
  std::array<float, 6> range_;
  std::array<int, 3> grid_size_;
  std::array<float, 3> recip_voxel_size_;
};

class VoxelGenerator : public VoxelGeneratorTemplate
{
public:
  using VoxelGeneratorTemplate::VoxelGeneratorTemplate;

  std::size_t generateSweepPoints(std::vector<float> & points) override;
};

}  // namespace centerpoint

#endif  // LIDAR_CENTERPOINT__PREPROCESS__VOXEL_GENERATOR_HPP_
