// Copyright 2022 TIER IV, Inc.
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

#include "lidar_centerpoint/utils.hpp"

#include <stdexcept>
#include <fstream>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

namespace centerpoint
{
// cspell: ignore divup
std::size_t divup(const std::size_t a, const std::size_t b)
{
  if (a == 0) {
    throw std::runtime_error("A dividend of divup isn't positive.");
  }
  if (b == 0) {
    throw std::runtime_error("A divisor of divup isn't positive.");
  }

  return (a + b - 1) / b;
}

std::vector<Eigen::Vector4f> loadBIN_kitti(const std::string & bin_path){
    // Open the file in binary mode
    std::ifstream file(bin_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + bin_path);
    }

    // Get the file size
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read the file into a buffer
    std::vector<float> buffer(size / sizeof(float));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Error reading file: " + bin_path);
    }
    constexpr double VERTICAL_ANGLE_OFFSET = (0.205 * M_PI) / 180.0;
    std::vector<Eigen::Vector4f> pointCloud;
    pointCloud.reserve(buffer.size()/3);
    std::vector<Eigen::Vector3d> points;
    for (size_t i = 0; i < buffer.size(); i += 4) {
        Eigen::Vector3d pt {buffer[i], buffer[i + 1], buffer[i + 2]};
        const Eigen::Vector3d rotationVector = pt.cross(Eigen::Vector3d(0., 0., 1.));
        pt = Eigen::AngleAxisd(VERTICAL_ANGLE_OFFSET, rotationVector.normalized()) * pt;
        pointCloud.emplace_back(pt[0], pt[1], pt[2], 1.0);
    }

    return pointCloud;
}
}  // namespace centerpoint
