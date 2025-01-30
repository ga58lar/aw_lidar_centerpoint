#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <filesystem>
#include <yaml-cpp/yaml.h>
#include <eigen3/Eigen/Core>
#include "lidar_centerpoint/centerpoint_trt.hpp"
#include "lidar_centerpoint/utils.hpp"
#include "lidar_centerpoint/centerpoint_config.hpp"
#include "lidar_centerpoint/preprocess/pointcloud_densification.hpp"

using namespace centerpoint;
struct ModelConfig {
    std::vector<std::string> class_names;
    int point_feature_size;
    int max_voxel_size;
    std::vector<double> point_cloud_range;
    std::vector<double> voxel_size;
    int downsample_factor;
    int encoder_in_feature_size;
    bool has_variance;
    bool has_twist;
};

struct NetworkConfig {
    std::string encoder_onnx_path;
    std::string encoder_engine_path;
    std::string head_onnx_path;
    std::string head_engine_path;
    std::string trt_precision;
    double score_threshold;
    double circle_nms_dist_threshold;
    std::vector<double> yaw_norm_thresholds;
};

struct DensificationConfig {
    int num_past_frames;
};

ModelConfig loadModelConfig(const YAML::Node& config) {
    ModelConfig model_cfg;
    const auto& ros_params = config["/**"]["ros__parameters"];
    const auto& model_params = ros_params["model_params"];
    
    model_cfg.class_names = model_params["class_names"].as<std::vector<std::string>>();
    model_cfg.point_feature_size = model_params["point_feature_size"].as<int>();
    model_cfg.max_voxel_size = model_params["max_voxel_size"].as<int>();
    model_cfg.point_cloud_range = model_params["point_cloud_range"].as<std::vector<double>>();
    model_cfg.voxel_size = model_params["voxel_size"].as<std::vector<double>>();
    model_cfg.downsample_factor = model_params["downsample_factor"].as<int>();
    model_cfg.encoder_in_feature_size = model_params["encoder_in_feature_size"].as<int>();
    model_cfg.has_variance = model_params["has_variance"].as<bool>();
    model_cfg.has_twist = model_params["has_twist"].as<bool>();
    return model_cfg;
}

NetworkConfig loadNetworkConfig(const YAML::Node& config) {
    NetworkConfig net_cfg;
    const auto& ros_params = config["/**"]["ros__parameters"];
    
    net_cfg.encoder_onnx_path = ros_params["encoder_onnx_path"].as<std::string>();
    net_cfg.encoder_engine_path = ros_params["encoder_engine_path"].as<std::string>();
    net_cfg.head_onnx_path = ros_params["head_onnx_path"].as<std::string>();
    net_cfg.head_engine_path = ros_params["head_engine_path"].as<std::string>();
    net_cfg.trt_precision = ros_params["trt_precision"].as<std::string>();
    
    const auto& post_process = ros_params["post_process_params"];
    net_cfg.score_threshold = post_process["score_threshold"].as<double>();
    net_cfg.circle_nms_dist_threshold = post_process["circle_nms_dist_threshold"].as<double>();
    net_cfg.yaw_norm_thresholds = post_process["yaw_norm_thresholds"].as<std::vector<double>>();
    return net_cfg;
}

DensificationConfig loadDensificationConfig(const YAML::Node& config) {
    DensificationConfig dense_cfg;
    const auto& ros_params = config["/**"]["ros__parameters"];
    const auto& dense_params = ros_params["densification_params"];
    
    dense_cfg.num_past_frames = dense_params["num_past_frames"].as<int>();
    return dense_cfg;
}

CenterPointConfig createCenterPointConfig(
    const ModelConfig& model_cfg, const NetworkConfig& net_cfg)
{
    return CenterPointConfig(
        model_cfg.class_names.size(),
        model_cfg.point_feature_size,
        model_cfg.max_voxel_size,
        model_cfg.point_cloud_range,
        model_cfg.voxel_size,
        model_cfg.downsample_factor,
        model_cfg.encoder_in_feature_size,
        net_cfg.score_threshold,
        net_cfg.circle_nms_dist_threshold,
        net_cfg.yaw_norm_thresholds,
        model_cfg.has_variance
    );
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " /path/to/config/folder/ " << std::endl;
        return 1;
    }

    const fs::path config_dir(argv[1]);
    const fs::path model_cfg_file("centerpoint_ml_package.param.yaml");
    const fs::path network_cfg_file("centerpoint.param.yaml");
    const fs::path dense_cfg_file("centerpoint.param.yaml");
    const fs::path pointcloud_cfg_file("pointcloud.yaml");

    try {
        // Load YAML configurations
        auto model_config = YAML::LoadFile((config_dir / model_cfg_file).string());
        auto network_config = YAML::LoadFile((config_dir / network_cfg_file).string());
        auto dense_config = YAML::LoadFile((config_dir / dense_cfg_file).string());
        auto pointcloud_config = YAML::LoadFile((config_dir / pointcloud_cfg_file).string());

        auto model_cfg = loadModelConfig(model_config);
        auto network_cfg = loadNetworkConfig(network_config);
        auto dense_cfg = loadDensificationConfig(dense_config);
        auto pointcloud_path = pointcloud_config["path"].as<std::string>();

        // Create parameters
        NetworkParam encoder_param(
            network_cfg.encoder_onnx_path,
            network_cfg.encoder_engine_path,
            network_cfg.trt_precision);
        NetworkParam head_param(
            network_cfg.head_onnx_path,
            network_cfg.head_engine_path,
            network_cfg.trt_precision);
        DensificationParam dense_param(
            dense_cfg.num_past_frames);
        auto centerpoint_cfg = createCenterPointConfig(model_cfg, network_cfg);

        // Initialize detector
        auto detector = std::make_unique<centerpoint::CenterPointTRT>(
            encoder_param, head_param, dense_param, centerpoint_cfg);

        // Load pointcloud and detect
        auto points = loadBIN_kitti(pointcloud_path);
        std::vector<centerpoint::Box3D> objects;
        Eigen::Isometry3f tf = Eigen::Isometry3f::Identity();
        double timestamp = 0.0;
        if (!detector->detect(points, tf, timestamp, objects)) {
            std::cerr << "Detection failed" << std::endl;
            return 1;
        }

        // Output results
        for (const auto& obj : objects) {
            std::cout << "Object detected at: " 
                    << obj.x << ", "
                    << obj.y << ", "
                    << obj.z << std::endl;
        }

        return 0;
    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading YAML file: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}