#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <yaml-cpp/yaml.h>
#include <eigen3/Eigen/Core>
#include "lidar_centerpoint/centerpoint_trt.hpp"
#include "lidar_centerpoint/utils.hpp"
#include "lidar_centerpoint/centerpoint_config.hpp"
#include "lidar_centerpoint/preprocess/pointcloud_densification.hpp"

#include <rerun.hpp>

#if (defined(_MSC_VER)or(defined(__GNUC__) and(7 <= __GNUC_MAJOR__)))
#include <filesystem>
namespace fs = ::std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = ::std::experimental::filesystem;
#endif

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

bool isPointInBox(const Eigen::Vector3f& point, 
                 const Eigen::Vector3f& box_center,
                 const Eigen::Vector3f& box_dims,
                 const Eigen::Quaternionf& box_rot) {
    // Transform point to box's local coordinates
    Eigen::Vector3f local_point = box_rot.inverse() * (point - box_center);
    
    // Check if point is within box dimensions
    return (std::abs(local_point.x()) <= box_dims.x()/2.0f &&
            std::abs(local_point.y()) <= box_dims.y()/2.0f &&
            std::abs(local_point.z()) <= box_dims.z()/2.0f);
}

void separatePoints(const std::vector<Eigen::Vector3f>& points,
                   const std::vector<centerpoint::Box3D>& boxes,
                   std::vector<Eigen::Vector3f>& inside_points,
                   std::vector<Eigen::Vector3f>& outside_points) {
    for (const auto& pt : points) {
        bool is_inside = false;
        for (const auto& box : boxes) {
            Eigen::Vector3f center(box.x, box.y, box.z);
            Eigen::Vector3f dims(box.length, box.width, box.height);
            
            // Create rotation quaternion from yaw
            Eigen::Quaternionf q;
            q = Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitX())
                    * Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitY())
                    * Eigen::AngleAxisf(-box.yaw - M_PI_2, Eigen::Vector3f::UnitZ());
            
            if (isPointInBox(pt, center, dims, q)) {
                is_inside = true;
                break;
            }
        }
        
        if (is_inside) {
            inside_points.push_back(pt);
        } else {
            outside_points.push_back(pt);
        }
    }
}

rerun::Color viridisColormap(float t) {
    // Viridis colormap values (x,r,g,b) where x is normalized position [0,1]
    static const std::vector<std::array<float, 4>> viridis = {
        {0.0f, 0.267004f, 0.004874f, 0.329415f},
        {0.2f, 0.253935f, 0.265254f, 0.529983f},
        {0.4f, 0.163625f, 0.471133f, 0.558148f},
        {0.6f, 0.134692f, 0.658636f, 0.517649f},
        {0.8f, 0.477504f, 0.821444f, 0.318195f},
        {1.0f, 0.993248f, 0.906157f, 0.143936f}
    };
    
    // Find indices for interpolation
    size_t i = 0;
    while (i < viridis.size()-1 && viridis[i+1][0] < t) {
        i++;
    }
    
    // Linear interpolation between two closest colors
    float x0 = viridis[i][0];
    float x1 = viridis[i+1][0];
    float alpha = (t - x0) / (x1 - x0);
    
    return rerun::Color(
        static_cast<uint8_t>((viridis[i][1] + alpha * (viridis[i+1][1] - viridis[i][1])) * 255),
        static_cast<uint8_t>((viridis[i][2] + alpha * (viridis[i+1][2] - viridis[i][2])) * 255),
        static_cast<uint8_t>((viridis[i][3] + alpha * (viridis[i+1][3] - viridis[i][3])) * 255),
        255
    );
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " /path/to/config/folder/ " << std::endl;
        return 1;
    }
    const auto rec = rerun::RecordingStream("MapBlend");
    rec.spawn().exit_on_failure();

    try {
        // Config paths setup
        const fs::path config_dir(argv[1]);
        const fs::path model_cfg_file("centerpoint_ml_package.param.yaml");
        const fs::path network_cfg_file("centerpoint.param.yaml");
        const fs::path dense_cfg_file("centerpoint.param.yaml");
        const fs::path pointcloud_cfg_file("pointcloud.yaml");

        // Load configurations
        auto model_config = YAML::LoadFile((config_dir / model_cfg_file).string());
        auto network_config = YAML::LoadFile((config_dir / network_cfg_file).string());
        auto dense_config = YAML::LoadFile((config_dir / dense_cfg_file).string());
        auto pointcloud_config = YAML::LoadFile((config_dir / pointcloud_cfg_file).string());

        // Get pointcloud directory
        fs::path pointcloud_dir(pointcloud_config["directory"].as<std::string>());
        
        // Get sorted list of .bin files
        std::vector<fs::path> bin_files;
        for(const auto& entry : fs::directory_iterator(pointcloud_dir)) {
            if(entry.path().extension() == ".bin") {
                bin_files.push_back(entry.path());
            }
        }
        std::sort(bin_files.begin(), bin_files.end());

        // Initialize detector
        auto model_cfg = loadModelConfig(model_config);
        auto network_cfg = loadNetworkConfig(network_config);
        auto dense_cfg = loadDensificationConfig(dense_config);
        
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

        auto detector = std::make_unique<centerpoint::CenterPointTRT>(
            encoder_param, head_param, dense_param, centerpoint_cfg);

        // Process each file
        for(const auto& bin_file : bin_files) {
            auto points = loadBIN_kitti(bin_file.string());
            std::vector<centerpoint::Box3D> objects;
            Eigen::Isometry3f tf = Eigen::Isometry3f::Identity();
            double timestamp = 0.0;

            // Visualize points
            // std::vector<rerun::Position3D> points3d;
            // for (const auto &pt : points) {
            //     points3d.push_back({pt[0], pt[1], pt[2]});
            // }
            // rec.log("source", rerun::Points3D(points3d));

            // Detect objects
            if (!detector->detect(points, tf, timestamp, objects)) {
                std::cerr << "Detection failed for " << bin_file << std::endl;
                continue;
            }

            std::vector<Eigen::Vector3f> eigen_points;
            for (const auto& pt : points) {
                eigen_points.push_back(Eigen::Vector3f(pt[0], pt[1], pt[2]));
            }
            std::vector<Eigen::Vector3f> inside_points, outside_points;
            separatePoints(eigen_points, objects, inside_points, outside_points);

            // Visualize points with different colors
            std::vector<rerun::Position3D> inside_points3d, outside_points3d;
            for (const auto& pt : inside_points) {
                inside_points3d.push_back({pt.x(), pt.y(), pt.z()});
            }

            constexpr float min_z = -3.0;
            constexpr float max_z = 8.0;
            std::vector<rerun::Color> outside_colors;
            for (const auto& pt : outside_points) {
                outside_points3d.push_back({pt.x(), pt.y(), pt.z()});
                float t = std::clamp((pt.z() - min_z) / (max_z - min_z), 0.0f, 1.0f);
                outside_colors.push_back(viridisColormap(t));
            }

            // Visualize detections
            std::vector<rerun::Position3D> centers;
            std::vector<rerun::Vector3D> half_sizes;
            std::vector<rerun::Quaternion> quat;

            for (const auto& obj : objects) {
                centers.push_back({obj.x, obj.y, obj.z});
                half_sizes.push_back({obj.length/2.0f, obj.width/2.0f, obj.height/2.0f});
                
                Eigen::Quaternionf q;
                q = Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitX())
                    * Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitY())
                    * Eigen::AngleAxisf(-obj.yaw - M_PI_2, Eigen::Vector3f::UnitZ());
                quat.push_back(rerun::Quaternion::from_xyzw(q.x(), q.y(), q.z(), q.w()));
            }

            auto boxes = rerun::Boxes3D::from_centers_and_half_sizes(centers, half_sizes)
                .with_quaternions(quat);


            // Log points with different colors
            rec.log("points_inside", 
                rerun::Points3D(inside_points3d)
                    .with_colors({rerun::Color(255, 0, 0, 255)}));
            rec.log("points_outside",
                rerun::Points3D(outside_points3d)
                    .with_colors(outside_colors)); // Red
            rec.log("detections", boxes);

            // Add delay between frames
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        return 0;
    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading YAML file: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}