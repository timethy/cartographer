/*
 * Copyright 2016 The Cartographer Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cartographer/mapping_2d/pose_graph/optimization_problem.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <sstream>
#include <vector>

#include "cartographer/common/ceres_solver_options.h"
#include "cartographer/common/histogram.h"
#include "cartographer/common/math.h"
#include "cartographer/mapping_2d/pose_graph/spa_cost_function.h"
#include "cartographer/sensor/odometry_data.h"
#include "cartographer/transform/transform.h"
#include "ceres/ceres.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping_2d {
namespace pose_graph {

namespace {

// Converts a pose into the 3 optimization variable format used for Ceres:
// translation in x and y, followed by the rotation angle representing the
// orientation.
std::array<double, 3> FromPose(const transform::Rigid2d& pose) {
  return {{pose.translation().x(), pose.translation().y(),
           pose.normalized_angle()}};
}

// Converts a pose as represented for Ceres back to an transform::Rigid2d pose.
transform::Rigid2d ToPose(const std::array<double, 3>& values) {
  return transform::Rigid2d({values[0], values[1]}, values[2]);
}

// (Almost) identical with Interpolate in mapping_3d, maybe move to common place.
template <typename MapByTimeType>
std::unique_ptr<transform::Rigid3d> Interpolate(
    const MapByTimeType& map_by_time, const int trajectory_id,
    const common::Time time) {
  const auto it = map_by_time.lower_bound(trajectory_id, time);
  if (it == map_by_time.EndOfTrajectory(trajectory_id)) {
    return nullptr;
  }
  if (it == map_by_time.BeginOfTrajectory(trajectory_id)) {
    if (it->time == time) {
      return common::make_unique<transform::Rigid3d>(it->pose);
    }
    return nullptr;
  }
  const auto prev_it = std::prev(it);
  return common::make_unique<transform::Rigid3d>(
      Interpolate(transform::TimestampedTransform{prev_it->time, prev_it->pose},
                  transform::TimestampedTransform{it->time, it->pose}, time)
          .transform);
}
// For landmark pose
template <typename MapByTimeType>
std::unique_ptr<transform::Rigid3d> Interpolate(
    const MapByTimeType& map_by_time, const int trajectory_id,
    const common::Time time, const common::Duration max_delta_time) {
  const auto it = map_by_time.lower_bound(trajectory_id, time);
  if (it == map_by_time.EndOfTrajectory(trajectory_id)) {
    return nullptr;
  }
  if (it == map_by_time.BeginOfTrajectory(trajectory_id)) {
    if (it->time == time) {
      return common::make_unique<transform::Rigid3d>(it->landmark.transform);
    }
    return nullptr;
  }
  const auto prev_it = std::prev(it);
  if (prev_it->time < time - max_delta_time) {
    return nullptr;
  }
  if (it->time > time + max_delta_time) {
    return nullptr;
  }

  return common::make_unique<transform::Rigid3d>(
      Interpolate(transform::TimestampedTransform{prev_it->time, prev_it->landmark.transform},
                  transform::TimestampedTransform{it->time, it->landmark.transform}, time)
          .transform);
}

}  // namespace

OptimizationProblem::OptimizationProblem(
    const mapping::pose_graph::proto::OptimizationProblemOptions& options)
    : options_(options) {}

OptimizationProblem::~OptimizationProblem() {}

void OptimizationProblem::AddImuData(const int trajectory_id,
                                     const sensor::ImuData& imu_data) {
  imu_data_.Append(trajectory_id, imu_data);
}

void OptimizationProblem::AddOdometryData(
    const int trajectory_id, const sensor::OdometryData& odometry_data) {
  odometry_data_.Append(trajectory_id, odometry_data);
}

void OptimizationProblem::AddLandmarkData(
    const int trajectory_id, const sensor::LandmarkData& landmark_data) {
  if(landmarks_in_trajectory_.find(trajectory_id) == landmarks_in_trajectory_.end()) {
    landmarks_in_trajectory_.emplace(trajectory_id, std::set<mapping::LandmarkId>{});
  }
  for (const auto& landmark: landmark_data.landmarks) {
    landmarks_in_trajectory_[trajectory_id].insert(landmark.id);
    if (landmark_data_by_id_.find(landmark.id) == landmark_data_by_id_.end()) {
      landmark_data_by_id_.emplace(landmark.id, sensor::MapByTime<sensor::TimedLandmark>());
    }
    landmark_data_by_id_[landmark.id].Append(trajectory_id, sensor::TimedLandmark{landmark_data.time, landmark});
  }
}

void OptimizationProblem::AddTrajectoryNode(
    const int trajectory_id, const common::Time time,
    const transform::Rigid2d& initial_pose, const transform::Rigid2d& pose,
    const Eigen::Quaterniond& gravity_alignment) {
  node_data_.Append(trajectory_id,
                    NodeData{time, initial_pose, pose, gravity_alignment});
}

void OptimizationProblem::InsertTrajectoryNode(
    const mapping::NodeId& node_id, const common::Time time,
    const transform::Rigid2d& initial_pose, const transform::Rigid2d& pose,
    const Eigen::Quaterniond& gravity_alignment) {
  node_data_.Insert(node_id,
                    NodeData{time, initial_pose, pose, gravity_alignment});
}

void OptimizationProblem::TrimTrajectoryNode(const mapping::NodeId& node_id) {
  imu_data_.Trim(node_data_, node_id);
  odometry_data_.Trim(node_data_, node_id);
  for(const auto& landmark_id: landmarks_in_trajectory_.at(node_id.trajectory_id)) {
    landmark_data_by_id_.at(landmark_id).Trim(node_data_, node_id);
  }
  node_data_.Trim(node_id);
}

void OptimizationProblem::AddSubmap(
    const int trajectory_id, const transform::Rigid2d& global_submap_pose) {
  submap_data_.Append(trajectory_id, SubmapData{global_submap_pose});
}

void OptimizationProblem::InsertSubmap(
    const mapping::SubmapId& submap_id,
    const transform::Rigid2d& global_submap_pose) {
  submap_data_.Insert(submap_id, SubmapData{global_submap_pose});
}

void OptimizationProblem::TrimSubmap(const mapping::SubmapId& submap_id) {
  submap_data_.Trim(submap_id);
}

void OptimizationProblem::SetMaxNumIterations(const int32 max_num_iterations) {
  options_.mutable_ceres_solver_options()->set_max_num_iterations(
      max_num_iterations);
}

void OptimizationProblem::Solve(const std::vector<Constraint>& constraints,
                                const std::set<int>& frozen_trajectories) {
  if (node_data_.empty()) {
    // Nothing to optimize.
    return;
  }

  ceres::Problem::Options problem_options;
  ceres::Problem problem(problem_options);

  // Set the starting point.
  // TODO(hrapp): Move ceres data into SubmapData.
  mapping::MapById<mapping::SubmapId, std::array<double, 3>> C_submaps;
  mapping::MapById<mapping::NodeId, std::array<double, 3>> C_nodes;
  std::map<mapping::LandmarkId, std::array<double, 3>> C_landmarks;
  bool first_submap = true;
  for (const auto& submap_id_data : submap_data_) {
    const bool frozen =
        frozen_trajectories.count(submap_id_data.id.trajectory_id) != 0;
    C_submaps.Insert(submap_id_data.id,
                     FromPose(submap_id_data.data.global_pose));
    problem.AddParameterBlock(C_submaps.at(submap_id_data.id).data(), 3);
    if (first_submap || frozen) {
      first_submap = false;
      // Fix the pose of the first submap or all submaps of a frozen
      // trajectory.
      problem.SetParameterBlockConstant(C_submaps.at(submap_id_data.id).data());
    }
  }
  for (const auto& node_id_data : node_data_) {
    const bool frozen =
        frozen_trajectories.count(node_id_data.id.trajectory_id) != 0;
    C_nodes.Insert(node_id_data.id, FromPose(node_id_data.data.pose));
    problem.AddParameterBlock(C_nodes.at(node_id_data.id).data(), 3);
    if (frozen) {
      problem.SetParameterBlockConstant(C_nodes.at(node_id_data.id).data());
    }
  }
  // Initialize landmarks
  for (const auto& landmark_id_data : landmark_poses_) {
    C_landmarks.emplace(landmark_id_data.first, FromPose(Project2D(landmark_id_data.second)));
    CHECK(C_landmarks.find(landmark_id_data.first) != C_landmarks.end());
    problem.AddParameterBlock(C_landmarks.at(landmark_id_data.first).data(), 3);
    // TODO (timethy): Add possibility to freeze landmarks
    bool frozen = false;
    if (frozen) {
      problem.SetParameterBlockConstant(C_landmarks.at(landmark_id_data.first).data());
    }
  }

  // Add cost functions for intra- and inter-submap constraints.
  for (const Constraint& constraint : constraints) {
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<SpaCostFunction, 3, 3, 3>(
            new SpaCostFunction(constraint.pose)),
        // Only loop closure constraints should have a loss function.
        constraint.tag == Constraint::INTER_SUBMAP
            ? new ceres::HuberLoss(options_.huber_scale())
            : nullptr,
        C_submaps.at(constraint.submap_id).data(),
        C_nodes.at(constraint.node_id).data());
  }

  for (auto node_it = node_data_.begin(); node_it != node_data_.end();) {
    const int trajectory_id = node_it->id.trajectory_id;
    const auto trajectory_end = node_data_.EndOfTrajectory(trajectory_id);

    if (frozen_trajectories.count(trajectory_id) != 0) {
      node_it = trajectory_end;
      continue;
    }

    auto prev_node_it = node_it;
    for (++node_it; node_it != trajectory_end; ++node_it) {
      const mapping::NodeId first_node_id = prev_node_it->id;
      const NodeData& first_node_data = prev_node_it->data;
      prev_node_it = node_it;
      const mapping::NodeId second_node_id = node_it->id;
      const NodeData& second_node_data = node_it->data;

      if (second_node_id.node_index != first_node_id.node_index + 1) {
        continue;
      }

      const transform::Rigid3d relative_pose =
          ComputeRelativePose(trajectory_id, first_node_data, second_node_data);
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<SpaCostFunction, 3, 3, 3>(
              new SpaCostFunction(Constraint::Pose{
                  relative_pose, options_.consecutive_node_translation_weight(),
                  options_.consecutive_node_rotation_weight()})),
          nullptr /* loss function */, C_nodes.at(first_node_id).data(),
          C_nodes.at(second_node_id).data());
    }
  }

  size_t n_landmark_constraints = 0;
  for (auto node_it = node_data_.begin(); node_it != node_data_.end();) {
    const int trajectory_id = node_it->id.trajectory_id;
    const auto trajectory_end = node_data_.EndOfTrajectory(trajectory_id);

    if (landmarks_in_trajectory_.find(trajectory_id) == landmarks_in_trajectory_.end()) {
      node_it = trajectory_end;
      continue;
    }

    for (; node_it != trajectory_end; ++node_it) {
      const mapping::NodeId node_id = node_it->id;
      const NodeData &node_data = node_it->data;

      for(const auto& landmark_id: landmarks_in_trajectory_.at(node_id.trajectory_id)) {
        const common::Duration kMaxTimeToNearestLandmarkObservation = common::FromMilliseconds(250);
        const std::unique_ptr<transform::Rigid3d> landmark_pose_ptr =
            Interpolate(landmark_data_by_id_.at(landmark_id), trajectory_id, node_data.time, kMaxTimeToNearestLandmarkObservation);
        if (landmark_pose_ptr == nullptr) {
          continue;
        }

        // Somehow there is a yaw part in here.
        const auto landmark_pose = transform::Rigid3d::Rotation(
            node_data.gravity_alignment) * (*landmark_pose_ptr);

        // Initialize landmark if this is the first observation of it
        if (C_landmarks.find(landmark_id) == C_landmarks.end()) {
          const auto initial_pose = Embed3D(node_data.pose) * landmark_pose;
          landmark_poses_.emplace(landmark_id, initial_pose);
          C_landmarks.emplace(landmark_id, FromPose(Project2D(initial_pose)));
          problem.AddParameterBlock(C_landmarks.at(landmark_id).data(), 3);
        }

        const mapping::PoseGraph::Constraint::Pose landmark_pose_constraint{
            landmark_pose, options_.landmark_translation_weight(),
            options_.landmark_rotation_weight()};

        VLOG(10) << "Add landmark pose constraint: " << landmark_pose << " from node: " << node_id << "\n"
                << "node at : " << node_data.pose << " with gravity alignment " << node_data.gravity_alignment.vec();

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<SpaCostFunction, 3, 3, 3>(
                new SpaCostFunction(landmark_pose_constraint)),
            nullptr,
            C_nodes.at(node_id).data(),
            C_landmarks.at(landmark_id).data());

        n_landmark_constraints++;
      }
    }
  }

  // Solve.
  ceres::Solver::Summary summary;
  ceres::Solve(
      common::CreateCeresSolverOptions(options_.ceres_solver_options()),
      &problem, &summary);
  if (options_.log_solver_summary()) {
    LOG(INFO) << summary.FullReport();
  }

  // Store the result.
  for (const auto& C_submap_id_data : C_submaps) {
    submap_data_.at(C_submap_id_data.id).global_pose =
        ToPose(C_submap_id_data.data);
  }
  for (const auto& C_node_id_data : C_nodes) {
    node_data_.at(C_node_id_data.id).pose = ToPose(C_node_id_data.data);
  }
  for (const auto& C_landmark_id_data : C_landmarks) {
    CHECK(landmark_poses_.find(C_landmark_id_data.first) != landmark_poses_.end());
    landmark_poses_.at(C_landmark_id_data.first) =
        Embed3D(ToPose(C_landmark_id_data.second));
  }
  LOG(WARNING) << "Added " << n_landmark_constraints << " landmark constraints.";
  std::ostringstream oss;
  for (const auto& C_landmark_id_data : C_landmarks)
  {
    oss << C_landmark_id_data.first << ", " <<
        C_landmark_id_data.second[0] << ", " <<
        C_landmark_id_data.second[1] << ", " <<
        C_landmark_id_data.second[2] << "\n";
  }
  LOG(WARNING) << oss.str();
}

const mapping::MapById<mapping::NodeId, NodeData>&
OptimizationProblem::node_data() const {
  return node_data_;
}

const mapping::MapById<mapping::SubmapId, SubmapData>&
OptimizationProblem::submap_data() const {
  return submap_data_;
}

const std::map<mapping::LandmarkId, transform::Rigid3d>&
OptimizationProblem::landmark_poses() const {
  return landmark_poses_;
}

const sensor::MapByTime<sensor::ImuData>& OptimizationProblem::imu_data()
    const {
  return imu_data_;
}

const sensor::MapByTime<sensor::OdometryData>&
OptimizationProblem::odometry_data() const {
  return odometry_data_;
}

const std::map<mapping::LandmarkId, sensor::MapByTime<sensor::TimedLandmark>>&
OptimizationProblem::landmark_data_by_id() const {
  return landmark_data_by_id_;
}

std::unique_ptr<transform::Rigid3d> OptimizationProblem::InterpolateOdometry(
    const int trajectory_id, const common::Time time) const {
  return Interpolate(odometry_data_, trajectory_id, time);
}

transform::Rigid3d OptimizationProblem::ComputeRelativePose(
    const int trajectory_id, const NodeData& first_node_data,
    const NodeData& second_node_data) const {
  if (odometry_data_.HasTrajectory(trajectory_id)) {
    const std::unique_ptr<transform::Rigid3d> first_node_odometry =
        InterpolateOdometry(trajectory_id, first_node_data.time);
    const std::unique_ptr<transform::Rigid3d> second_node_odometry =
        InterpolateOdometry(trajectory_id, second_node_data.time);
    if (first_node_odometry != nullptr && second_node_odometry != nullptr) {
      return transform::Rigid3d::Rotation(first_node_data.gravity_alignment) *
             first_node_odometry->inverse() * (*second_node_odometry) *
             transform::Rigid3d::Rotation(
                 second_node_data.gravity_alignment.inverse());
    }
  }
  return transform::Embed3D(first_node_data.initial_pose.inverse() *
                            second_node_data.initial_pose);
}

}  // namespace pose_graph
}  // namespace mapping_2d
}  // namespace cartographer
