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

#include "cartographer/io/min_max_height_filtering_points_processor.h"

#include "cartographer/common/lua_parameter_dictionary.h"
#include "cartographer/common/make_unique.h"
#include "cartographer/io/points_batch.h"

namespace cartographer {
namespace io {

std::unique_ptr<MinMaxHeightFiteringPointsProcessor>
MinMaxHeightFiteringPointsProcessor::FromDictionary(
    common::LuaParameterDictionary* const dictionary,
    PointsProcessor* const next) {
  return common::make_unique<MinMaxHeightFiteringPointsProcessor>(
      dictionary->GetDouble("min_z"), dictionary->GetDouble("max_z"),
      next);
}

MinMaxHeightFiteringPointsProcessor::MinMaxHeightFiteringPointsProcessor(
    const double min_height, const double max_height, PointsProcessor* next)
    : min_height_(min_height), max_height_(max_height), next_(next) {}

void MinMaxHeightFiteringPointsProcessor::Process(
    std::unique_ptr<PointsBatch> batch) {
  std::unordered_set<int> to_remove;
  for (size_t i = 0; i < batch->points.size(); ++i) {
    const float height = (batch->points[i] - batch->origin)(2);
    if (!(min_height_ <= height && height <= max_height_)) {
      to_remove.insert(i);
    }
  }
  RemovePoints(to_remove, batch.get());
  next_->Process(std::move(batch));
}

PointsProcessor::FlushResult MinMaxHeightFiteringPointsProcessor::Flush() {
  return next_->Flush();
}

}  // namespace io
}  // namespace cartographer
