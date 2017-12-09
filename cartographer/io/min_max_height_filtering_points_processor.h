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

#ifndef CARTOGRAPHER_IO_MIN_MAX_HEIGHT_FILTERING_POINTS_PROCESSOR_H_
#define CARTOGRAPHER_IO_MIN_MAX_HEIGHT_FILTERING_POINTS_PROCESSOR_H_

#include <memory>

#include "cartographer/common/lua_parameter_dictionary.h"
#include "cartographer/io/points_processor.h"

namespace cartographer {
namespace io {

// Filters all points that are higher from their 'origin' as 'max_height'
// or lower than 'min_height'.
class MinMaxHeightFiteringPointsProcessor : public PointsProcessor {
 public:
  constexpr static const char* kConfigurationFileActionName =
      "min_max_height_filter";
  MinMaxHeightFiteringPointsProcessor(double min_height, double max_height,
                                     PointsProcessor* next);
  static std::unique_ptr<MinMaxHeightFiteringPointsProcessor> FromDictionary(
      common::LuaParameterDictionary* dictionary, PointsProcessor* next);

  ~MinMaxHeightFiteringPointsProcessor() override {}

  MinMaxHeightFiteringPointsProcessor(
      const MinMaxHeightFiteringPointsProcessor&) = delete;
  MinMaxHeightFiteringPointsProcessor& operator=(
      const MinMaxHeightFiteringPointsProcessor&) = delete;

  void Process(std::unique_ptr<PointsBatch> batch) override;
  FlushResult Flush() override;

 private:
  const double min_height_;
  const double max_height_;
  PointsProcessor* const next_;
};

}  // namespace io
}  // namespace cartographer

#endif  // CARTOGRAPHER_IO_MIN_MAX_HEIGHT_FILTERING_POINTS_PROCESSOR_H_
