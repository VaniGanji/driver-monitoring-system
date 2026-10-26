#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace dms
{

std::string estimateHeadDirection(
    const std::vector<cv::Point>& landmarks);

} // namespace dms