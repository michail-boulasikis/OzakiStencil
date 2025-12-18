#include <ozaki/utils/utils.hpp>
#include <ozaki/core/types.hpp>

#include <bitset>
#include <string>
#include <iostream>
#include <algorithm>

void print_bin_f64(const float64_t a) {
  auto bitset = std::bitset<8 * sizeof(float64_t)>(*reinterpret_cast<const uint64_t *>(&a));
  std::string sign = bitset.to_string().substr(0, 1);
  std::string exponent = bitset.to_string().substr(2, 11);
  std::string mantissa = bitset.to_string().substr(13, 52);
  std::cout << sign << "'" << exponent << "'(1)" << mantissa << "\n";
}

std::vector<bool> get_flip_thresholds(const std::vector<int32_t> &exps) {

  constexpr int32_t FLIP_THRESHOLD = 62;
  std::vector<bool> res(exps.size());

  // We know that in the end we have to flip and finish off the reconstruction
  res.push_back(true);

  int32_t cnt = 32; // We start with 32 bits already occupied by the LSB slice
  for (size_t i = 1; i < exps.size(); ++i) {
    cnt += exps[i - 1] + exps[i];
    
    // If we are over the flip threshold that means that the next accumulation might overflow
    if (cnt > FLIP_THRESHOLD) {
      cnt = 32;
      res.push_back(true);
    } else {
      res.push_back(false);
    }
    
  }
  std::reverse(res.begin(), res.end());
  return res;
}
