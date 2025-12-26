#include <ozaki/utils/utils.hpp>
#include <ozaki/core/types.hpp>

#include <bitset>
#include <string>
#include <iostream>

void print_bin_f64(const float64_t a) {
  auto bitset = std::bitset<8 * sizeof(float64_t)>(*reinterpret_cast<const uint64_t *>(&a));
  std::string sign = bitset.to_string().substr(0, 1);
  std::string exponent = bitset.to_string().substr(2, 11);
  std::string mantissa = bitset.to_string().substr(13, 52);
  std::cout << sign << "'" << exponent << "'(1)" << mantissa << "\n";
}
