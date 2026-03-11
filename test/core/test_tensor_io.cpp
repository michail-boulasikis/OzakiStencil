#include <gtest/gtest.h>

#include <ozaki/utils/tensor.hpp>

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#ifdef __GNUC__
#include <quadmath.h>
#endif

// ============================================================
// Roundtrip test fixture
// ============================================================

class TensorIOTest : public ::testing::Test {
protected:
  std::vector<std::string> tmp_files;

  std::string tmp_path(const std::string &name) {
    auto path = std::filesystem::temp_directory_path() / ("test_tensor_" + name + ".dat");
    tmp_files.push_back(path.string());
    return path.string();
  }

  void TearDown() override {
    for (const auto &p : tmp_files)
      std::filesystem::remove(p);
  }
};

// ============================================================
// Roundtrip tests (with shape accessor checks consolidated)
// ============================================================

TEST_F(TensorIOTest, Roundtrip_Float_1D) {
  Tensor<float> t;
  t.shape = {10};
  t.data.resize(10);
  for (size_t i = 0; i < 10; ++i)
    t.data[i] = static_cast<float>(i) * 1.5f;

  auto path = tmp_path("float_1d");
  write_tensor(path, t);
  auto t2 = read_tensor<float>(path);

  ASSERT_EQ(t2.shape, t.shape);
  ASSERT_EQ(t2.data.size(), t.data.size());
  for (size_t i = 0; i < t.data.size(); ++i)
    EXPECT_EQ(t2.data[i], t.data[i]) << "Mismatch at index " << i;

  // Shape accessor checks
  EXPECT_EQ(t2.shape.size(), 1u);
  EXPECT_EQ(t2.width(), 10u);
  EXPECT_EQ(t2.height(), 1u);
}

TEST_F(TensorIOTest, Roundtrip_Double_2D) {
  Tensor<double> t;
  t.shape = {4, 5};
  t.data.resize(20);
  for (size_t i = 0; i < 20; ++i)
    t.data[i] = static_cast<double>(i) * 3.14;

  auto path = tmp_path("double_2d");
  write_tensor(path, t);
  auto t2 = read_tensor<double>(path);

  ASSERT_EQ(t2.shape, t.shape);
  ASSERT_EQ(t2.data.size(), t.data.size());
  for (size_t i = 0; i < t.data.size(); ++i)
    EXPECT_EQ(t2.data[i], t.data[i]) << "Mismatch at index " << i;

  // Shape accessor checks
  EXPECT_EQ(t2.shape.size(), 2u);
  EXPECT_EQ(t2.width(), 4u);
  EXPECT_EQ(t2.height(), 5u);
}

TEST_F(TensorIOTest, Roundtrip_LongDouble_3D) {
  Tensor<long double> t;
  t.shape = {2, 3, 4};
  t.data.resize(24);
  for (size_t i = 0; i < 24; ++i)
    t.data[i] = static_cast<long double>(i) * 2.718L;

  auto path = tmp_path("longdouble_3d");
  write_tensor(path, t);
  auto t2 = read_tensor<long double>(path);

  ASSERT_EQ(t2.shape, t.shape);
  ASSERT_EQ(t2.data.size(), t.data.size());
  for (size_t i = 0; i < t.data.size(); ++i)
    EXPECT_EQ(t2.data[i], t.data[i]) << "Mismatch at index " << i;

  // Shape accessor checks
  EXPECT_EQ(t2.shape.size(), 3u);
  EXPECT_EQ(t2.width(), 2u);
  EXPECT_EQ(t2.height(), 3u);
}

TEST_F(TensorIOTest, Roundtrip_Float128_1D) {
  Tensor<__float128> t;
  t.shape = {5};
  t.data.resize(5);
  for (size_t i = 0; i < 5; ++i)
    t.data[i] = static_cast<__float128>(i) * static_cast<__float128>(1.23456789012345L);

  auto path = tmp_path("float128_1d");
  write_tensor(path, t);
  auto t2 = read_tensor<__float128>(path);

  ASSERT_EQ(t2.shape, t.shape);
  ASSERT_EQ(t2.data.size(), t.data.size());
  // __float128 doesn't support == well, use memcmp
  EXPECT_EQ(std::memcmp(t2.data.data(), t.data.data(), 5 * sizeof(__float128)), 0);
}

TEST_F(TensorIOTest, Roundtrip_SingleElement) {
  Tensor<double> t;
  t.shape = {1};
  t.data = {42.0};

  auto path = tmp_path("single");
  write_tensor(path, t);
  auto t2 = read_tensor<double>(path);

  ASSERT_EQ(t2.shape, t.shape);
  ASSERT_EQ(t2.data.size(), 1u);
  EXPECT_EQ(t2.data[0], 42.0);
}

TEST_F(TensorIOTest, Roundtrip_Large) {
  Tensor<double> t;
  t.shape = {100, 100};
  t.data.resize(10000);
  for (size_t i = 0; i < 10000; ++i)
    t.data[i] = static_cast<double>(i);

  auto path = tmp_path("large");
  write_tensor(path, t);
  auto t2 = read_tensor<double>(path);

  ASSERT_EQ(t2.shape, t.shape);
  ASSERT_EQ(t2.data.size(), t.data.size());
  for (size_t i = 0; i < t.data.size(); ++i)
    EXPECT_EQ(t2.data[i], t.data[i]);
}

// ============================================================
// Error condition tests
// ============================================================

TEST_F(TensorIOTest, ReadNonexistent_Throws) {
  EXPECT_THROW(read_tensor<double>("/nonexistent/path/tensor.dat"), std::runtime_error);
}

TEST_F(TensorIOTest, TypeMismatch_Throws) {
  Tensor<double> t;
  t.shape = {3};
  t.data = {1.0, 2.0, 3.0};

  auto path = tmp_path("type_mismatch");
  write_tensor(path, t);

  // Read as float should throw due to type_id mismatch (double=1, float=0)
  EXPECT_THROW(read_tensor<float>(path), std::runtime_error);
}
