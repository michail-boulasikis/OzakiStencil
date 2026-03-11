#include <gtest/gtest.h>

#include <ozaki/utils/stencil_spec.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifdef OZAKI_HAVE_FLOAT128
#include <quadmath.h>
#endif

// ============================================================
// parse_stencil_arg tests
// ============================================================

TEST(ParseStencilArg, Parse1D) {
  auto spec = parse_stencil_arg("3:1,-2,1");
  EXPECT_EQ(spec.shape.size(), 1u);
  EXPECT_EQ(spec.shape[0], 3);
  EXPECT_EQ(spec.offsets.size(), 1u);
  EXPECT_EQ(spec.offsets[0], 1);
  ASSERT_EQ(spec.coeffs.size(), 3u);
  EXPECT_EQ(static_cast<double>(spec.coeffs[0]), 1.0);
  EXPECT_EQ(static_cast<double>(spec.coeffs[1]), -2.0);
  EXPECT_EQ(static_cast<double>(spec.coeffs[2]), 1.0);
}

TEST(ParseStencilArg, Parse2D) {
  auto spec = parse_stencil_arg("3x3:0,3,0,3,4,3,0,3,0");
  EXPECT_EQ(spec.shape.size(), 2u);
  EXPECT_EQ(spec.shape[0], 3);
  EXPECT_EQ(spec.shape[1], 3);
  EXPECT_EQ(spec.offsets.size(), 2u);
  EXPECT_EQ(spec.offsets[0], 1);
  EXPECT_EQ(spec.offsets[1], 1);
  ASSERT_EQ(spec.coeffs.size(), 9u);
}

TEST(ParseStencilArg, Parse3D) {
  std::string arg = "3x3x3:";
  for (int i = 0; i < 27; ++i) {
    if (i > 0) arg += ",";
    arg += "1";
  }
  auto spec = parse_stencil_arg(arg);
  EXPECT_EQ(spec.shape.size(), 3u);
  EXPECT_EQ(spec.shape[0], 3);
  EXPECT_EQ(spec.shape[1], 3);
  EXPECT_EQ(spec.shape[2], 3);
  ASSERT_EQ(spec.coeffs.size(), 27u);
}

TEST(ParseStencilArg, PrecisionInt8) {
  auto spec = parse_stencil_arg("3:1,-2,1");
  // All integer values that fit in int8, so precision should be Int8
  EXPECT_EQ(spec.specified_precision, OriginalPrecision::Int8);
}

TEST(ParseStencilArg, PrecisionDouble) {
  auto spec = parse_stencil_arg("3:0.5,-1,0.5");
  // 0.5 doesn't roundtrip through int8_t, so precision should be Double
  EXPECT_EQ(spec.specified_precision, OriginalPrecision::Double);
}

TEST(ParseStencilArg, MissingColon_Throws) {
  EXPECT_THROW(parse_stencil_arg("3x3 1,2,3"), std::invalid_argument);
}

TEST(ParseStencilArg, WrongCoeffCount_Throws) {
  EXPECT_THROW(parse_stencil_arg("3x3:1,2,3"), std::invalid_argument);
}

TEST(ParseStencilArg, DefaultOffsets) {
  auto spec = parse_stencil_arg("5:1,2,3,4,5");
  ASSERT_EQ(spec.offsets.size(), 1u);
  EXPECT_EQ(spec.offsets[0], 2); // 5/2 = 2
}

// ============================================================
// read_stencil_file tests
// ============================================================

class StencilFileTest : public ::testing::Test {
protected:
  std::vector<std::string> tmp_files;

  std::string write_tmp_file(const std::string &content, const std::string &suffix = ".stc") {
    auto path = std::filesystem::temp_directory_path() / ("test_stencil_" + std::to_string(tmp_files.size()) + suffix);
    std::ofstream f(path);
    f << content;
    f.close();
    tmp_files.push_back(path.string());
    return path.string();
  }

  void TearDown() override {
    for (const auto &p : tmp_files)
      std::filesystem::remove(p);
  }
};

TEST_F(StencilFileTest, FullSpec) {
  std::string content = R"(
name TestStencil
shape 3 3
offsets 1 1
precision double
coeffs
0.0 3.0 0.0
3.0 4.0 3.0
0.0 3.0 0.0
)";
  auto path = write_tmp_file(content);
  auto spec = read_stencil_file(path);

  EXPECT_EQ(spec.name, "TestStencil");
  ASSERT_EQ(spec.shape.size(), 2u);
  EXPECT_EQ(spec.shape[0], 3);
  EXPECT_EQ(spec.shape[1], 3);
  ASSERT_EQ(spec.offsets.size(), 2u);
  EXPECT_EQ(spec.offsets[0], 1);
  EXPECT_EQ(spec.offsets[1], 1);
  EXPECT_EQ(spec.specified_precision, OriginalPrecision::Double);
  ASSERT_EQ(spec.coeffs.size(), 9u);
  EXPECT_EQ(static_cast<double>(spec.coeffs[4]), 4.0);
}

TEST_F(StencilFileTest, DefaultName) {
  std::string content = R"(
shape 3
coeffs
1.0 -2.0 1.0
)";
  auto path = write_tmp_file(content);
  auto spec = read_stencil_file(path);
  EXPECT_EQ(spec.name, "unnamed");
}

TEST_F(StencilFileTest, DefaultOffsets) {
  std::string content = R"(
name TestDefault
shape 5
coeffs
1 2 3 4 5
)";
  auto path = write_tmp_file(content);
  auto spec = read_stencil_file(path);
  ASSERT_EQ(spec.offsets.size(), 1u);
  EXPECT_EQ(spec.offsets[0], 2); // 5/2 = 2
}

TEST_F(StencilFileTest, CommentsAndBlanks) {
  std::string content = R"(
# This is a comment
name CommentTest

# Another comment
shape 3

   # Indented comment

coeffs
1.0 -2.0 1.0
)";
  auto path = write_tmp_file(content);
  auto spec = read_stencil_file(path);
  EXPECT_EQ(spec.name, "CommentTest");
  ASSERT_EQ(spec.coeffs.size(), 3u);
  EXPECT_EQ(static_cast<double>(spec.coeffs[1]), -2.0);
}

TEST_F(StencilFileTest, AllPrecisions) {
  auto test_precision = [&](const std::string &prec_str, OriginalPrecision expected) {
    std::string content = "shape 3\nprecision " + prec_str + "\ncoeffs\n1 -2 1\n";
    auto path = write_tmp_file(content);
    auto spec = read_stencil_file(path);
    EXPECT_EQ(spec.specified_precision, expected) << "Failed for precision: " << prec_str;
  };

  test_precision("int8", OriginalPrecision::Int8);
  test_precision("float", OriginalPrecision::Float);
  test_precision("double", OriginalPrecision::Double);
  test_precision("longdouble", OriginalPrecision::LongDouble);
  test_precision("fp128", OriginalPrecision::Float128);
}

TEST_F(StencilFileTest, UnknownPrecision_Throws) {
  std::string content = "shape 3\nprecision halfie\ncoeffs\n1 -2 1\n";
  auto path = write_tmp_file(content);
  EXPECT_THROW(read_stencil_file(path), std::invalid_argument);
}

TEST_F(StencilFileTest, NonexistentFile_Throws) {
  EXPECT_THROW(read_stencil_file("/nonexistent/path/to/stencil.stc"), std::runtime_error);
}

TEST_F(StencilFileTest, CoeffMismatch_Throws) {
  std::string content = "shape 3 3\ncoeffs\n1 -2 1\n";
  auto path = write_tmp_file(content);
  EXPECT_THROW(read_stencil_file(path), std::invalid_argument);
}

TEST_F(StencilFileTest, Shape1D) {
  std::string content = "shape 5\ncoeffs\n1 2 3 4 5\n";
  auto path = write_tmp_file(content);
  auto spec = read_stencil_file(path);
  ASSERT_EQ(spec.shape.size(), 1u);
  EXPECT_EQ(spec.shape[0], 5);
  ASSERT_EQ(spec.coeffs.size(), 5u);
}
