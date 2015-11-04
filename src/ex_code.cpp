#include <emmintrin.h>
#include <smmintrin.h>
#include <iostream>
#include <cassert>

#include "MyMat.h"
#include "EasyBMP.h"
#include "Timer.h"
#include "Environment.h"

/// Types of function to process
enum {NAIVE, FLOAT_SSE, INT_SSE};

typedef unsigned char uchar;

/// Weight of red channel stored as 4 equal floating point values
static const __m128 CONST_RED = _mm_set1_ps(0.2125f);

/// Weight of green channel stored as 4 equal floating point values
static const __m128 CONST_GREEN = _mm_set1_ps(0.7154f);

/// Weight of blue channel stored as 4 equal floating point values
static const __m128 CONST_BLUE = _mm_set1_ps(0.0721f);

/// Weight of red channel multiplied by 256. It's stored as 8 equal 16 bit integer values
static const __m128i CONST_RED_16_INT = _mm_set1_epi16(short(0.2125f * 256));
/// Weight of green channel multiplied by 256. It's stored as 8 equal 16 bit integer values
static const __m128i CONST_GREEN_16_INT = _mm_set1_epi16(short(0.7154f * 256));
/// Weight of blue channel multiplied by 256. It's stored as 8 equal 16 bit integer values
static const __m128i CONST_BLUE_16_INT = _mm_set1_epi16(short(0.0721f * 256));



/**
@function Get4Pixels16Bit
reads four consecutive pixels of the specified row started from given column and writes they to the
two registers out_BG and out_RA. Uses 16 bit per channel
@param in_img is a input image
@param in_row_idx is an index of a row to read pixels
@param in_col_idx is an index of a column with a first pixel
@param out_BG is a pointer to a 128bit register to store blue and green channels for the pixels four
consecutive pixels in format BBBB GGGG. Order of pixels is [0, 1, 2, 3]
@param out_RA is a pointer to a 128bit register to store red and alpha channels for the pixels four
consecutive pixels in format RRRR AAAA. Order of pixels is [0, 1, 2, 3]
*/
inline void Get4Pixels16Bit(BMP &in_img, int in_row_idx, int in_col_idx,
                            __m128i *out_BG, __m128i *out_RA)
{
  // read four consecutive pixels
  RGBApixel pixel0 = in_img.GetPixel(in_col_idx, in_row_idx);
  RGBApixel pixel1 = in_img.GetPixel(in_col_idx + 1, in_row_idx);
  RGBApixel pixel2 = in_img.GetPixel(in_col_idx + 2, in_row_idx);
  RGBApixel pixel3 = in_img.GetPixel(in_col_idx + 3, in_row_idx);

  // write two pixel0 and pixel2 to the p02 and other to the p13
  __m128i p02 = _mm_setr_epi32(*(int*)&pixel0, *(int*)&pixel2, 0, 0);
  __m128i p13 = _mm_setr_epi32(*(int*)&pixel1, *(int*)&pixel3, 0, 0);

  /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  * convert BGRA BGRA BGRA BGRA
  * to BBBB GGGG RRRR AAAA
  * order of pixels corresponds to the digits in the name of variables
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
  // BGRA BGRA 0000 0000 + BGRA BGRA 0000 0000 -> BBGG RRAA BBGG RRAA
  __m128i p0123 = _mm_unpacklo_epi8(p02, p13);
  // extract BBGG RRAA 0000 0000 of pixel2 and pixel3
  __m128i p23xx = _mm_unpackhi_epi64(p0123, _mm_setzero_si128());
  // BBGG RRAA XXXX XXXX + BBGG RRAA 0000 0000 -> BBBB GGGG RRRR AAAA
  // X denotes unused characters
  __m128i p0123_8bit = _mm_unpacklo_epi16(p0123, p23xx);

  // extend to 16bit 
  *out_BG = _mm_unpacklo_epi8(p0123_8bit, _mm_setzero_si128());
  *out_RA = _mm_unpackhi_epi8(p0123_8bit, _mm_setzero_si128());
}

/**
@function toGrayScale
realizes naive approach to convert RGBA image to grayscale
@param in_input is an input image.
@param out_mat is an output image. Each pixel is represented by a single unsigned char value.
*/
void toGrayScale(BMP &in_input, MyMat<uchar> &out_mat)
{
  // pointer to the processed row of the result image
  uchar *row_ptr = out_mat.data;
  // pointer to the processed element of the result image
  uchar *elem_ptr;
  for (size_t row_idx = 0; row_idx < out_mat.rows; ++row_idx)
  {
    elem_ptr = row_ptr;
    for (size_t col_idx = 0; col_idx < out_mat.cols; ++col_idx)
    {
      RGBApixel pixel = in_input.GetPixel((int)col_idx, (int)row_idx);
      *elem_ptr = (uchar)(pixel.Red * 0.2125f + pixel.Green * 0.7154f + pixel.Blue * 0.0721f);
      ++elem_ptr;
    }
    // go to next row
    row_ptr += out_mat.step;
  }
}

/**
@function toGrayScaleSSE
utilizes SSE to realize precise approach to convert RGBA image to grayscale.
@param in_input is an input image.
@param out_mat is an output image. Each pixel is represented by a single unsigned char value.
*/
void toGrayScaleSSE(BMP &in_input, MyMat<uchar> &out_mat)
{
  // pointer to the processed row of the result image
  uchar *row_ptr = out_mat.data;
  // pointer to the processed element of the result image
  uchar *elem_ptr;
  // number of elements to process at a time
  const int block_size = 4;
  // number of elements that will not be processed block-wise
  const int left_cols = out_mat.cols % block_size;
  // number of elements that will be processed block-wise
  const int block_cols = (int)out_mat.cols - left_cols;

  for (int row_idx = 0; row_idx < int(out_mat.rows); ++row_idx)
  {
    elem_ptr = row_ptr;
    // process block_size elements at a time.
    for (int col_idx = 0; col_idx < int(block_cols); col_idx += block_size)
    {
      // read four pixels
      __m128i BG;
      __m128i RA;
      Get4Pixels16Bit(in_input, row_idx, col_idx, &BG, &RA);

      // extend to 32bit 
      __m128i pB = _mm_unpacklo_epi8(BG, _mm_setzero_si128());
      __m128i pG = _mm_unpackhi_epi8(BG, _mm_setzero_si128());
      __m128i pR = _mm_unpacklo_epi8(RA, _mm_setzero_si128());

      // convert to float
      __m128 blue = _mm_cvtepi32_ps(pB);
      __m128 green = _mm_cvtepi32_ps(pG);
      __m128 red = _mm_cvtepi32_ps(pR);
      // multiply channels by weightsm

      red = _mm_mul_ps(red, CONST_RED);
      green = _mm_mul_ps(green, CONST_GREEN);
      blue = _mm_mul_ps(blue, CONST_BLUE);

      // sum up channels
      red = _mm_add_ps(red, green);
      red = _mm_add_ps(red, blue);

      // convert to 32bit integer
      __m128i color = _mm_cvttps_epi32(red);
      // convert to 16bit
      color = _mm_packus_epi32(color, _mm_setzero_si128());
      // convert to 8bit
      color = _mm_packus_epi16(color, _mm_setzero_si128());

      // write results to the out_mat
      *((long*)elem_ptr) = _mm_cvtsi128_si32(color);
      // go to next block in the row
      elem_ptr += block_size;
    }
    // process left elements in the row
    for (int col_idx = block_cols; col_idx < (int)out_mat.cols; ++col_idx)
    {
      RGBApixel pixel = in_input.GetPixel(col_idx, row_idx);
      *elem_ptr = (uchar)(pixel.Red * 0.2125f + pixel.Green * 0.7154f + pixel.Blue * 0.0721f);
      ++elem_ptr;      
    }
    // go to next row
    row_ptr += out_mat.step;
  }
}

/**
@function toGrayScale
utilizes SSE to realize fast approach to convert RGBA image to grayscale.
It is faster than toGrayScaleSSE, but not so precise
@param in_input is an input image.
@param out_mat is an output image. Each pixel is represented by a single unsigned char value.
*/
void toGrayScaleSSE_16BIT(BMP &in_input, MyMat<uchar> &out_mat)
{
  // pointer to the processed row of the result image
  uchar *row_ptr = out_mat.data;
  // pointer to the processed element of the result image
  uchar *elem_ptr;
  // number of elements to process at a time
  const int block_size = 8;
  // number of elements that will not be processed block-wise
  const int left_cols = (int)out_mat.cols % block_size;
  // number of elements that will be processed block-wise
  const int block_cols = (int)out_mat.cols - left_cols;

  for (int row_idx = 0; row_idx < int(out_mat.rows); ++row_idx)
  {
    elem_ptr = row_ptr;
    // process block_size elements at a time
    for (int col_idx = 0; col_idx < block_cols; col_idx += block_size)
    {
      // read four pixels
      __m128i BG1;
      __m128i RA1;
      Get4Pixels16Bit(in_input, row_idx, col_idx, &BG1, &RA1);

      // read another four pixels
      __m128i BG2;
      __m128i RA2;
      Get4Pixels16Bit(in_input, row_idx, col_idx + 4, &BG2, &RA2);

      // extract channels
      __m128i blue = _mm_unpacklo_epi64(BG1, BG2);
      __m128i green = _mm_unpackhi_epi64(BG1, BG2);
      __m128i red = _mm_unpacklo_epi64(RA1, RA2);

      // multiply channels by weights
      blue = _mm_mullo_epi16(blue, CONST_BLUE_16_INT);
      green = _mm_mullo_epi16(green, CONST_GREEN_16_INT);
      red = _mm_mullo_epi16(red, CONST_RED_16_INT);

      // sum up channels
      __m128i color = _mm_add_epi16(red, green);
      color = _mm_add_epi16(color, blue);

      // divide by 256
      color = _mm_srli_epi16(color, 8); 

      // convert to 8bit
      color = _mm_packus_epi16(color, _mm_setzero_si128());

      // write results to the output image
      _mm_storel_epi64((__m128i*)elem_ptr, color);
      elem_ptr += block_size;
    }
    // process left elements in the row
    for (int col_idx = block_cols; col_idx < int(out_mat.cols); ++col_idx)
    {
      RGBApixel pixel = in_input.GetPixel(col_idx, row_idx);
      *elem_ptr = (uchar)(pixel.Red * 0.2125f + pixel.Green * 0.7154f + pixel.Blue * 0.0721f);
      ++elem_ptr;      
    }
    // go to next row
    row_ptr += out_mat.step;
  }
}

/**
@function imgDif
computes difference between two matrices of the same size.
@param in_mat1 is a first matrix
@param in_mat2 is a second matrix
@return is a maximum difference between elements of the specified matrices
*/
uchar imgDif(const MyMat<uchar> &in_mat1, const MyMat<uchar> &in_mat2)
{
  assert((in_mat1.rows == in_mat2.rows) && (in_mat1.cols == in_mat2.cols) &&
    (in_mat1.channels == in_mat2.channels));

  uchar res = 0;
  uchar pixel_res;

  uchar *row_ptr1 = in_mat1.data;
  uchar *elem_ptr1;
  uchar *row_ptr2 = in_mat2.data;
  uchar *elem_ptr2;
  for (size_t row_idx = 0; row_idx < in_mat1.rows; ++row_idx)
  {
    elem_ptr1 = row_ptr1;
    elem_ptr2 = row_ptr2;
    for (size_t col_idx = 0; col_idx < in_mat1.cols; ++col_idx)
    {
      pixel_res = (*elem_ptr1 > *elem_ptr2) ? *elem_ptr1 - *elem_ptr2 : *elem_ptr2 - *elem_ptr1;
      res = res > pixel_res ? res : pixel_res;
      ++elem_ptr1;
      ++elem_ptr2;
    }
    row_ptr1 += in_mat1.step;
    row_ptr2 += in_mat2.step;
  }
  return res;
}

/**
@function parseParams
parses parameters of the command line. Following parameters are allowed:
<ul>
<li> --naive, -n -- converts image to grayscale without SSE
<li> --float, -f -- converts image to grayscale using floating point SSE operations
<li> --int, -i   -- converts image to grayscale using integer SSE operations
</ul>
@param argc is a number of parameters
@param argv ia an array of string parameters
@return a type of function that should be applyed
*/
int parseParams(int argc, char *argv[])
{
  if (argc != 2)
  {
    std::cerr << "Wrong number of parameters!" << std::endl <<
      "Usage" << std::endl <<
      "\t sse_test.exe parameter" << std::endl <<
      "Parameters" << std::endl <<
      "\t --naive, -n -- converts image to grayscale without SSE" << std::endl <<
      "\t --float, -f -- converts image to grayscale using floating point SSE operations" << std::endl <<
      "\t --int, -i   -- converts image to grayscale using integer SSE operations" << std::endl;
    return -1;
  }
  if (!(strcmp(argv[1], "--naive") && strcmp(argv[1], "-n")))
    return NAIVE;
  if (!(strcmp(argv[1], "--float") && strcmp(argv[1], "-f")))
    return FLOAT_SSE;
  if (!(strcmp(argv[1], "--int") && strcmp(argv[1], "-i")))
    return INT_SSE;

  std::cerr << "Unknown parameter!" << std::endl;
  return -1;
}

int main1(int argc, char *argv[])
{
  int type = parseParams(argc, argv);
  if (type == -1)
    return -1;

  BMP input_img;
  input_img.ReadFromFile("Lenna.bmp");
  MyMat<uchar> img(input_img.TellHeight(), input_img.TellWidth(), 1);
  MyMat<uchar> gt(input_img.TellHeight(), input_img.TellWidth(), 1);
  Timer t;
  const int NUM_ITER = 1000;

  switch (type)
  {
  case NAIVE:
    t.start();
    for (auto idx = 0; idx < NUM_ITER; ++idx)
      toGrayScale(input_img, img);
    t.check("Naive implementation");
    break;
  case FLOAT_SSE:
    t.start();
    for (auto idx = 0; idx < NUM_ITER; ++idx)
      toGrayScaleSSE(input_img, img);
    t.check("SSE (float)");
    break;
  case INT_SSE:
    t.start();
    for (auto idx = 0; idx < NUM_ITER; ++idx)
      toGrayScaleSSE_16BIT(input_img, img);
    t.check("SSE (int)");
    break;
  default : break;
  }
  t.stop();

  toGrayScale(input_img, gt);
  std::cout << "Error value = " << (int)imgDif(gt, img) << std::endl;

  return 0;
}