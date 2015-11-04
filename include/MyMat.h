//This code from example code

/**
@class MyMat
Simple matrix realization. Allows implicit allocating and deallocating memory for the matrix.
Gives C-style interface for memory access. Matrix is stored row-by-row.
Each element can contain few numbers.
For example, MyMat<uchar> img(N, M, 4) can store RGBA image with N rows and M columns.
<ul>
<li>To access the <i>i<sup>th</sup></i> row of the matrix use 
@code (T*)((char*)data + i * step) @endcode
<li>To access the <i>j<sup>th</sup></i> element in the <i>i<sup>th</sup></i> row use
@code (T*)((char*)data + i * step) + j * channels @endcode
<li>To access the <i>k<sup>th</sup></i> channel of the <i>j<sup>th</sup></i> element
in the <i>i<sup>th</sup></i> row use
@code (T*)((char*)data + i * step) + j * channels @endcode
</ul>
*/
template<class T>
class MyMat
{
public:
  /// stores pointer to the data
  T *data;

  /// stores number of rows in the matrix
  size_t rows;

  /// stores number of columns in the matrix
  size_t cols;

  /// stores number of bytes between two consecutive rows of the matrix. step >= cols * channels * sizeof(T)
  size_t step;

  /// stores number of channels per element
  size_t channels;

  /// default constructor. Constructs empty matrix
  MyMat()
    : rows(0), cols(0), channels(0), step(0), data(nullptr)
  {
  }

  /// constructor.
  /// Constructs matrix with specified number of rows, columns and channels per element
  /// @param in_rows is a number of rows in the matrix
  /// @param in_cols is a number of columns in the matrix
  /// @param in_channels is a number of channels per element
  MyMat(size_t in_rows, size_t in_cols, size_t in_channels)
  {
    data = nullptr;
    Init(in_rows, in_cols, in_channels);
  }

  /// destructor.
  /// Deallocates used memory
  ~MyMat()
  {
    clear();
  }

  /// Constructs matrix with specified number of rows, columns and channels per element
  /// @param in_rows is a number of rows in the matrix
  /// @param in_cols is a number of columns in the matrix
  /// @param in_channels is a number of channels per element
  void Init(size_t in_rows, size_t in_cols, size_t in_channels)
  {
    clear();
    rows = in_rows;
    cols = in_cols;
    channels = in_channels;
    size_t elems_in_row = cols * channels;
    size_t elems_in_mat = elems_in_row * in_rows;
    data = new T[elems_in_mat];
    step = elems_in_row * sizeof(T);
  }

private:
  /// Deallocates used memory
  void clear()
  {
    if (data != nullptr)
    {
      delete [] data;
      data = nullptr;
    }
  }
  
  /// Copy constructor. Copying is not allowed.
  MyMat(const MyMat &in_obj)
  {
  }
};
