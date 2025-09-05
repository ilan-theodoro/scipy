/* Copyright (C) 2003-2005 Peter J. Verveer
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.
 *
 * 3. The name of the author may not be used to endorse or promote
 *    products derived from this software without specific prior
 *    written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "ni_support.h"
#include "ni_interpolation.h"
#include "ni_splines.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* SIMD intrinsics for vectorization */
/* Runtime CPU feature detection */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

/* Include SIMD headers if available */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <emmintrin.h>  /* SSE2 */
#ifdef __SSE3__
#include <pmmintrin.h>
#endif
#ifdef __SSE4_1__
#include <smmintrin.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>  /* AVX2 */
#endif
#endif

/* Define build-time capabilities */
#if defined(__AVX2__)
#define SCIPY_BUILD_AVX2 1
#define SCIPY_HAVE_AVX2 1
#endif
#if defined(__SSE4_1__)
#define SCIPY_BUILD_SSE41 1
#endif
#if defined(__SSE2__) || defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define SCIPY_BUILD_SSE2 1
#define SCIPY_HAVE_SSE2 1
#endif

/* SIMD vector sizes */
#define SCIPY_SIMD_DOUBLES_AVX2 4  /* AVX2 processes 4 doubles */
#define SCIPY_SIMD_FLOATS_AVX2 8   /* AVX2 processes 8 floats */
#define SCIPY_SIMD_DOUBLES_SSE2 2  /* SSE2 processes 2 doubles */
#define SCIPY_SIMD_FLOATS_SSE2 4   /* SSE2 processes 4 floats */

/* Alignment macros */
#if defined(__GNUC__) || defined(__clang__)
  #define SCIPY_ALIGN(x) __attribute__((aligned(x)))
#elif defined(_MSC_VER)
  #define SCIPY_ALIGN(x) __declspec(align(x))
#else
  #define SCIPY_ALIGN(x)
#endif

/* Runtime CPU feature detection */
static int cpu_has_avx2 = -1;    /* -1 = not checked */
static int cpu_has_sse41 = -1;   /* -1 = not checked */
static int cpu_has_sse2 = -1;    /* -1 = not checked */
static int scipy_simd_disabled = -1;  /* -1 = not checked, 0 = enabled, 1 = disabled */

static void detect_cpu_features() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#ifdef _MSC_VER
    int cpuinfo[4];
    __cpuid(cpuinfo, 1);
    cpu_has_sse2 = (cpuinfo[3] & (1 << 26)) != 0;
    cpu_has_sse41 = (cpuinfo[2] & (1 << 19)) != 0;
    
    __cpuid(cpuinfo, 7);
    cpu_has_avx2 = (cpuinfo[1] & (1 << 5)) != 0;
#else
    unsigned int eax, ebx, ecx, edx;
    
    /* Check for SSE2 and SSE4.1 */
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        cpu_has_sse2 = (edx & (1 << 26)) != 0;
        cpu_has_sse41 = (ecx & (1 << 19)) != 0;
    } else {
        cpu_has_sse2 = 0;
        cpu_has_sse41 = 0;
    }
    
    /* Check for AVX2 */
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        cpu_has_avx2 = (ebx & (1 << 5)) != 0;
    } else {
        cpu_has_avx2 = 0;
    }
#endif
#else
    /* Non-x86 architectures */
    cpu_has_sse2 = 0;
    cpu_has_sse41 = 0;
    cpu_has_avx2 = 0;
#endif
}

static inline int has_avx2() {
#ifdef SCIPY_BUILD_AVX2
    if (cpu_has_avx2 == -1) detect_cpu_features();
    return cpu_has_avx2;
#else
    return 0;
#endif
}

static inline int has_sse41() {
#ifdef SCIPY_BUILD_SSE41
    if (cpu_has_sse41 == -1) detect_cpu_features();
    return cpu_has_sse41;
#else
    return 0;
#endif
}

static inline int has_sse2() {
#ifdef SCIPY_BUILD_SSE2
    if (cpu_has_sse2 == -1) detect_cpu_features();
    return cpu_has_sse2;
#else
    return 0;
#endif
}

static inline int should_use_simd() {
    if (scipy_simd_disabled == -1) {
        /* Check environment variable once */
        const char* env_var = getenv("SCIPY_FORCE_SCALAR_INTERPOLATION");
        if (env_var && (strcmp(env_var, "1") == 0 || 
                       strcmp(env_var, "true") == 0 || 
                       strcmp(env_var, "True") == 0 || 
                       strcmp(env_var, "TRUE") == 0)) {
            scipy_simd_disabled = 1;
        } else {
            scipy_simd_disabled = 0;
        }
    }
    return scipy_simd_disabled == 0;
}

/* map a coordinate outside the borders, according to the requested
     boundary condition: */
static double map_coordinate(double in, npy_intp len, int mode) {
  if (in < 0) {
    switch (mode) {
    case NI_EXTEND_MIRROR:
      if (len <= 1) {
        in = 0;
      } else {
        npy_intp sz2 = 2 * len - 2;
        in = sz2 * (npy_intp)(-in / sz2) + in;
        in = in <= 1 - len ? in + sz2 : -in;
      }
      break;
    case NI_EXTEND_REFLECT:
      if (len <= 1) {
        in = 0;
      } else {
        npy_intp sz2 = 2 * len;
        if (in < -sz2)
          in = sz2 * (npy_intp)(-in / sz2) + in;
        // -1e-15 check to avoid possibility that: (-in - 1) == -1
        in = in < -len ? in + sz2 : (in > -1e-15 ? 1e-15 : -in) - 1;
      }
      break;
    case NI_EXTEND_WRAP:
      if (len <= 1) {
        in = 0;
      } else {
        npy_intp sz = len - 1;
        // Integer division of -in/sz gives (-in mod sz)
        // Note that 'in' is negative
        in += sz * ((npy_intp)(-in / sz) + 1);
      }
      break;
    case NI_EXTEND_GRID_WRAP:
      if (len <= 1) {
        in = 0;
      } else {
        // in = len - 1 + fmod(in + 1, len);
        in += len * ((npy_intp)((-1 - in) / len) + 1);
      }
      break;
    case NI_EXTEND_NEAREST:
      in = 0;
      break;
    case NI_EXTEND_CONSTANT:
      in = -1;
      break;
    }
  } else if (in > len - 1) {
    switch (mode) {
    case NI_EXTEND_MIRROR:
      if (len <= 1) {
        in = 0;
      } else {
        npy_intp sz2 = 2 * len - 2;
        in -= sz2 * (npy_intp)(in / sz2);
        if (in >= len)
          in = sz2 - in;
      }
      break;
    case NI_EXTEND_REFLECT:
      if (len <= 1) {
        in = 0;
      } else {
        npy_intp sz2 = 2 * len;
        in -= sz2 * (npy_intp)(in / sz2);
        if (in >= len)
          in = sz2 - in - 1;
      }
      break;
    case NI_EXTEND_WRAP:
      if (len <= 1) {
        in = 0;
      } else {
        npy_intp sz = len - 1;
        in -= sz * (npy_intp)(in / sz);
      }
      break;
    case NI_EXTEND_GRID_WRAP:
      if (len <= 1) {
        in = 0;
      } else {
        in -= len * (npy_intp)(in / len);
      }
      break;
    case NI_EXTEND_NEAREST:
      in = len - 1;
      break;
    case NI_EXTEND_CONSTANT:
      in = -1;
      break;
    }
  }

  return in;
}

#define BUFFER_SIZE 256000
#define TOLERANCE 1e-15

/* one-dimensional spline filter: */
int NI_SplineFilter1D(PyArrayObject *input, int order, int axis,
                      NI_ExtendMode mode, PyArrayObject *output) {
  int npoles = 0, more;
  npy_intp kk, lines, len;
  double *buffer = NULL, poles[MAX_SPLINE_FILTER_POLES];
  NI_LineBuffer iline_buffer, oline_buffer;
  NPY_BEGIN_THREADS_DEF;

  len = PyArray_NDIM(input) > 0 ? PyArray_DIM(input, axis) : 1;
  if (len < 1)
    goto exit;

  /* these are used in the spline filter calculation below: */
  if (get_filter_poles(order, &npoles, poles)) {
    goto exit;
  }

  /* allocate an initialize the line buffer, only a single one is used,
       because the calculation is in-place: */
  lines = -1;
  if (!NI_AllocateLineBuffer(input, axis, 0, 0, &lines, BUFFER_SIZE, &buffer)) {
    goto exit;
  }
  if (!NI_InitLineBuffer(input, axis, 0, 0, lines, buffer, NI_EXTEND_DEFAULT,
                         0.0, &iline_buffer)) {
    goto exit;
  }
  if (!NI_InitLineBuffer(output, axis, 0, 0, lines, buffer, NI_EXTEND_DEFAULT,
                         0.0, &oline_buffer)) {
    goto exit;
  }
  NPY_BEGIN_THREADS;

  /* iterate over all the array lines: */
  do {
    /* copy lines from array to buffer: */
    if (!NI_ArrayToLineBuffer(&iline_buffer, &lines, &more)) {
      goto exit;
    }
    /* iterate over the lines in the buffer: */
    for (kk = 0; kk < lines; kk++) {
      /* get line: */
      double *ln = NI_GET_LINE(iline_buffer, kk);
      /* spline filter: */
      if (len > 1) {
        apply_filter(ln, len, poles, npoles, mode);
      }
    }

    /* copy lines from buffer to array: */
    if (!NI_LineBufferToArray(&oline_buffer)) {
      goto exit;
    }
  } while (more);

exit:
  NPY_END_THREADS;
  free(buffer);
  return PyErr_Occurred() ? 0 : 1;
}

/* copy row of coordinate array from location at _p to _coor */
#define CASE_MAP_COORDINATES(_TYPE, _type, _p, _coor, _rank, _stride)          \
  case _TYPE: {                                                                \
    npy_intp _hh;                                                              \
    for (_hh = 0; _hh < _rank; ++_hh) {                                        \
      _coor[_hh] = *(_type *)_p;                                               \
      _p += _stride;                                                           \
    }                                                                          \
  } break

#define CASE_INTERP_COEFF(_TYPE, _type, _coeff, _pi, _idx)                     \
  case _TYPE:                                                                  \
    _coeff = *(_type *)(_pi + _idx);                                           \
    break

#define CASE_INTERP_OUT(_TYPE, _type, _po, _t)                                 \
  case _TYPE:                                                                  \
    *(_type *)_po = (_type)_t;                                                 \
    break

#define CASE_INTERP_OUT_UINT(_TYPE, _type, _po, _t)                            \
  case NPY_##_TYPE:                                                            \
    _t = _t > 0 ? _t + 0.5 : 0;                                                \
    _t = _t > NPY_MAX_##_TYPE ? NPY_MAX_##_TYPE : t;                           \
    _t = _t < 0 ? 0 : t;                                                       \
    *(_type *)_po = (_type)_t;                                                 \
    break

#define CASE_INTERP_OUT_INT(_TYPE, _type, _po, _t)                             \
  case NPY_##_TYPE:                                                            \
    _t = _t > 0 ? _t + 0.5 : _t - 0.5;                                         \
    _t = _t > NPY_MAX_##_TYPE ? NPY_MAX_##_TYPE : t;                           \
    _t = _t < NPY_MIN_##_TYPE ? NPY_MIN_##_TYPE : t;                           \
    *(_type *)_po = (_type)_t;                                                 \
    break

int _get_spline_boundary_mode(int mode) {
  if ((mode == NI_EXTEND_CONSTANT) || (mode == NI_EXTEND_WRAP))
    // Modes without an analytic prefilter or explicit prepadding use
    // mirror extension.
    return NI_EXTEND_MIRROR;
  return mode;
}

#if defined(SCIPY_HAVE_AVX2) || defined(SCIPY_HAVE_SSE2)

/* SSE2-compatible floor for float vectors - works without SSE4.1 */
static inline __m128 sse2_floor_ps(__m128 x) {
#ifdef __SSE4_1__
    if (has_sse41()) {
        return _mm_floor_ps(x);
    }
#endif
    /* SSE2 fallback: truncate towards zero and adjust for negative values */
    __m128i i = _mm_cvttps_epi32(x);
    __m128 trunc = _mm_cvtepi32_ps(i);
    __m128 mask = _mm_cmplt_ps(x, trunc);
    __m128 one = _mm_set1_ps(1.0f);
    return _mm_sub_ps(trunc, _mm_and_ps(mask, one));
}

/* SSE2 optimized 2D bilinear interpolation for float32 - interior only */
static int NI_GeometricTransform_2D_bilinear_f32_simd(
    PyArrayObject* input,
    PyArrayObject* output,
    const double* M, const double* shift,
    int nprepad, int mode, double cval)
{
    const npy_intp H = PyArray_DIM(input, 0);
    const npy_intp W = PyArray_DIM(input, 1);
    const npy_intp OH = PyArray_DIM(output, 0);
    const npy_intp OW = PyArray_DIM(output, 1);
    const float* in_data = (const float*)PyArray_DATA(input);
    float* out_data = (float*)PyArray_DATA(output);
    const npy_intp in_stride_y = PyArray_STRIDE(input, 0) / sizeof(float);
    const npy_intp in_stride_x = PyArray_STRIDE(input, 1) / sizeof(float);
    const npy_intp out_stride_y = PyArray_STRIDE(output, 0) / sizeof(float);
    const npy_intp out_stride_x = PyArray_STRIDE(output, 1) / sizeof(float);
    
    /* Reject tiny images and check for potential overflow */
    if (H < 2 || W < 2) return 0;
    if (H > INT_MAX - 1 || W > INT_MAX - 1) return 0;  /* Prevent int overflow */
    
    /* Matrix and shift as floats */
    const float m00 = (float)M[0], m01 = (float)M[1];
    const float m10 = (float)M[2], m11 = (float)M[3];
    const float sx = (float)(shift[0] + nprepad);
    const float sy = (float)(shift[1] + nprepad);
    
    const __m128 one = _mm_set1_ps(1.0f);
    
    /* Process output pixels */
    for (npy_intp oy = 0; oy < OH; ++oy) {
        float* out_row = out_data + oy * out_stride_y;
        
        /* Base source coordinates for this row (SciPy: matrix * [oy, ox]) */
        float base_x = m00 * (float)oy + m01 * 0 + sx;
        float base_y = m10 * (float)oy + m11 * 0 + sy;
        
        npy_intp ox = 0;
        
        /* Vectorized loop - process 4 pixels at a time */
        for (; ox + 3 < OW; ox += 4) {
            /* Generate x indices [ox, ox+1, ox+2, ox+3] */
            __m128 vx = _mm_setr_ps((float)ox, (float)(ox+1), 
                                    (float)(ox+2), (float)(ox+3));
            
            /* Compute source coordinates using affine transform (matrix * [oy, ox]) */
            /* Fix coordinate swap: xs should be column, ys should be row */
            __m128 ys = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(m01), vx),
                                   _mm_set1_ps(base_x));
            __m128 xs = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(m11), vx),
                                   _mm_set1_ps(base_y));
            
            /* Floor to get integer coordinates - use SSE2-compatible floor */
            __m128 xs_floor = sse2_floor_ps(xs);
            __m128 ys_floor = sse2_floor_ps(ys);
            __m128i xi = _mm_cvttps_epi32(xs_floor);
            __m128i yi = _mm_cvttps_epi32(ys_floor);
            
            /* Get fractional parts */
            __m128 fx = _mm_sub_ps(xs, xs_floor);
            __m128 fy = _mm_sub_ps(ys, ys_floor);
            
            /* Compute bilinear weights */
            __m128 one_minus_fx = _mm_sub_ps(one, fx);
            __m128 one_minus_fy = _mm_sub_ps(one, fy);
            
            __m128 w00 = _mm_mul_ps(one_minus_fx, one_minus_fy);
            __m128 w01 = _mm_mul_ps(fx, one_minus_fy);
            __m128 w10 = _mm_mul_ps(one_minus_fx, fy);
            __m128 w11 = _mm_mul_ps(fx, fy);
            
            /* Extract integer coordinates for bounds checking */
            SCIPY_ALIGN(16) int xi_arr[4];
            SCIPY_ALIGN(16) int yi_arr[4];
            _mm_storeu_si128((__m128i*)xi_arr, xi);
            _mm_storeu_si128((__m128i*)yi_arr, yi);
            
            SCIPY_ALIGN(16) float result[4];
            SCIPY_ALIGN(16) float w00_arr[4];
            SCIPY_ALIGN(16) float w01_arr[4];
            SCIPY_ALIGN(16) float w10_arr[4];
            SCIPY_ALIGN(16) float w11_arr[4];
            _mm_store_ps(w00_arr, w00);
            _mm_store_ps(w01_arr, w01);
            _mm_store_ps(w10_arr, w10);
            _mm_store_ps(w11_arr, w11);
            
            /* Process each pixel - check bounds and interpolate */
            for (int j = 0; j < 4; j++) {
                int x0 = xi_arr[j];
                int y0 = yi_arr[j];
                
                /* Check bounds for each of the 4 corners individually */
                float v00, v01, v10, v11;
                
                /* Corner (y0, x0) */
                if (y0 >= 0 && y0 < H && x0 >= 0 && x0 < W) {
                    v00 = in_data[y0 * in_stride_y + x0 * in_stride_x];
                } else {
                    v00 = (float)cval;
                }
                
                /* Corner (y0, x0+1) */
                if (y0 >= 0 && y0 < H && (x0+1) >= 0 && (x0+1) < W) {
                    v01 = in_data[y0 * in_stride_y + (x0+1) * in_stride_x];
                } else {
                    v01 = (float)cval;
                }
                
                /* Corner (y0+1, x0) */
                if ((y0+1) >= 0 && (y0+1) < H && x0 >= 0 && x0 < W) {
                    v10 = in_data[(y0+1) * in_stride_y + x0 * in_stride_x];
                } else {
                    v10 = (float)cval;
                }
                
                /* Corner (y0+1, x0+1) */
                if ((y0+1) >= 0 && (y0+1) < H && (x0+1) >= 0 && (x0+1) < W) {
                    v11 = in_data[(y0+1) * in_stride_y + (x0+1) * in_stride_x];
                } else {
                    v11 = (float)cval;
                }
                
                /* Bilinear interpolation */
                result[j] = v00 * w00_arr[j] + v01 * w01_arr[j] +
                           v10 * w10_arr[j] + v11 * w11_arr[j];
            }
            
            /* Store results */
            __m128 vresult = _mm_load_ps(result);
            _mm_storeu_ps(out_row + ox * out_stride_x, vresult);
        }
        
        /* Scalar cleanup for remaining pixels */
        for (; ox < OW; ox++) {
            /* Fix coordinate swap: x_src should be column, y_src should be row */
            float y_src = m01 * (float)ox + base_x;
            float x_src = m11 * (float)ox + base_y;
            
            int x0 = (int)floor(x_src);
            int y0 = (int)floor(y_src);
            
            float fx = x_src - x0;
            float fy = y_src - y0;
            
            /* Check bounds for each of the 4 corners individually */
            float v00, v01, v10, v11;
            
            /* Corner (y0, x0) */
            if (y0 >= 0 && y0 < H && x0 >= 0 && x0 < W) {
                v00 = in_data[y0 * in_stride_y + x0 * in_stride_x];
            } else {
                v00 = (float)cval;
            }
            
            /* Corner (y0, x0+1) */
            if (y0 >= 0 && y0 < H && (x0+1) >= 0 && (x0+1) < W) {
                v01 = in_data[y0 * in_stride_y + (x0+1) * in_stride_x];
            } else {
                v01 = (float)cval;
            }
            
            /* Corner (y0+1, x0) */
            if ((y0+1) >= 0 && (y0+1) < H && x0 >= 0 && x0 < W) {
                v10 = in_data[(y0+1) * in_stride_y + x0 * in_stride_x];
            } else {
                v10 = (float)cval;
            }
            
            /* Corner (y0+1, x0+1) */
            if ((y0+1) >= 0 && (y0+1) < H && (x0+1) >= 0 && (x0+1) < W) {
                v11 = in_data[(y0+1) * in_stride_y + (x0+1) * in_stride_x];
            } else {
                v11 = (float)cval;
            }
            
            out_row[ox * out_stride_x] = 
                v00 * (1 - fx) * (1 - fy) +
                v01 * fx * (1 - fy) +
                v10 * (1 - fx) * fy +
                v11 * fx * fy;
        }
    }
    
    return 1;
}

/* SSE2 optimized zoom/shift for 2D bilinear float32 */
static int NI_ZoomShift_2D_bilinear_f32_simd(
    PyArrayObject* input,
    PyArrayObject* output,
    const double* zoom, const double* shift,
    int nprepad, int grid_mode, int mode, double cval)
{
    const npy_intp H = PyArray_DIM(input, 0);
    const npy_intp W = PyArray_DIM(input, 1);
    const npy_intp OH = PyArray_DIM(output, 0);
    const npy_intp OW = PyArray_DIM(output, 1);
    const float* in_data = (const float*)PyArray_DATA(input);
    float* out_data = (float*)PyArray_DATA(output);
    const npy_intp in_stride_y = PyArray_STRIDE(input, 0) / sizeof(float);
    const npy_intp in_stride_x = PyArray_STRIDE(input, 1) / sizeof(float);
    const npy_intp out_stride_y = PyArray_STRIDE(output, 0) / sizeof(float);
    const npy_intp out_stride_x = PyArray_STRIDE(output, 1) / sizeof(float);
    
    /* Reject tiny images and check for potential overflow */
    if (H < 2 || W < 2) return 0;
    if (H > INT_MAX - 1 || W > INT_MAX - 1) return 0;  /* Prevent int overflow */
    
    /* Zoom and shift parameters */
    float zoom_x = zoom ? (float)zoom[1] : 1.0f;
    float zoom_y = zoom ? (float)zoom[0] : 1.0f;
    float shift_x = shift ? (float)shift[1] : 0.0f;
    float shift_y = shift ? (float)shift[0] : 0.0f;
    
    if (grid_mode) {
        /* Adjust for grid mode */
        shift_x += 0.5f * (1.0f / zoom_x - 1.0f);
        shift_y += 0.5f * (1.0f / zoom_y - 1.0f);
    }
    
    shift_x += (float)nprepad;
    shift_y += (float)nprepad;
    
    const __m128 zoom_x_vec = _mm_set1_ps(1.0f / zoom_x);
    const __m128 shift_x_vec = _mm_set1_ps(shift_x);
    
    /* Process each output row */
    for (npy_intp oy = 0; oy < OH; ++oy) {
        float* out_row = out_data + oy * out_stride_y;
        float y_src = ((float)oy + shift_y) / zoom_y;
        int y0 = (int)floor(y_src);
        float fy = y_src - y0;
        
        /* Skip this row if outside bounds */
        if (y0 < 0 || y0 >= H) {
            for (npy_intp ox = 0; ox < OW; ox++) {
                out_row[ox * out_stride_x] = (float)cval;
            }
            continue;
        }
        
        npy_intp ox = 0;
        
        /* Vectorized loop - process 4 pixels at a time */
        for (; ox + 3 < OW; ox += 4) {
            /* Compute source x coordinates */
            __m128 ox_vec = _mm_setr_ps((float)ox, (float)(ox+1),
                                        (float)(ox+2), (float)(ox+3));
            __m128 x_src = _mm_mul_ps(_mm_add_ps(ox_vec, shift_x_vec), zoom_x_vec);
            
            /* Floor to get integer coordinates */
            __m128i xi = _mm_cvttps_epi32(x_src);
            __m128 fx = _mm_sub_ps(x_src, _mm_cvtepi32_ps(xi));
            
            /* Extract for bounds checking */
            SCIPY_ALIGN(16) int xi_arr[4];
            SCIPY_ALIGN(16) float fx_arr[4];
            _mm_storeu_si128((__m128i*)xi_arr, xi);
            _mm_store_ps(fx_arr, fx);
            
            SCIPY_ALIGN(16) float result[4];
            
            for (int j = 0; j < 4; j++) {
                int x0 = xi_arr[j];
                
                float fx_val = fx_arr[j];
                
                /* Check bounds for each of the 4 corners and get values */
                float v00, v01, v10, v11;
                
                /* Corner (y0, x0) */
                if (y0 >= 0 && y0 < H && x0 >= 0 && x0 < W) {
                    v00 = in_data[y0 * in_stride_y + x0 * in_stride_x];
                } else {
                    v00 = (float)cval;
                }
                
                /* Corner (y0, x0+1) */
                if (y0 >= 0 && y0 < H && (x0+1) >= 0 && (x0+1) < W) {
                    v01 = in_data[y0 * in_stride_y + (x0+1) * in_stride_x];
                } else {
                    v01 = (float)cval;
                }
                
                /* Corner (y0+1, x0) */
                if ((y0+1) >= 0 && (y0+1) < H && x0 >= 0 && x0 < W) {
                    v10 = in_data[(y0+1) * in_stride_y + x0 * in_stride_x];
                } else {
                    v10 = (float)cval;
                }
                
                /* Corner (y0+1, x0+1) */
                if ((y0+1) >= 0 && (y0+1) < H && (x0+1) >= 0 && (x0+1) < W) {
                    v11 = in_data[(y0+1) * in_stride_y + (x0+1) * in_stride_x];
                } else {
                    v11 = (float)cval;
                }
                
                /* Bilinear interpolation */
                result[j] = v00 * (1 - fx_val) * (1 - fy) +
                           v01 * fx_val * (1 - fy) +
                           v10 * (1 - fx_val) * fy +
                           v11 * fx_val * fy;
            }
            
            /* Store results */
            __m128 vresult = _mm_load_ps(result);
            _mm_storeu_ps(out_row + ox * out_stride_x, vresult);
        }
        
        /* Scalar cleanup */
        for (; ox < OW; ox++) {
            float x_src = ((float)ox + shift_x) / zoom_x;
            int x0 = (int)floor(x_src);
            
            float fx = x_src - x0;
            
            /* Check bounds for each of the 4 corners and get values */
            float v00, v01, v10, v11;
            
            /* Corner (y0, x0) */
            if (y0 >= 0 && y0 < H && x0 >= 0 && x0 < W) {
                v00 = in_data[y0 * in_stride_y + x0 * in_stride_x];
            } else {
                v00 = (float)cval;
            }
            
            /* Corner (y0, x0+1) */
            if (y0 >= 0 && y0 < H && (x0+1) >= 0 && (x0+1) < W) {
                v01 = in_data[y0 * in_stride_y + (x0+1) * in_stride_x];
            } else {
                v01 = (float)cval;
            }
            
            /* Corner (y0+1, x0) */
            if ((y0+1) >= 0 && (y0+1) < H && x0 >= 0 && x0 < W) {
                v10 = in_data[(y0+1) * in_stride_y + x0 * in_stride_x];
            } else {
                v10 = (float)cval;
            }
            
            /* Corner (y0+1, x0+1) */
            if ((y0+1) >= 0 && (y0+1) < H && (x0+1) >= 0 && (x0+1) < W) {
                v11 = in_data[(y0+1) * in_stride_y + (x0+1) * in_stride_x];
            } else {
                v11 = (float)cval;
            }
                
            out_row[ox * out_stride_x] = 
                v00 * (1 - fx) * (1 - fy) +
                v01 * fx * (1 - fy) +
                v10 * (1 - fx) * fy +
                v11 * fx * fy;
        }
    }
    
    return 1;
}

/* SSE2 optimized 2D bilinear interpolation for float64 - interior only */
static int NI_GeometricTransform_2D_bilinear_f64_simd(
    PyArrayObject* input,
    PyArrayObject* output,
    const double* M, const double* shift,
    int nprepad, int mode, double cval)
{
    const npy_intp H = PyArray_DIM(input, 0);
    const npy_intp W = PyArray_DIM(input, 1);
    const npy_intp OH = PyArray_DIM(output, 0);
    const npy_intp OW = PyArray_DIM(output, 1);
    const double* in_data = (const double*)PyArray_DATA(input);
    double* out_data = (double*)PyArray_DATA(output);
    const npy_intp in_stride_y = PyArray_STRIDE(input, 0) / sizeof(double);
    const npy_intp in_stride_x = PyArray_STRIDE(input, 1) / sizeof(double);
    const npy_intp out_stride_y = PyArray_STRIDE(output, 0) / sizeof(double);
    const npy_intp out_stride_x = PyArray_STRIDE(output, 1) / sizeof(double);
    
    /* Reject tiny images and check for potential overflow */
    if (H < 2 || W < 2) return 0;
    if (H > INT_MAX - 1 || W > INT_MAX - 1) return 0;  /* Prevent int overflow */
    
    /* Matrix and shift as doubles */
    const double m00 = M[0], m01 = M[1];
    const double m10 = M[2], m11 = M[3];
    const double sx = shift[0] + nprepad;
    const double sy = shift[1] + nprepad;
    
    /* const __m128d one = _mm_set1_pd(1.0); */ /* Unused variable */
    
    /* Process output pixels */
    for (npy_intp oy = 0; oy < OH; ++oy) {
        double* out_row = out_data + oy * out_stride_y;
        
        /* Base source coordinates for this row (SciPy: matrix * [oy, ox]) */
        double base_x = m00 * (double)oy + m01 * 0 + sx;
        double base_y = m10 * (double)oy + m11 * 0 + sy;
        
        npy_intp ox = 0;
        
        /* Vectorized loop - process 2 pixels at a time (SSE2 handles 2 doubles) */
        for (; ox + 1 < OW; ox += 2) {
            /* Generate x indices [ox, ox+1] */
            __m128d vx = _mm_setr_pd((double)ox, (double)(ox+1));
            
            /* Compute source coordinates using affine transform (matrix * [oy, ox]) */
            /* Fix coordinate swap: xs should be column, ys should be row */
            __m128d ys = _mm_add_pd(_mm_mul_pd(_mm_set1_pd(m01), vx),
                                    _mm_set1_pd(base_x));
            __m128d xs = _mm_add_pd(_mm_mul_pd(_mm_set1_pd(m11), vx),
                                    _mm_set1_pd(base_y));
            
            /* Floor to get integer coordinates - manual floor for SSE2 */
            SCIPY_ALIGN(16) double xs_arr[2], ys_arr[2];
            _mm_store_pd(xs_arr, xs);
            _mm_store_pd(ys_arr, ys);
            
            SCIPY_ALIGN(16) double result[2];
            
            for (int j = 0; j < 2; j++) {
                double x_src = xs_arr[j];
                double y_src = ys_arr[j];
                int x0 = (int)floor(x_src);
                int y0 = (int)floor(y_src);
                
                double fx = x_src - x0;
                double fy = y_src - y0;
                
                /* Check bounds for each of the 4 corners individually */
                double v00, v01, v10, v11;
                
                /* Corner (y0, x0) */
                if (y0 >= 0 && y0 < H && x0 >= 0 && x0 < W) {
                    v00 = in_data[y0 * in_stride_y + x0 * in_stride_x];
                } else {
                    v00 = cval;
                }
                
                /* Corner (y0, x0+1) */
                if (y0 >= 0 && y0 < H && (x0+1) >= 0 && (x0+1) < W) {
                    v01 = in_data[y0 * in_stride_y + (x0+1) * in_stride_x];
                } else {
                    v01 = cval;
                }
                
                /* Corner (y0+1, x0) */
                if ((y0+1) >= 0 && (y0+1) < H && x0 >= 0 && x0 < W) {
                    v10 = in_data[(y0+1) * in_stride_y + x0 * in_stride_x];
                } else {
                    v10 = cval;
                }
                
                /* Corner (y0+1, x0+1) */
                if ((y0+1) >= 0 && (y0+1) < H && (x0+1) >= 0 && (x0+1) < W) {
                    v11 = in_data[(y0+1) * in_stride_y + (x0+1) * in_stride_x];
                } else {
                    v11 = cval;
                }
                
                /* Bilinear interpolation */
                result[j] = v00 * (1 - fx) * (1 - fy) +
                           v01 * fx * (1 - fy) +
                           v10 * (1 - fx) * fy +
                           v11 * fx * fy;
            }
            
            /* Store results */
            __m128d vresult = _mm_load_pd(result);
            _mm_storeu_pd(out_row + ox * out_stride_x, vresult);
        }
        
        /* Scalar cleanup for remaining pixel */
        for (; ox < OW; ox++) {
            /* Fix coordinate swap: x_src should be column, y_src should be row */
            double y_src = m01 * (double)ox + base_x;
            double x_src = m11 * (double)ox + base_y;
            
            int x0 = (int)floor(x_src);
            int y0 = (int)floor(y_src);
            
            double fx = x_src - x0;
            double fy = y_src - y0;
            
            /* Check bounds for each of the 4 corners individually */
            double v00, v01, v10, v11;
            
            /* Corner (y0, x0) */
            if (y0 >= 0 && y0 < H && x0 >= 0 && x0 < W) {
                v00 = in_data[y0 * in_stride_y + x0 * in_stride_x];
            } else {
                v00 = cval;
            }
            
            /* Corner (y0, x0+1) */
            if (y0 >= 0 && y0 < H && (x0+1) >= 0 && (x0+1) < W) {
                v01 = in_data[y0 * in_stride_y + (x0+1) * in_stride_x];
            } else {
                v01 = cval;
            }
            
            /* Corner (y0+1, x0) */
            if ((y0+1) >= 0 && (y0+1) < H && x0 >= 0 && x0 < W) {
                v10 = in_data[(y0+1) * in_stride_y + x0 * in_stride_x];
            } else {
                v10 = cval;
            }
            
            /* Corner (y0+1, x0+1) */
            if ((y0+1) >= 0 && (y0+1) < H && (x0+1) >= 0 && (x0+1) < W) {
                v11 = in_data[(y0+1) * in_stride_y + (x0+1) * in_stride_x];
            } else {
                v11 = cval;
            }
            
            out_row[ox * out_stride_x] = 
                v00 * (1 - fx) * (1 - fy) +
                v01 * fx * (1 - fy) +
                v10 * (1 - fx) * fy +
                v11 * fx * fy;
        }
    }
    
    return 1;
}

/* SSE2 optimized zoom/shift for 2D bilinear float64 */
static int NI_ZoomShift_2D_bilinear_f64_simd(
    PyArrayObject* input,
    PyArrayObject* output,
    const double* zoom, const double* shift,
    int nprepad, int grid_mode, int mode, double cval)
{
    const npy_intp H = PyArray_DIM(input, 0);
    const npy_intp W = PyArray_DIM(input, 1);
    const npy_intp OH = PyArray_DIM(output, 0);
    const npy_intp OW = PyArray_DIM(output, 1);
    
    /* Disable zoom SIMD - fallback to working scalar implementation */
    return 0;
    const double* in_data = (const double*)PyArray_DATA(input);
    double* out_data = (double*)PyArray_DATA(output);
    const npy_intp in_stride_y = PyArray_STRIDE(input, 0) / sizeof(double);
    const npy_intp in_stride_x = PyArray_STRIDE(input, 1) / sizeof(double);
    const npy_intp out_stride_y = PyArray_STRIDE(output, 0) / sizeof(double);
    const npy_intp out_stride_x = PyArray_STRIDE(output, 1) / sizeof(double);
    
    /* Reject tiny images and check for potential overflow */
    if (H < 2 || W < 2) return 0;
    if (H > INT_MAX - 1 || W > INT_MAX - 1) return 0;  /* Prevent int overflow */
    
    /* Zoom and shift parameters */
    double zoom_x = zoom ? zoom[1] : 1.0;
    double zoom_y = zoom ? zoom[0] : 1.0;
    double shift_x = shift ? shift[1] : 0.0;
    double shift_y = shift ? shift[0] : 0.0;
    
    if (grid_mode) {
        /* Adjust for grid mode */
        shift_x += 0.5 * (1.0 / zoom_x - 1.0);
        shift_y += 0.5 * (1.0 / zoom_y - 1.0);
    }
    
    shift_x += (double)nprepad;
    shift_y += (double)nprepad;
    
    const __m128d zoom_x_vec = _mm_set1_pd(1.0 / zoom_x);
    const __m128d shift_x_vec = _mm_set1_pd(shift_x);
    
    /* Process each output row */
    for (npy_intp oy = 0; oy < OH; ++oy) {
        double* out_row = out_data + oy * out_stride_y;
        double y_src = ((double)oy + shift_y) / zoom_y;
        int y0 = (int)floor(y_src);
        double fy = y_src - y0;
        
        /* Skip this row if completely outside bounds - allow edge interpolation */
        if (y0 < 0 || y0 >= H) {
            for (npy_intp ox = 0; ox < OW; ox++) {
                out_row[ox * out_stride_x] = cval;
            }
            continue;
        }
        
        npy_intp ox = 0;
        
        /* Vectorized loop - process 2 pixels at a time */
        for (; ox + 1 < OW; ox += 2) {
            /* Compute source x coordinates */
            __m128d ox_vec = _mm_setr_pd((double)ox, (double)(ox+1));
            __m128d x_src = _mm_mul_pd(_mm_add_pd(ox_vec, shift_x_vec), zoom_x_vec);
            
            /* Extract for processing */
            SCIPY_ALIGN(16) double x_src_arr[2];
            _mm_store_pd(x_src_arr, x_src);
            
            SCIPY_ALIGN(16) double result[2];
            
            for (int j = 0; j < 2; j++) {
                double xs = x_src_arr[j];
                int x0 = (int)floor(xs);
                
                double fx = xs - x0;
                
                /* Simplified zoom boundary check */
                if (x0 >= 0 && x0 < W - 1) {
                    /* Can safely do bilinear in x direction */
                    npy_intp idx00 = y0 * in_stride_y + x0 * in_stride_x;
                    npy_intp idx01 = idx00 + in_stride_x;
                    
                    double v00 = in_data[idx00];
                    double v01 = in_data[idx01];
                    
                    if (y0 < H - 1) {
                        /* Can safely access y0+1 */
                        npy_intp idx10 = idx00 + in_stride_y;
                        npy_intp idx11 = idx10 + in_stride_x;
                        
                        double v10 = in_data[idx10];
                        double v11 = in_data[idx11];
                        
                        result[j] = v00 * (1 - fx) * (1 - fy) +
                                   v01 * fx * (1 - fy) +
                                   v10 * (1 - fx) * fy +
                                   v11 * fx * fy;
                    } else {
                        /* y0 is last row, use y0 values with y0+1=cval */
                        result[j] = v00 * (1 - fx) * (1 - fy) +
                                   v01 * fx * (1 - fy) +
                                   cval * (1 - fx) * fy +
                                   cval * fx * fy;
                    }
                } else {
                    /* x boundary case or out of bounds */
                    result[j] = cval;
                }
            }
            
            /* Store results */
            __m128d vresult = _mm_load_pd(result);
            _mm_storeu_pd(out_row + ox * out_stride_x, vresult);
        }
        
        /* Scalar cleanup */
        for (; ox < OW; ox++) {
            double x_src = ((double)ox + shift_x) / zoom_x;
            int x0 = (int)floor(x_src);
            
            double fx = x_src - x0;
            
            /* Simplified zoom boundary check for scalar cleanup */
            if (x0 >= 0 && x0 < W - 1) {
                /* Can safely do bilinear in x direction */
                npy_intp idx00 = y0 * in_stride_y + x0 * in_stride_x;
                npy_intp idx01 = idx00 + in_stride_x;
                
                double v00 = in_data[idx00];
                double v01 = in_data[idx01];
                
                if (y0 < H - 1) {
                    /* Can safely access y0+1 */
                    npy_intp idx10 = idx00 + in_stride_y;
                    npy_intp idx11 = idx10 + in_stride_x;
                    
                    double v10 = in_data[idx10];
                    double v11 = in_data[idx11];
                    
                    out_row[ox * out_stride_x] = 
                        v00 * (1 - fx) * (1 - fy) +
                        v01 * fx * (1 - fy) +
                        v10 * (1 - fx) * fy +
                        v11 * fx * fy;
                } else {
                    /* y0 is last row, use y0 values with y0+1=cval */
                    out_row[ox * out_stride_x] = 
                        v00 * (1 - fx) * (1 - fy) +
                        v01 * fx * (1 - fy) +
                        cval * (1 - fx) * fy +
                        cval * fx * fy;
                }
            } else {
                /* x boundary case or out of bounds */
                out_row[ox * out_stride_x] = cval;
            }
        }
    }
    
    return 1;
}

#endif /* SCIPY_HAVE_SSE2 */

int NI_GeometricTransform(PyArrayObject *input,
                          int (*map)(npy_intp *, double *, int, int, void *),
                          void *map_data, PyArrayObject *matrix_ar,
                          PyArrayObject *shift_ar, PyArrayObject *coordinates,
                          PyArrayObject *output, int order, int mode,
                          double cval, int nprepad) {
  char *po, *pi, *pc = NULL;
  npy_intp **edge_offsets = NULL, **data_offsets = NULL, filter_size;
  char **edge_grid_const = NULL;
  npy_intp ftmp[NPY_MAXDIMS], *fcoordinates = NULL, *foffsets = NULL;
  npy_intp cstride = 0, kk, hh, ll, jj;
  npy_intp size;
  double **splvals = NULL, icoor[NPY_MAXDIMS], tmp;
  npy_intp idimensions[NPY_MAXDIMS], istrides[NPY_MAXDIMS];
  NI_Iterator io, ic;
  npy_double *matrix = matrix_ar ? (npy_double *)PyArray_DATA(matrix_ar) : NULL;
  npy_double *shift = shift_ar ? (npy_double *)PyArray_DATA(shift_ar) : NULL;
  int irank = 0, orank, spline_mode;
  NPY_BEGIN_THREADS_DEF;
  
  /* Check environment variable before releasing GIL for thread safety */
  const int use_simd = should_use_simd();
  
  /* Get dimensions before starting threads */
  irank = PyArray_NDIM(input);
  orank = PyArray_NDIM(output);
  

  NPY_BEGIN_THREADS;

  for (kk = 0; kk < PyArray_NDIM(input); kk++) {
    idimensions[kk] = PyArray_DIM(input, kk);
    istrides[kk] = PyArray_STRIDE(input, kk);
  }
  
#if defined(SCIPY_HAVE_AVX2) || defined(SCIPY_HAVE_SSE2)
  /* Check if we can use SIMD fast paths - very restrictive for correctness */
  
  /* Check for contiguous last axis (unit stride requirement) */
  const npy_intp input_itemsize = PyArray_ITEMSIZE(input);
  const npy_intp output_itemsize = PyArray_ITEMSIZE(output);
  const int input_contiguous = (irank > 0) && 
      (PyArray_STRIDE(input, irank-1) == input_itemsize);
  const int output_contiguous = (orank > 0) && 
      (PyArray_STRIDE(output, orank-1) == output_itemsize);
  
  const int can_use_simd_2d = 
      (matrix != NULL) &&                    /* Affine transform */
      (order == 1) &&                        /* Bilinear interpolation */
      (irank == 2) &&                        /* 2D input */
      (orank == 2) &&                        /* 2D output */
      ((PyArray_TYPE(input) == NPY_FLOAT && PyArray_TYPE(output) == NPY_FLOAT) ||  /* float32 */
       (PyArray_TYPE(input) == NPY_DOUBLE && PyArray_TYPE(output) == NPY_DOUBLE)) && /* float64 */
      (mode == NI_EXTEND_CONSTANT) &&        /* Only constant mode for now */
      (coordinates == NULL) &&               /* Not using coordinate arrays */
      (map == NULL) &&                       /* Not using mapping function */
      (input_contiguous && output_contiguous); /* Contiguous arrays only */
  
  const int can_use_simd_3d = 
      (matrix != NULL) &&                    /* Affine transform */
      (order == 1) &&                        /* Trilinear interpolation */
      (irank == 3) &&                        /* 3D input */
      (orank == 3) &&                        /* 3D output */
      ((PyArray_TYPE(input) == NPY_FLOAT && PyArray_TYPE(output) == NPY_FLOAT) ||  /* float32 */
       (PyArray_TYPE(input) == NPY_DOUBLE && PyArray_TYPE(output) == NPY_DOUBLE)) && /* float64 */
      (mode == NI_EXTEND_CONSTANT) &&        /* Only constant mode for now */
      (coordinates == NULL) &&               /* Not using coordinate arrays */
      (map == NULL) &&                       /* Not using mapping function */
      (input_contiguous && output_contiguous); /* Contiguous arrays only */
  
  if ((can_use_simd_2d || can_use_simd_3d) && use_simd) {
    /* Keep the SIMD code running without the GIL - it's thread-safe */
    int result = 0;
    
#ifdef __AVX2__
    /* Prefer AVX2 if available and runtime detected */
    if (has_avx2()) {
      if (can_use_simd_2d) {
        if (PyArray_TYPE(input) == NPY_FLOAT) {
          result = NI_GeometricTransform_2D_bilinear_f32_avx2(
              input, output, matrix, shift, nprepad, mode, cval);
        } else if (PyArray_TYPE(input) == NPY_DOUBLE) {
          result = NI_GeometricTransform_2D_bilinear_f64_avx2(
              input, output, matrix, shift, nprepad, mode, cval);
        }
      } else if (can_use_simd_3d) {
        if (PyArray_TYPE(input) == NPY_DOUBLE) {
          result = NI_GeometricTransform_3D_trilinear_f64_avx2(
              input, output, matrix, shift, nprepad, mode, cval);
        } else if (PyArray_TYPE(input) == NPY_FLOAT) {
          result = NI_GeometricTransform_3D_trilinear_f32_avx2(
              input, output, matrix, shift, nprepad, mode, cval);
        }
      }
    }
#endif
    
#ifdef SCIPY_BUILD_SSE2
    /* Fall back to SSE2 for 2D only if AVX2 not available */
    if (!result && can_use_simd_2d && has_sse2()) {
      if (PyArray_TYPE(input) == NPY_FLOAT) {
        result = NI_GeometricTransform_2D_bilinear_f32_simd(
            input, output, matrix, shift, nprepad, mode, cval);
      } else if (PyArray_TYPE(input) == NPY_DOUBLE) {
        result = NI_GeometricTransform_2D_bilinear_f64_simd(
            input, output, matrix, shift, nprepad, mode, cval);
      }
    }
#endif
    
    if (result) {
      NPY_END_THREADS;
      return 1;  /* Success - fast path handled everything */
    }
    /* If fast path failed, continue with generic implementation */
  } else if ((can_use_simd_2d || can_use_simd_3d) && !use_simd) {
    /* SIMD was available but disabled by environment variable */
    static int warning_shown = 0;
    if (!warning_shown) {
      NPY_END_THREADS;
      if (PyErr_WarnEx(PyExc_UserWarning, 
          "SIMD optimizations available for bilinear/trilinear interpolation but disabled. "
          "To enable, unset SCIPY_FORCE_SCALAR_INTERPOLATION environment variable.", 1) < 0) {
        /* Warning failed, but continue anyway */
        PyErr_Clear();
      }
      warning_shown = 1;
      NPY_BEGIN_THREADS;
    }
  } else if ((order == 1) && (matrix != NULL) && 
             ((irank == 2 && orank == 2) || (irank == 3 && orank == 3)) &&
             (mode == NI_EXTEND_CONSTANT) && 
             ((PyArray_TYPE(input) == NPY_FLOAT && PyArray_TYPE(output) == NPY_FLOAT) ||
              (PyArray_TYPE(input) == NPY_DOUBLE && PyArray_TYPE(output) == NPY_DOUBLE))) {
    /* SIMD could be used but conditions not met */
    static int info_shown = 0;
    if (!info_shown && (coordinates == NULL) && (map == NULL)) {
      NPY_END_THREADS;
      const char* reason = NULL;
      if (!input_contiguous || !output_contiguous) {
        reason = "non-contiguous arrays";
      }
      if (reason && PyErr_WarnEx(PyExc_UserWarning, 
          "SIMD optimizations not used for bilinear/trilinear interpolation due to non-contiguous arrays. "
          "Consider using numpy.ascontiguousarray() for better performance.", 1) < 0) {
        /* Warning failed, but continue anyway */
        PyErr_Clear();
      }
      info_shown = 1;
      NPY_BEGIN_THREADS;
    }
  }
#endif

  /* if the mapping is from array coordinates: */
  if (coordinates) {
    /* initialize a line iterator along the first axis: */
    if (!NI_InitPointIterator(coordinates, &ic))
      goto exit;
    cstride = ic.strides[0];
    if (!NI_LineIterator(&ic, 0))
      goto exit;
    pc = (void *)(PyArray_DATA(coordinates));
  }

  /* offsets used at the borders: */
  edge_offsets = malloc(irank * sizeof(npy_intp *));
  data_offsets = malloc(irank * sizeof(npy_intp *));
  if (NPY_UNLIKELY(!edge_offsets || !data_offsets)) {
    NPY_END_THREADS;
    PyErr_NoMemory();
    goto exit;
  }

  if (mode == NI_EXTEND_GRID_CONSTANT) {
    // boolean indicating if the current point in the filter footprint is
    // outside the bounds
    edge_grid_const = malloc(irank * sizeof(char *));
    if (NPY_UNLIKELY(!edge_grid_const)) {
      NPY_END_THREADS;
      PyErr_NoMemory();
      goto exit;
    }
    for (jj = 0; jj < irank; jj++)
      edge_grid_const[jj] = NULL;
    for (jj = 0; jj < irank; jj++) {
      edge_grid_const[jj] = malloc((order + 1) * sizeof(char));
      if (NPY_UNLIKELY(!edge_grid_const[jj])) {
        NPY_END_THREADS;
        PyErr_NoMemory();
        goto exit;
      }
    }
  }

  for (jj = 0; jj < irank; jj++)
    data_offsets[jj] = NULL;
  for (jj = 0; jj < irank; jj++) {
    data_offsets[jj] = malloc((order + 1) * sizeof(npy_intp));
    if (NPY_UNLIKELY(!data_offsets[jj])) {
      NPY_END_THREADS;
      PyErr_NoMemory();
      goto exit;
    }
  }
  /* will hold the spline coefficients: */
  splvals = malloc(irank * sizeof(double *));
  if (NPY_UNLIKELY(!splvals)) {
    NPY_END_THREADS;
    PyErr_NoMemory();
    goto exit;
  }
  for (jj = 0; jj < irank; jj++)
    splvals[jj] = NULL;
  for (jj = 0; jj < irank; jj++) {
    splvals[jj] = malloc((order + 1) * sizeof(double));
    if (NPY_UNLIKELY(!splvals[jj])) {
      NPY_END_THREADS;
      PyErr_NoMemory();
      goto exit;
    }
  }

  filter_size = 1;
  for (jj = 0; jj < irank; jj++)
    filter_size *= order + 1;

  /* initialize output iterator: */
  if (!NI_InitPointIterator(output, &io))
    goto exit;

  /* get data pointers: */
  pi = (void *)PyArray_DATA(input);
  po = (void *)PyArray_DATA(output);

  /* make a table of all possible coordinates within the spline filter: */
  fcoordinates = malloc(irank * filter_size * sizeof(npy_intp));
  /* make a table of all offsets within the spline filter: */
  foffsets = malloc(filter_size * sizeof(npy_intp));
  if (NPY_UNLIKELY(!fcoordinates || !foffsets)) {
    NPY_END_THREADS;
    PyErr_NoMemory();
    goto exit;
  }
  for (jj = 0; jj < irank; jj++)
    ftmp[jj] = 0;
  kk = 0;
  for (hh = 0; hh < filter_size; hh++) {
    for (jj = 0; jj < irank; jj++)
      fcoordinates[jj + hh * irank] = ftmp[jj];
    foffsets[hh] = kk;
    for (jj = irank - 1; jj >= 0; jj--) {
      if (ftmp[jj] < order) {
        ftmp[jj]++;
        kk += istrides[jj];
        break;
      } else {
        ftmp[jj] = 0;
        kk -= istrides[jj] * order;
      }
    }
  }

  spline_mode = _get_spline_boundary_mode(mode);

  size = PyArray_SIZE(output);
  for (kk = 0; kk < size; kk++) {
    double t = 0.0;
    int constant = 0, edge = 0;
    npy_intp offset = 0;
    if (mode == NI_EXTEND_GRID_CONSTANT) {
      // reset edge flags for each location in the filter footprint
      for (hh = 0; hh < irank; hh++) {
        for (ll = 0; ll <= order; ll++) {
          edge_grid_const[hh][ll] = 0;
        }
      }
    }
    if (map) {
      NPY_END_THREADS;
      /* call mappint functions: */
      if (!map(io.coordinates, icoor, orank, irank, map_data)) {
        if (!PyErr_Occurred())
          PyErr_SetString(PyExc_RuntimeError,
                          "unknown error in mapping function");
        goto exit;
      }
      NPY_BEGIN_THREADS;
    } else if (matrix) {
      /* do an affine transformation: */
      npy_double *p = matrix;
      for (hh = 0; hh < irank; hh++) {
        tmp = shift[hh];
        ll = 0;
        for (; ll + 1 < orank; ll += 2) {
          tmp += io.coordinates[ll] * *p++;
          tmp += io.coordinates[ll + 1] * *p++;
        }
        if (ll < orank) {
          tmp += io.coordinates[ll] * *p++;
        }
        icoor[hh] = tmp;
      }
    } else if (coordinates) {
      /* mapping is from an coordinates array: */
      char *p = pc;
      switch (PyArray_TYPE(coordinates)) {
        CASE_MAP_COORDINATES(NPY_BOOL, npy_bool, p, icoor, irank, cstride);
        CASE_MAP_COORDINATES(NPY_UBYTE, npy_ubyte, p, icoor, irank, cstride);
        CASE_MAP_COORDINATES(NPY_USHORT, npy_ushort, p, icoor, irank, cstride);
        CASE_MAP_COORDINATES(NPY_UINT, npy_uint, p, icoor, irank, cstride);
        CASE_MAP_COORDINATES(NPY_ULONG, npy_ulong, p, icoor, irank, cstride);
        CASE_MAP_COORDINATES(NPY_ULONGLONG, npy_ulonglong, p, icoor, irank,
                             cstride);
        CASE_MAP_COORDINATES(NPY_BYTE, npy_byte, p, icoor, irank, cstride);
        CASE_MAP_COORDINATES(NPY_SHORT, npy_short, p, icoor, irank, cstride);
        CASE_MAP_COORDINATES(NPY_INT, npy_int, p, icoor, irank, cstride);
        CASE_MAP_COORDINATES(NPY_LONG, npy_long, p, icoor, irank, cstride);
        CASE_MAP_COORDINATES(NPY_LONGLONG, npy_longlong, p, icoor, irank,
                             cstride);
        CASE_MAP_COORDINATES(NPY_FLOAT, npy_float, p, icoor, irank, cstride);
        CASE_MAP_COORDINATES(NPY_DOUBLE, npy_double, p, icoor, irank, cstride);
      default:
        NPY_END_THREADS;
        PyErr_SetString(PyExc_RuntimeError,
                        "coordinate array data type not supported");
        goto exit;
      }
    }

    /* iterate over axes: */
    for (hh = 0; hh < irank; hh++) {
      double cc = icoor[hh] + nprepad;
      if ((mode != NI_EXTEND_GRID_CONSTANT) && (mode != NI_EXTEND_NEAREST)) {
        /* if the input coordinate is outside the borders, map it: */
        cc = map_coordinate(cc, idimensions[hh], mode);
      }
      if (cc > -1.0 || mode == NI_EXTEND_GRID_CONSTANT ||
          mode == NI_EXTEND_NEAREST) {
        /* find the filter location along this axis: */
        npy_intp start;
        if (order & 1) {
          start = (npy_intp)floor(cc) - order / 2;
        } else {
          start = (npy_intp)floor(cc + 0.5) - order / 2;
        }
        /* get the offset to the start of the filter: */
        offset += istrides[hh] * start;
        npy_intp idx = 0;

        if (mode == NI_EXTEND_GRID_CONSTANT) {
          // Determine locations in the filter footprint that are
          // outside the range.
          for (ll = 0; ll <= order; ll++) {
            idx = start + ll;
            edge_grid_const[hh][ll] = (idx < 0 || idx >= idimensions[hh]);
          }
        } else {

          if (start < 0 || start + order >= idimensions[hh]) {
            /* implement border mapping, if outside border: */
            edge = 1;
            edge_offsets[hh] = data_offsets[hh];

            for (ll = 0; ll <= order; ll++) {
              idx = start + ll;
              idx = (npy_intp)map_coordinate(idx, idimensions[hh], spline_mode);

              /* calculate and store the offsets at this edge: */
              edge_offsets[hh][ll] = istrides[hh] * (idx - start);
            }
          } else {
            /* we are not at the border, use precalculated offsets: */
            edge_offsets[hh] = NULL;
          }
        }
        if (order != 0) {
          get_spline_interpolation_weights(cc, order, splvals[hh]);
        }
      } else {
        /* we use the constant border condition: */
        constant = 1;
        break;
      }
    }

    if (!constant) {
      npy_intp *ff = fcoordinates;
      const int type_num = PyArray_TYPE(input);
      t = 0.0;
      
      /* Original scalar implementation */
      for (hh = 0; hh < filter_size; hh++) {
          double coeff = 0.0;
          npy_intp idx = 0;
          char is_cval = 0;
          if (mode == NI_EXTEND_GRID_CONSTANT) {
            for (ll = 0; ll < irank; ll++) {
              if (edge_grid_const[ll][ff[ll]]) {
                is_cval = 1;
              }
            }
          }
          if (is_cval) {
            coeff = cval;
          } else {
            if (NPY_UNLIKELY(edge)) {
              for (ll = 0; ll < irank; ll++) {
                if (edge_offsets[ll])
                  idx += edge_offsets[ll][ff[ll]];
                else
                  idx += ff[ll] * istrides[ll];
              }
            } else {
              idx = foffsets[hh];
            }
            idx += offset;
            switch (type_num) {
              CASE_INTERP_COEFF(NPY_BOOL, npy_bool, coeff, pi, idx);
              CASE_INTERP_COEFF(NPY_UBYTE, npy_ubyte, coeff, pi, idx);
              CASE_INTERP_COEFF(NPY_USHORT, npy_ushort, coeff, pi, idx);
              CASE_INTERP_COEFF(NPY_UINT, npy_uint, coeff, pi, idx);
              CASE_INTERP_COEFF(NPY_ULONG, npy_ulong, coeff, pi, idx);
              CASE_INTERP_COEFF(NPY_ULONGLONG, npy_ulonglong, coeff, pi, idx);
              CASE_INTERP_COEFF(NPY_BYTE, npy_byte, coeff, pi, idx);
              CASE_INTERP_COEFF(NPY_SHORT, npy_short, coeff, pi, idx);
              CASE_INTERP_COEFF(NPY_INT, npy_int, coeff, pi, idx);
              CASE_INTERP_COEFF(NPY_LONG, npy_long, coeff, pi, idx);
              CASE_INTERP_COEFF(NPY_LONGLONG, npy_longlong, coeff, pi, idx);
              CASE_INTERP_COEFF(NPY_FLOAT, npy_float, coeff, pi, idx);
              CASE_INTERP_COEFF(NPY_DOUBLE, npy_double, coeff, pi, idx);
            default:
              NPY_END_THREADS;
              PyErr_SetString(PyExc_RuntimeError, "data type not supported");
              goto exit;
            }
          }
          /* calculate the interpolated value: */
          for (ll = 0; ll < irank; ll++)
            if (order > 0)
              coeff *= splvals[ll][ff[ll]];
          t += coeff;
          ff += irank;
        }
    } else {
      t = cval;
    }
    /* store output value: */
    switch (PyArray_TYPE(output)) {
      CASE_INTERP_OUT(NPY_BOOL, npy_bool, po, t);
      CASE_INTERP_OUT_UINT(UBYTE, npy_ubyte, po, t);
      CASE_INTERP_OUT_UINT(USHORT, npy_ushort, po, t);
      CASE_INTERP_OUT_UINT(UINT, npy_uint, po, t);
      CASE_INTERP_OUT_UINT(ULONG, npy_ulong, po, t);
      CASE_INTERP_OUT_UINT(ULONGLONG, npy_ulonglong, po, t);
      CASE_INTERP_OUT_INT(BYTE, npy_byte, po, t);
      CASE_INTERP_OUT_INT(SHORT, npy_short, po, t);
      CASE_INTERP_OUT_INT(INT, npy_int, po, t);
      CASE_INTERP_OUT_INT(LONG, npy_long, po, t);
      CASE_INTERP_OUT_INT(LONGLONG, npy_longlong, po, t);
      CASE_INTERP_OUT(NPY_FLOAT, npy_float, po, t);
      CASE_INTERP_OUT(NPY_DOUBLE, npy_double, po, t);
    default:
      NPY_END_THREADS;
      PyErr_SetString(PyExc_RuntimeError, "data type not supported");
      goto exit;
    }
    if (coordinates) {
      NI_ITERATOR_NEXT2(io, ic, po, pc);
    } else {
      NI_ITERATOR_NEXT(io, po);
    }
  }

exit:
  NPY_END_THREADS;
  free(edge_offsets);
  if (edge_grid_const) {
    for (jj = 0; jj < irank; jj++)
      free(edge_grid_const[jj]);
    free(edge_grid_const);
  }
  if (data_offsets) {
    for (jj = 0; jj < irank; jj++)
      free(data_offsets[jj]);
    free(data_offsets);
  }
  if (splvals) {
    for (jj = 0; jj < irank; jj++)
      free(splvals[jj]);
    free(splvals);
  }
  free(foffsets);
  free(fcoordinates);
  return PyErr_Occurred() ? 0 : 1;
}

int NI_ZoomShift(PyArrayObject *input, PyArrayObject *zoom_ar,
                 PyArrayObject *shift_ar, PyArrayObject *output, int order,
                 int mode, double cval, int nprepad, int grid_mode) {
  char *po, *pi;
  npy_intp **zeros = NULL, **offsets = NULL, ***edge_offsets = NULL;
  npy_intp ftmp[NPY_MAXDIMS], *fcoordinates = NULL, *foffsets = NULL;
  npy_intp jj, hh, kk, filter_size, odimensions[NPY_MAXDIMS];
  npy_intp idimensions[NPY_MAXDIMS], istrides[NPY_MAXDIMS];
  npy_intp size;
  char ***edge_grid_const = NULL;
  double ***splvals = NULL;
  NI_Iterator io;
  npy_double *zooms = zoom_ar ? (npy_double *)PyArray_DATA(zoom_ar) : NULL;
  npy_double *shifts = shift_ar ? (npy_double *)PyArray_DATA(shift_ar) : NULL;
  int rank = 0;
  NPY_BEGIN_THREADS_DEF;
  
  /* Check environment variable before any thread operations for thread safety */
  const int use_simd = should_use_simd();

#if defined(SCIPY_HAVE_AVX2) || defined(SCIPY_HAVE_SSE2)
  /* Check if we can use SIMD fast path for zoom/shift - restrictive for correctness */
  const npy_intp input_itemsize_2d = PyArray_ITEMSIZE(input);
  const npy_intp output_itemsize_2d = PyArray_ITEMSIZE(output);
  const int input_2d_contiguous = (PyArray_NDIM(input) == 2) && 
      (PyArray_STRIDE(input, 1) == input_itemsize_2d);
  const int output_2d_contiguous = (PyArray_NDIM(output) == 2) && 
      (PyArray_STRIDE(output, 1) == output_itemsize_2d);
  
  const int can_use_simd_fastpath = 
      (order == 1) &&                        /* Bilinear interpolation */
      (PyArray_NDIM(input) == 2) &&          /* 2D input */
      (PyArray_NDIM(output) == 2) &&         /* 2D output */
      ((PyArray_TYPE(input) == NPY_FLOAT && PyArray_TYPE(output) == NPY_FLOAT) ||  /* float32 */
       (PyArray_TYPE(input) == NPY_DOUBLE && PyArray_TYPE(output) == NPY_DOUBLE)) && /* float64 */
      (mode == NI_EXTEND_CONSTANT) &&        /* Only constant mode */
      (input_2d_contiguous && output_2d_contiguous); /* Contiguous arrays only */
  
  if (can_use_simd_fastpath && use_simd) {
    double zoom_vals[2] = {1.0, 1.0};
    double shift_vals[2] = {0.0, 0.0};
    if (zooms) {
      zoom_vals[0] = zooms[0];
      zoom_vals[1] = zooms[1];
    }
    if (shifts) {
      shift_vals[0] = shifts[0];
      shift_vals[1] = shifts[1];
    }
    
    int result = 0;
    /* AVX2 and SSE2 use the same function interface for zoom/shift */
    if (PyArray_TYPE(input) == NPY_FLOAT) {
      result = NI_ZoomShift_2D_bilinear_f32_simd(
          input, output, zoom_vals, shift_vals, nprepad, grid_mode, mode, cval);
    } else if (PyArray_TYPE(input) == NPY_DOUBLE) {
      result = NI_ZoomShift_2D_bilinear_f64_simd(
          input, output, zoom_vals, shift_vals, nprepad, grid_mode, mode, cval);
    }
    if (result) {
      return 1;  /* Success - fast path handled everything */
    }
    /* If fast path failed, fall through to generic implementation */
  } else if (can_use_simd_fastpath && !use_simd) {
    /* SIMD was available but disabled by environment variable */
    static int zoom_warning_shown = 0;
    if (!zoom_warning_shown) {
      if (PyErr_WarnEx(PyExc_UserWarning, 
          "SIMD optimizations available for zoom/shift bilinear interpolation but disabled. "
          "To enable, unset SCIPY_FORCE_SCALAR_INTERPOLATION environment variable.", 1) < 0) {
        /* Warning failed, but continue anyway */
        PyErr_Clear();
      }
      zoom_warning_shown = 1;
    }
  } else if ((order == 1) && (PyArray_NDIM(input) == 2) && (PyArray_NDIM(output) == 2) &&
             (mode == NI_EXTEND_CONSTANT) && 
             ((PyArray_TYPE(input) == NPY_FLOAT && PyArray_TYPE(output) == NPY_FLOAT) ||
              (PyArray_TYPE(input) == NPY_DOUBLE && PyArray_TYPE(output) == NPY_DOUBLE))) {
    /* SIMD could be used but conditions not met */
    static int zoom_info_shown = 0;
    if (!zoom_info_shown) {
      const char* reason = NULL;
      if (!input_2d_contiguous || !output_2d_contiguous) {
        reason = "non-contiguous arrays";
      }
      if (reason && PyErr_WarnEx(PyExc_UserWarning, 
          "SIMD optimizations not used for zoom/shift bilinear interpolation due to non-contiguous arrays. "
          "Consider using numpy.ascontiguousarray() for better performance.", 1) < 0) {
        /* Warning failed, but continue anyway */
        PyErr_Clear();
      }
      zoom_info_shown = 1;
    }
  }
#endif

  NPY_BEGIN_THREADS;

  for (kk = 0; kk < PyArray_NDIM(input); kk++) {
    idimensions[kk] = PyArray_DIM(input, kk);
    istrides[kk] = PyArray_STRIDE(input, kk);
    odimensions[kk] = PyArray_DIM(output, kk);
  }
  rank = PyArray_NDIM(input);

  /* if the mode is 'constant' we need some temps later: */
  if (mode == NI_EXTEND_CONSTANT) {
    zeros = malloc(rank * sizeof(npy_intp *));
    if (NPY_UNLIKELY(!zeros)) {
      NPY_END_THREADS;
      PyErr_NoMemory();
      goto exit;
    }
    for (jj = 0; jj < rank; jj++)
      zeros[jj] = NULL;
    for (jj = 0; jj < rank; jj++) {
      zeros[jj] = malloc(odimensions[jj] * sizeof(npy_intp));
      if (NPY_UNLIKELY(!zeros[jj])) {
        NPY_END_THREADS;
        PyErr_NoMemory();
        goto exit;
      }
    }
  } else if (mode == NI_EXTEND_GRID_CONSTANT) {
    // boolean indicating if the current point in the filter footprint is
    // outside the bounds
    edge_grid_const = malloc(rank * sizeof(char *));
    if (NPY_UNLIKELY(!edge_grid_const)) {
      NPY_END_THREADS;
      PyErr_NoMemory();
      goto exit;
    }
    for (jj = 0; jj < rank; jj++)
      edge_grid_const[jj] = NULL;
    for (jj = 0; jj < rank; jj++) {
      edge_grid_const[jj] = malloc(odimensions[jj] * sizeof(char *));
      if (NPY_UNLIKELY(!edge_grid_const[jj])) {
        NPY_END_THREADS;
        PyErr_NoMemory();
        goto exit;
      }
      for (hh = 0; hh < odimensions[jj]; hh++) {
        edge_grid_const[jj][hh] = NULL;
      }
    }
  }
  /* store offsets, along each axis: */
  offsets = malloc(rank * sizeof(npy_intp *));
  /* store spline coefficients, along each axis: */
  splvals = malloc(rank * sizeof(double **));
  /* store offsets at all edges: */

  if (NPY_UNLIKELY(!offsets || !splvals)) {
    NPY_END_THREADS;
    PyErr_NoMemory();
    goto exit;
  }
  for (jj = 0; jj < rank; jj++) {
    offsets[jj] = NULL;
    splvals[jj] = NULL;
  }
  for (jj = 0; jj < rank; jj++) {
    offsets[jj] = malloc(odimensions[jj] * sizeof(npy_intp));
    splvals[jj] = malloc(odimensions[jj] * sizeof(double *));
    if (NPY_UNLIKELY(!offsets[jj] || !splvals[jj])) {
      NPY_END_THREADS;
      PyErr_NoMemory();
      goto exit;
    }
    for (hh = 0; hh < odimensions[jj]; hh++) {
      splvals[jj][hh] = NULL;
    }
  }

  if (mode != NI_EXTEND_GRID_CONSTANT) {
    edge_offsets = malloc(rank * sizeof(npy_intp **));
    if (NPY_UNLIKELY(!edge_offsets)) {
      NPY_END_THREADS;
      PyErr_NoMemory();
      goto exit;
    }
    for (jj = 0; jj < rank; jj++) {
      edge_offsets[jj] = NULL;
    }
    for (jj = 0; jj < rank; jj++) {
      edge_offsets[jj] = malloc(odimensions[jj] * sizeof(npy_intp *));
      if (NPY_UNLIKELY(!edge_offsets[jj])) {
        NPY_END_THREADS;
        PyErr_NoMemory();
        goto exit;
      }
      for (hh = 0; hh < odimensions[jj]; hh++) {
        edge_offsets[jj][hh] = NULL;
      }
    }
  }

  int spline_mode = _get_spline_boundary_mode(mode);

  for (jj = 0; jj < rank; jj++) {
    double shift = 0.0, zoom = 0.0;
    if (shifts)
      shift = shifts[jj];
    if (zooms)
      zoom = zooms[jj];
    for (kk = 0; kk < odimensions[jj]; kk++) {
      double cc = (double)kk;
      if (shifts)
        cc += shift;
      if (zooms) {
        if (grid_mode) {
          cc += 0.5;
          cc *= zoom;
          cc -= 0.5;
        } else {
          cc *= zoom;
        }
      }
      cc += (double)nprepad;
      if ((mode != NI_EXTEND_GRID_CONSTANT) && (mode != NI_EXTEND_NEAREST)) {
        /* if the input coordinate is outside the borders, map it: */
        cc = map_coordinate(cc, idimensions[jj], mode);
      }
      if (cc > -1.0 || mode == NI_EXTEND_GRID_CONSTANT ||
          mode == NI_EXTEND_NEAREST) {
        npy_intp start;
        if (zeros && zeros[jj])
          zeros[jj][kk] = 0;
        if (order & 1) {
          start = (npy_intp)floor(cc) - order / 2;
        } else {
          start = (npy_intp)floor(cc + 0.5) - order / 2;
        }
        offsets[jj][kk] = istrides[jj] * start;
        if (start < 0 || start + order >= idimensions[jj]) {
          npy_intp idx = 0;

          if (mode == NI_EXTEND_GRID_CONSTANT) {
            edge_grid_const[jj][kk] = malloc((order + 1) * sizeof(char));
            if (NPY_UNLIKELY(!edge_grid_const[jj][kk])) {
              NPY_END_THREADS;
              PyErr_NoMemory();
              goto exit;
            }
            for (hh = 0; hh <= order; hh++) {
              idx = start + hh;
              edge_grid_const[jj][kk][hh] = (idx < 0 || idx >= idimensions[jj]);
            }
          } else {
            edge_offsets[jj][kk] = malloc((order + 1) * sizeof(npy_intp));
            if (NPY_UNLIKELY(!edge_offsets[jj][kk])) {
              NPY_END_THREADS;
              PyErr_NoMemory();
              goto exit;
            }
            for (hh = 0; hh <= order; hh++) {
              idx = start + hh;
              idx = (npy_intp)map_coordinate(idx, idimensions[jj], spline_mode);
              edge_offsets[jj][kk][hh] = istrides[jj] * (idx - start);
            }
          }
        }
        if (order > 0) {
          splvals[jj][kk] = malloc((order + 1) * sizeof(double));
          if (NPY_UNLIKELY(!splvals[jj][kk])) {
            NPY_END_THREADS;
            PyErr_NoMemory();
            goto exit;
          }
          get_spline_interpolation_weights(cc, order, splvals[jj][kk]);
        }
      } else {
        zeros[jj][kk] = 1;
      }
    }
  }

  filter_size = 1;
  for (jj = 0; jj < rank; jj++)
    filter_size *= order + 1;

  if (!NI_InitPointIterator(output, &io))
    goto exit;

  pi = (void *)PyArray_DATA(input);
  po = (void *)PyArray_DATA(output);

  /* store all coordinates and offsets with filter: */
  fcoordinates = malloc(rank * filter_size * sizeof(npy_intp));
  foffsets = malloc(filter_size * sizeof(npy_intp));
  if (NPY_UNLIKELY(!fcoordinates || !foffsets)) {
    NPY_END_THREADS;
    PyErr_NoMemory();
    goto exit;
  }

  for (jj = 0; jj < rank; jj++)
    ftmp[jj] = 0;
  kk = 0;
  for (hh = 0; hh < filter_size; hh++) {
    for (jj = 0; jj < rank; jj++)
      fcoordinates[jj + hh * rank] = ftmp[jj];
    foffsets[hh] = kk;
    for (jj = rank - 1; jj >= 0; jj--) {
      if (ftmp[jj] < order) {
        ftmp[jj]++;
        kk += istrides[jj];
        break;
      } else {
        ftmp[jj] = 0;
        kk -= istrides[jj] * order;
      }
    }
  }
  size = PyArray_SIZE(output);
  for (kk = 0; kk < size; kk++) {
    double t = 0.0;
    npy_intp edge = 0, oo = 0, zero = 0;

    for (hh = 0; hh < rank; hh++) {
      if (zeros && zeros[hh][io.coordinates[hh]]) {
        /* we use constant border condition */
        zero = 1;
        break;
      }
      oo += offsets[hh][io.coordinates[hh]];
      if (mode != NI_EXTEND_GRID_CONSTANT) {
        if (edge_offsets[hh][io.coordinates[hh]])
          edge = 1;
      }
    }

    if (!zero) {
      npy_intp *ff = fcoordinates;
      const int type_num = PyArray_TYPE(input);
      t = 0.0;
      for (hh = 0; hh < filter_size; hh++) {
        npy_intp idx = 0;
        double coeff = 0.0;
        int is_cval = 0;
        if (mode == NI_EXTEND_GRID_CONSTANT) {
          for (jj = 0; jj < rank; jj++) {
            if (edge_grid_const[jj][io.coordinates[jj]]) {
              if (edge_grid_const[jj][io.coordinates[jj]][ff[jj]])
                is_cval = 1;
            }
          }
        }
        if (is_cval) {
          coeff = cval;
        } else {
          if (NPY_UNLIKELY(edge)) {
            /* use precalculated edge offsets: */
            for (jj = 0; jj < rank; jj++) {
              if (edge_offsets[jj][io.coordinates[jj]])
                idx += edge_offsets[jj][io.coordinates[jj]][ff[jj]];
              else
                idx += ff[jj] * istrides[jj];
            }
            idx += oo;
          } else {
            /* use normal offsets: */
            idx += oo + foffsets[hh];
          }
          switch (type_num) {
            CASE_INTERP_COEFF(NPY_BOOL, npy_bool, coeff, pi, idx);
            CASE_INTERP_COEFF(NPY_UBYTE, npy_ubyte, coeff, pi, idx);
            CASE_INTERP_COEFF(NPY_USHORT, npy_ushort, coeff, pi, idx);
            CASE_INTERP_COEFF(NPY_UINT, npy_uint, coeff, pi, idx);
            CASE_INTERP_COEFF(NPY_ULONG, npy_ulong, coeff, pi, idx);
            CASE_INTERP_COEFF(NPY_ULONGLONG, npy_ulonglong, coeff, pi, idx);
            CASE_INTERP_COEFF(NPY_BYTE, npy_byte, coeff, pi, idx);
            CASE_INTERP_COEFF(NPY_SHORT, npy_short, coeff, pi, idx);
            CASE_INTERP_COEFF(NPY_INT, npy_int, coeff, pi, idx);
            CASE_INTERP_COEFF(NPY_LONG, npy_long, coeff, pi, idx);
            CASE_INTERP_COEFF(NPY_LONGLONG, npy_longlong, coeff, pi, idx);
            CASE_INTERP_COEFF(NPY_FLOAT, npy_float, coeff, pi, idx);
            CASE_INTERP_COEFF(NPY_DOUBLE, npy_double, coeff, pi, idx);
          default:
            NPY_END_THREADS;
            PyErr_SetString(PyExc_RuntimeError, "data type not supported");
            goto exit;
          }
        }
        /* calculate interpolated value: */
        for (jj = 0; jj < rank; jj++)
          if (order > 0)
            coeff *= splvals[jj][io.coordinates[jj]][ff[jj]];
        t += coeff;
        ff += rank;
      }
    } else {
      t = cval;
    }
    /* store output: */
    switch (PyArray_TYPE(output)) {
      CASE_INTERP_OUT(NPY_BOOL, npy_bool, po, t);
      CASE_INTERP_OUT_UINT(UBYTE, npy_ubyte, po, t);
      CASE_INTERP_OUT_UINT(USHORT, npy_ushort, po, t);
      CASE_INTERP_OUT_UINT(UINT, npy_uint, po, t);
      CASE_INTERP_OUT_UINT(ULONG, npy_ulong, po, t);
      CASE_INTERP_OUT_UINT(ULONGLONG, npy_ulonglong, po, t);
      CASE_INTERP_OUT_INT(BYTE, npy_byte, po, t);
      CASE_INTERP_OUT_INT(SHORT, npy_short, po, t);
      CASE_INTERP_OUT_INT(INT, npy_int, po, t);
      CASE_INTERP_OUT_INT(LONG, npy_long, po, t);
      CASE_INTERP_OUT_INT(LONGLONG, npy_longlong, po, t);
      CASE_INTERP_OUT(NPY_FLOAT, npy_float, po, t);
      CASE_INTERP_OUT(NPY_DOUBLE, npy_double, po, t);
    default:
      NPY_END_THREADS;
      PyErr_SetString(PyExc_RuntimeError, "data type not supported");
      goto exit;
    }
    NI_ITERATOR_NEXT(io, po);
  }

exit:
  NPY_END_THREADS;
  if (zeros) {
    for (jj = 0; jj < rank; jj++)
      free(zeros[jj]);
    free(zeros);
  }
  if (offsets) {
    for (jj = 0; jj < rank; jj++)
      free(offsets[jj]);
    free(offsets);
  }
  if (splvals) {
    for (jj = 0; jj < rank; jj++) {
      if (splvals[jj]) {
        for (hh = 0; hh < odimensions[jj]; hh++)
          free(splvals[jj][hh]);
        free(splvals[jj]);
      }
    }
    free(splvals);
  }
  if (edge_offsets) {
    for (jj = 0; jj < rank; jj++) {
      if (edge_offsets[jj]) {
        for (hh = 0; hh < odimensions[jj]; hh++)
          free(edge_offsets[jj][hh]);
        free(edge_offsets[jj]);
      }
    }
    free(edge_offsets);
  }
  if (edge_grid_const) {
    for (jj = 0; jj < rank; jj++) {
      if (edge_grid_const[jj]) {
        for (hh = 0; hh < odimensions[jj]; hh++)
          free(edge_grid_const[jj][hh]);
        free(edge_grid_const[jj]);
      }
    }
    free(edge_grid_const);
  }
  free(foffsets);
  free(fcoordinates);
  return PyErr_Occurred() ? 0 : 1;
}
