/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TRT_EFFICIENT_NMS_INFERENCE_CUH
#define TRT_EFFICIENT_NMS_INFERENCE_CUH

#include <cuda_fp16.h>
#include <cassert>

// FP32 Intrinsics

float __device__ __inline__ exp_mp(const float a)
{
    return __expf(a);
}
float __device__ __inline__ sigmoid_mp(const float a)
{
    return __frcp_rn(__fadd_rn(1.f, __expf(-a)));
}
float __device__ __inline__ add_mp(const float a, const float b)
{
    return __fadd_rn(a, b);
}
float __device__ __inline__ sub_mp(const float a, const float b)
{
    return __fsub_rn(a, b);
}
float __device__ __inline__ mul_mp(const float a, const float b)
{
    return __fmul_rn(a, b);
}
bool __device__ __inline__ gt_mp(const float a, const float b)
{
    return a > b;
}
bool __device__ __inline__ lt_mp(const float a, const float b)
{
    return a < b;
}
bool __device__ __inline__ lte_mp(const float a, const float b)
{
    return a <= b;
}
bool __device__ __inline__ gte_mp(const float a, const float b)
{
    return a >= b;
}

#if __CUDA_ARCH__ >= 530

// FP16 Intrinsics

__half __device__ __inline__ exp_mp(const __half a)
{
    return hexp(a);
}
__half __device__ __inline__ sigmoid_mp(const __half a)
{
    return hrcp(__hadd((__half) 1, hexp(__hneg(a))));
}
__half __device__ __inline__ add_mp(const __half a, const __half b)
{
    return __hadd(a, b);
}
__half __device__ __inline__ sub_mp(const __half a, const __half b)
{
    return __hsub(a, b);
}
__half __device__ __inline__ mul_mp(const __half a, const __half b)
{
    return __hmul(a, b);
}
bool __device__ __inline__ gt_mp(const __half a, const __half b)
{
    return __hgt(a, b);
}
bool __device__ __inline__ lt_mp(const __half a, const __half b)
{
    return __hlt(a, b);
}
bool __device__ __inline__ lte_mp(const __half a, const __half b)
{
    return __hle(a, b);
}
bool __device__ __inline__ gte_mp(const __half a, const __half b)
{
    return __hge(a, b);
}

#else

// FP16 Fallbacks on older architectures that lack support

__half __device__ __inline__ exp_mp(const __half a)
{
    return __float2half(exp_mp(__half2float(a)));
}
__half __device__ __inline__ sigmoid_mp(const __half a)
{
    return __float2half(sigmoid_mp(__half2float(a)));
}
__half __device__ __inline__ add_mp(const __half a, const __half b)
{
    return __float2half(add_mp(__half2float(a), __half2float(b)));
}
__half __device__ __inline__ sub_mp(const __half a, const __half b)
{
    return __float2half(sub_mp(__half2float(a), __half2float(b)));
}
__half __device__ __inline__ mul_mp(const __half a, const __half b)
{
    return __float2half(mul_mp(__half2float(a), __half2float(b)));
}
bool __device__ __inline__ gt_mp(const __half a, const __half b)
{
    return __float2half(gt_mp(__half2float(a), __half2float(b)));
}
bool __device__ __inline__ lt_mp(const __half a, const __half b)
{
    return __float2half(lt_mp(__half2float(a), __half2float(b)));
}
bool __device__ __inline__ lte_mp(const __half a, const __half b)
{
    return __float2half(lte_mp(__half2float(a), __half2float(b)));
}
bool __device__ __inline__ gte_mp(const __half a, const __half b)
{
    return __float2half(gte_mp(__half2float(a), __half2float(b)));
}

#endif

#ifdef __CUDACC__
// Designates functions callable from the host (CPU) and the device (GPU)
#define HOST_DEVICE __host__ __device__
#define HOST_DEVICE_INLINE HOST_DEVICE __forceinline__
#else
#include <algorithm>
#define HOST_DEVICE
#define HOST_DEVICE_INLINE HOST_DEVICE inline
#endif


template <typename T>
struct __align__(1 * sizeof(T)) BoxCenterSize;

template <typename T>
struct Point {
  T x, y;
  HOST_DEVICE_INLINE Point(const T& px = 0, const T& py = 0) : x(px), y(py) {}
  HOST_DEVICE_INLINE Point operator+(const Point& p) const {
    return Point(x + p.x, y + p.y);
  }
  HOST_DEVICE_INLINE Point& operator+=(const Point& p) {
    x += p.x;
    y += p.y;
    return *this;
  }
  HOST_DEVICE_INLINE Point operator-(const Point& p) const {
    return Point(x - p.x, y - p.y);
  }
  HOST_DEVICE_INLINE Point operator*(const T coeff) const {
    return Point(x * coeff, y * coeff);
  }
};

template <typename T>
HOST_DEVICE_INLINE T dot_2d(const Point<T>& A, const Point<T>& B) {
  return A.x * B.x + A.y * B.y;
}

template <typename T>
HOST_DEVICE_INLINE T cross_2d(const Point<T>& A, const Point<T>& B) {
  return A.x * B.y - B.x * A.y;
}

template <typename T>
HOST_DEVICE_INLINE void get_rotated_vertices(const BoxCenterSize<T>& box,
                                             Point<T> (&pts)[4]) {
  // M_PI / 180. == 0.01745329251
  // double theta = box.a * 0.01745329251;
  // MODIFIED
  double theta = box.theta;
  T cosTheta2 = (T)cos(theta) * T(0.5f);
  T sinTheta2 = (T)sin(theta) * T(0.5f);

  // y: top --> down; x: left --> right
  pts[0].x = box.x - sinTheta2 * box.h - cosTheta2 * box.w;
  pts[0].y = box.y + cosTheta2 * box.h - sinTheta2 * box.w;
  pts[1].x = box.x + sinTheta2 * box.h - cosTheta2 * box.w;
  pts[1].y = box.y - cosTheta2 * box.h - sinTheta2 * box.w;
  pts[2].x = T(2) * box.x - pts[0].x;
  pts[2].y = T(2) * box.y - pts[0].y;
  pts[3].x = T(2) * box.x - pts[1].x;
  pts[3].y = T(2) * box.y - pts[1].y;
}

template <typename T>
HOST_DEVICE_INLINE int get_intersection_points(const Point<T> (&pts1)[4],
                                               const Point<T> (&pts2)[4],
                                               Point<T> (&intersections)[24]) {
  // Line vector
  // A line from p1 to p2 is: p1 + (p2-p1)*t, t=[0,1]
  Point<T> vec1[4], vec2[4];
  for (int i = 0; i < 4; i++) {
    vec1[i] = pts1[(i + 1) % 4] - pts1[i];
    vec2[i] = pts2[(i + 1) % 4] - pts2[i];
  }

  // Line test - test all line combos for intersection
  int num = 0;  // number of intersections
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      // Solve for 2x2 Ax=b
      float det = cross_2d<T>(vec2[j], vec1[i]);

      // This takes care of parallel lines
      if (fabs(det) <= 1e-14) {
        continue;
      }

      auto vec12 = pts2[j] - pts1[i];

      T t1 = cross_2d<T>(vec2[j], vec12) / (T)det;
      T t2 = cross_2d<T>(vec1[i], vec12) / (T)det;

      if (t1 >= T(0.0f) && t1 <= T(1.0f) && t2 >= T(0.0f) && t2 <= T(1.0f)) {
        intersections[num++] = pts1[i] + vec1[i] * t1;
      }
    }
  }

  // Check for vertices of rect1 inside rect2
  {
    const auto& AB = vec2[0];
    const auto& DA = vec2[3];
    auto ABdotAB = dot_2d<T>(AB, AB);
    auto ADdotAD = dot_2d<T>(DA, DA);
    for (int i = 0; i < 4; i++) {
      // assume ABCD is the rectangle, and P is the point to be judged
      // P is inside ABCD iff. P's projection on AB lies within AB
      // and P's projection on AD lies within AD

      auto AP = pts1[i] - pts2[0];

      auto APdotAB = dot_2d<T>(AP, AB);
      auto APdotAD = -dot_2d<T>(AP, DA);

      if ((APdotAB >= T(0)) && (APdotAD >= T(0)) && (APdotAB <= ABdotAB) &&
          (APdotAD <= ADdotAD)) {
        intersections[num++] = pts1[i];
      }
    }
  }

  // Reverse the check - check for vertices of rect2 inside rect1
  {
    const auto& AB = vec1[0];
    const auto& DA = vec1[3];
    auto ABdotAB = dot_2d<T>(AB, AB);
    auto ADdotAD = dot_2d<T>(DA, DA);
    for (int i = 0; i < 4; i++) {
      auto AP = pts2[i] - pts1[0];

      auto APdotAB = dot_2d<T>(AP, AB);
      auto APdotAD = -dot_2d<T>(AP, DA);

      if ((APdotAB >= T(0)) && (APdotAD >= T(0)) && (APdotAB <= ABdotAB) &&
          (APdotAD <= ADdotAD)) {
        intersections[num++] = pts2[i];
      }
    }
  }

  return num;
}

template <typename T>
HOST_DEVICE_INLINE int convex_hull_graham(const Point<T> (&p)[24],
                                          const int& num_in, Point<T> (&q)[24],
                                          bool shift_to_zero = false) {
  assert(num_in >= 2);

  // Step 1:
  // Find point with minimum y
  // if more than 1 points have the same minimum y,
  // pick the one with the minimum x.
  int t = 0;
  for (int i = 1; i < num_in; i++) {
    if (p[i].y < p[t].y || (p[i].y == p[t].y && p[i].x < p[t].x)) {
      t = i;
    }
  }
  auto& start = p[t];  // starting point

  // Step 2:
  // Subtract starting point from every points (for sorting in the next step)
  for (int i = 0; i < num_in; i++) {
    q[i] = p[i] - start;
  }

  // Swap the starting point to position 0
  auto tmp = q[0];
  q[0] = q[t];
  q[t] = tmp;

  // Step 3:
  // Sort point 1 ~ num_in according to their relative cross-product values
  // (essentially sorting according to angles)
  // If the angles are the same, sort according to their distance to origin
  T dist[24];
  for (int i = 0; i < num_in; i++) {
    dist[i] = dot_2d<T>(q[i], q[i]);
  }

#ifdef __CUDACC__
  // CUDA version
  // In the future, we can potentially use thrust
  // for sorting here to improve speed (though not guaranteed)
  for (int i = 1; i < num_in - 1; i++) {
    for (int j = i + 1; j < num_in; j++) {
      T crossProduct = cross_2d<T>(q[i], q[j]);
      if ((crossProduct < T(-1e-6)) ||
          (fabs((float)crossProduct) < 1e-6 && dist[i] > dist[j])) {
        auto q_tmp = q[i];
        q[i] = q[j];
        q[j] = q_tmp;
        auto dist_tmp = dist[i];
        dist[i] = dist[j];
        dist[j] = dist_tmp;
      }
    }
  }
#else
  // CPU version
  std::sort(q + 1, q + num_in,
            [](const Point<T>& A, const Point<T>& B) -> bool {
              T temp = cross_2d<T>(A, B);
              if (fabs(temp) < 1e-6) {
                return dot_2d<T>(A, A) < dot_2d<T>(B, B);
              } else {
                return temp > 0;
              }
            });
  // compute distance to origin after sort, since the points are now different.
  for (int i = 0; i < num_in; i++) {
    dist[i] = dot_2d<T>(q[i], q[i]);
  }
#endif

  // Step 4:
  // Make sure there are at least 2 points (that don't overlap with each other)
  // in the stack
  int k;  // index of the non-overlapped second point
  for (k = 1; k < num_in; k++) {
    if (dist[k] > T(1e-8)) {
      break;
    }
  }
  if (k == num_in) {
    // We reach the end, which means the convex hull is just one point
    q[0] = p[t];
    return 1;
  }
  q[1] = q[k];
  int m = 2;  // 2 points in the stack
  // Step 5:
  // Finally we can start the scanning process.
  // When a non-convex relationship between the 3 points is found
  // (either concave shape or duplicated points),
  // we pop the previous point from the stack
  // until the 3-point relationship is convex again, or
  // until the stack only contains two points
  for (int i = k + 1; i < num_in; i++) {
    while (m > 1 && cross_2d<T>(q[i] - q[m - 2], q[m - 1] - q[m - 2]) >= T(0)) {
      m--;
    }
    q[m++] = q[i];
  }

  // Step 6 (Optional):
  // In general sense we need the original coordinates, so we
  // need to shift the points back (reverting Step 2)
  // But if we're only interested in getting the area/perimeter of the shape
  // We can simply return.
  if (!shift_to_zero) {
    for (int i = 0; i < m; i++) {
      q[i] += start;
    }
  }

  return m;
}

template <typename T>
HOST_DEVICE_INLINE T quadri_box_area(const Point<T> (&q)[4]) {
  T area = 0;
#pragma unroll
  for (int i = 1; i < 3; i++) {
    area += fabs(cross_2d<T>(q[i] - q[0], q[i + 1] - q[0]));
  }

  return area / 2.0;
}

template <typename T>
HOST_DEVICE_INLINE T polygon_area(const Point<T> (&q)[24], const int& m) {
  if (m <= 2) {
    return 0;
  }

  T area = 0;
  for (int i = 1; i < m - 1; i++) {
    area += fabs((float)cross_2d<T>(q[i] - q[0], q[i + 1] - q[0]));
  }

  return area / T(2.0);
}



template <typename T>
HOST_DEVICE_INLINE T rotated_boxes_intersection(const BoxCenterSize<T>& box1,
                                                const BoxCenterSize<T>& box2) {
  // There are up to 4 x 4 + 4 + 4 = 24 intersections (including dups) returned
  // from rotated_rect_intersection_pts
  Point<T> intersectPts[24], orderedPts[24];

  Point<T> pts1[4];
  Point<T> pts2[4];
  get_rotated_vertices<T>(box1, pts1);
  get_rotated_vertices<T>(box2, pts2);

  int num = get_intersection_points<T>(pts1, pts2, intersectPts);

  if (num <= 2) {
    return 0.0;
  }

  // Convex Hull to order the intersection points in clockwise order and find
  // the contour area.
  int num_convex = convex_hull_graham<T>(intersectPts, num, orderedPts, true);
  return polygon_area<T>(orderedPts, num_convex);
}



template <typename T>
struct __align__(1 * sizeof(T)) BoxCenterSize
{
    // For NMS/IOU purposes, YXHW coding is identical to XYWH
    T x, y, w, h, theta;

    __device__ void reorder() {}

    __device__ BoxCenterSize<T> clip(T low, T high) const
    {
        return {lt_mp(x, low) ? low : (gt_mp(x, high) ? high : x),
            lt_mp(y, low) ? low : (gt_mp(y, high) ? high : y), lt_mp(w, low) ? low : (gt_mp(w, high) ? high : w),
            lt_mp(h, low) ? low : (gt_mp(h, high) ? high : h)};
    }

    __device__ BoxCenterSize<T> decode(BoxCenterSize<T> anchor) const
    {
        return {add_mp(mul_mp(y, anchor.h), anchor.y), add_mp(mul_mp(x, anchor.w), anchor.x),
            mul_mp(anchor.h, exp_mp(h)), mul_mp(anchor.w, exp_mp(w)), theta};
    }

    __device__ float area() const
    {
        if (h <= (T) 0)
        {
            return 0;
        }
        if (w <= (T) 0)
        {
            return 0;
        }
        return (float) h * (float) w;
    }

    __device__ static inline float intersect_area(const BoxCenterSize<T>& a, const BoxCenterSize<T>& b)
    {
        // copy code from mmcv

        // shift center to the middle point to achieve higher precision in result
        BoxCenterSize<T> box1(a), box2(b);
        auto center_shift_x = (box1.x + box2.x) / T(2.0);
        auto center_shift_y = (box1.y + box2.y) / T(2.0);
        box1.x -= center_shift_x;
        box1.y -= center_shift_y;

        box2.x -= center_shift_x;
        box2.y -= center_shift_y;

        const T area1 = box1.w * box1.h;
        const T area2 = box2.w * box2.h;
        if (area1 < T(1e-14) || area2 < T(1e-14)) {
            return 0.f;
        }

        const T intersection_area = rotated_boxes_intersection<T>(box1, box2);
        return float(intersection_area);
    }
};

#endif