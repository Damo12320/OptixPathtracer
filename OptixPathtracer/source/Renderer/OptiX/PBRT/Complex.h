#include "../glmCUDA.h"

namespace PBRT {
	namespace Complex {
        struct complex {
            __device__ __host__ complex(float re) : re(re), im(0) {}
            __device__ __host__ complex(float re, float im) : re(re), im(im) {}

            __device__ __host__ complex operator-() const { return { -re, -im }; }

            __device__ __host__ complex operator+(complex z) const { return { re + z.re, im + z.im }; }

            __device__ __host__ complex operator-(complex z) const { return { re - z.re, im - z.im }; }

            __device__ __host__ complex operator*(complex z) const {
                return { re * z.re - im * z.im, re * z.im + im * z.re };
            }

            __device__ __host__ complex operator/(complex z) const {
                float scale = 1 / (z.re * z.re + z.im * z.im);
                return { scale * (re * z.re + im * z.im), scale * (im * z.re - re * z.im) };
            }

            friend __device__ __host__ complex operator+(float value, complex z) {
                return complex(value) + z;
            }

            friend __device__ __host__ complex operator-(float value, complex z) {
                return complex(value) - z;
            }

            friend __device__ __host__ complex operator*(float value, complex z) {
                return complex(value) * z;
            }

            friend __device__ __host__ complex operator/(float value, complex z) {
                return complex(value) / z;
            }

            float re, im;
        };


        __device__ __host__ float norm(const complex& z) {
            return z.re * z.re + z.im * z.im;
        }

        __device__ __host__ float abs(const complex& z) {
            return glm::sqrt(norm(z));
        }

        __device__ __host__ complex sqrt(const complex& z) {
            float n = abs(z), t1 = glm::sqrt(0.5f * (n + glm::abs(z.re))),
                t2 = 0.5f * z.im / t1;

            if (n == 0.0f)
                return 0;

            if (z.re >= 0.0f)
                return { t1, t2 };
            else
                return { glm::abs(t2), copysignf(t1, z.im) };
        }
	}
}