#pragma once
#include <cstdarg>
#include <stdint.h>
#include <math.h>

namespace Eloquent
{
    namespace ML
    {
        namespace Port
        {
            class SVM
            {
            public:
                /**
                 * Predict class for features vector
                 */
                int predict(float *x)
                {
                    float kernels[23] = {0};
                    float decisions[1] = {0};
                    int votes[2] = {0};
                    kernels[0] = compute_kernel(x, -57.0, -75.0, 0.0, -78.0, 0.0);
                    kernels[1] = compute_kernel(x, -58.0, -67.0, -76.0, -82.0, 0.0);
                    kernels[2] = compute_kernel(x, -64.0, -90.0, -74.0, 0.0, 0.0);
                    kernels[3] = compute_kernel(x, -52.0, -71.0, -70.0, 0.0, 0.0);
                    kernels[4] = compute_kernel(x, -56.0, -85.0, -79.0, -84.0, -80.0);
                    kernels[5] = compute_kernel(x, -49.0, -86.0, -71.0, 0.0, 0.0);
                    kernels[6] = compute_kernel(x, -61.0, -79.0, -74.0, -81.0, 0.0);
                    kernels[7] = compute_kernel(x, -49.0, -79.0, -86.0, 0.0, -94.0);
                    kernels[8] = compute_kernel(x, -67.0, -84.0, -80.0, 0.0, -84.0);
                    kernels[9] = compute_kernel(x, -51.0, -85.0, -75.0, -74.0, 0.0);
                    kernels[10] = compute_kernel(x, -57.0, -57.0, -72.0, 0.0, -92.0);
                    kernels[11] = compute_kernel(x, -56.0, -79.0, -79.0, -78.0, -92.0);
                    kernels[12] = compute_kernel(x, -66.0, -66.0, 0.0, 0.0, 0.0);
                    kernels[13] = compute_kernel(x, -65.0, -70.0, -77.0, -75.0, -86.0);
                    kernels[14] = compute_kernel(x, -80.0, -77.0, 0.0, -71.0, 0.0);
                    kernels[15] = compute_kernel(x, -78.0, -70.0, 0.0, 0.0, 0.0);
                    kernels[16] = compute_kernel(x, -72.0, -52.0, 0.0, -76.0, 0.0);
                    kernels[17] = compute_kernel(x, -75.0, -59.0, 0.0, -82.0, 0.0);
                    kernels[18] = compute_kernel(x, -89.0, -78.0, 0.0, -72.0, 0.0);
                    kernels[19] = compute_kernel(x, -77.0, -63.0, 0.0, 0.0, 0.0);
                    kernels[20] = compute_kernel(x, -84.0, -67.0, -85.0, -75.0, 0.0);
                    kernels[21] = compute_kernel(x, -74.0, -51.0, 0.0, 0.0, 0.0);
                    kernels[22] = compute_kernel(x, -73.0, -53.0, 0.0, -79.0, 0.0);
                    float decision = -0.329285234322;
                    decision = decision - (+kernels[0] * -1.0 + kernels[1] * -0.643258122406 + kernels[2] * -0.402853388823 + kernels[3] * -0.402309680002 + kernels[4] * -0.370793179167 + kernels[5] * -0.038936322785 + kernels[6] * -0.533310665603 + kernels[7] * -0.257843649392 + kernels[8] * -0.351545843166 + kernels[9] * -0.02872194168 + kernels[10] * -0.412314060885 + kernels[11] * -0.073010071996 + kernels[12] * -1.0 + kernels[13] * -0.36645508827);
                    decision = decision - (+kernels[14] * 1.0 + kernels[15] * 1.0 + kernels[16] * 1.0 + kernels[17] * 0.059786278266 + kernels[18] * 0.285214274735 + kernels[19] * 0.391960240042 + kernels[20] * 1.0 + kernels[21] * 1.0 + kernels[22] * 0.144391221132);

                    return decision > 0 ? 0 : 1;
                }

                /**
                 * Predict readable class name
                 */
                const char *predictLabel(float *x)
                {
                    return idxToLabel(predict(x));
                }

                /**
                 * Convert class idx to readable name
                 */
                const char *idxToLabel(uint8_t classIdx)
                {
                    switch (classIdx)
                    {
                    case 0:
                        return "LIG 2";
                    case 1:
                        return "LPY 4";
                    case 2:
                        return "JALAN";
                    default:
                        return "Houston we have a problem";
                    }
                }

            protected:
                /**
                 * Compute kernel between feature vector and support vector.
                 * Kernel type: rbf
                 */
                float compute_kernel(float *x, ...)
                {
                    va_list w;
                    va_start(w, 5);
                    float kernel = 0.0;

                    for (uint16_t i = 0; i < 5; i++)
                    {
                        kernel += pow(x[i] - va_arg(w, double), 2);
                    }

                    return exp(-0.001 * kernel);
                }
            };
        }
    }
}