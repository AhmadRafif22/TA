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
                    float kernels[44] = {0};
                    float decisions[1] = {0};
                    int votes[2] = {0};
                    kernels[0] = compute_kernel(x, -59.0, -77.0, -81.0, -75.0, 0.0);
                    kernels[1] = compute_kernel(x, -54.0, -87.0, -78.0, 0.0, -86.0);
                    kernels[2] = compute_kernel(x, -44.0, -67.0, 0.0, -78.0, 0.0);
                    kernels[3] = compute_kernel(x, -62.0, -73.0, 0.0, 0.0, 0.0);
                    kernels[4] = compute_kernel(x, -61.0, -68.0, -74.0, -76.0, 0.0);
                    kernels[5] = compute_kernel(x, -67.0, -85.0, 0.0, -78.0, -75.0);
                    kernels[6] = compute_kernel(x, -66.0, -85.0, 0.0, 0.0, 0.0);
                    kernels[7] = compute_kernel(x, -44.0, -79.0, 0.0, 0.0, -76.0);
                    kernels[8] = compute_kernel(x, -57.0, -73.0, -80.0, 0.0, -76.0);
                    kernels[9] = compute_kernel(x, -66.0, -72.0, -83.0, -82.0, -73.0);
                    kernels[10] = compute_kernel(x, -49.0, -84.0, -70.0, -79.0, -75.0);
                    kernels[11] = compute_kernel(x, -60.0, -71.0, 0.0, -76.0, 0.0);
                    kernels[12] = compute_kernel(x, -59.0, -74.0, 0.0, 0.0, 0.0);
                    kernels[13] = compute_kernel(x, -60.0, -83.0, -71.0, -71.0, 0.0);
                    kernels[14] = compute_kernel(x, -63.0, -90.0, -71.0, 0.0, -80.0);
                    kernels[15] = compute_kernel(x, -54.0, -83.0, -78.0, 0.0, 0.0);
                    kernels[16] = compute_kernel(x, -59.0, -80.0, 0.0, -79.0, -81.0);
                    kernels[17] = compute_kernel(x, -70.0, -85.0, 0.0, -75.0, 0.0);
                    kernels[18] = compute_kernel(x, -51.0, -81.0, -71.0, 0.0, -74.0);
                    kernels[19] = compute_kernel(x, -53.0, -76.0, -75.0, -73.0, -82.0);
                    kernels[20] = compute_kernel(x, -71.0, -85.0, -67.0, 0.0, 0.0);
                    kernels[21] = compute_kernel(x, -60.0, -70.0, -73.0, 0.0, 0.0);
                    kernels[22] = compute_kernel(x, -53.0, -74.0, -67.0, -77.0, 0.0);
                    kernels[23] = compute_kernel(x, -54.0, -89.0, -74.0, -72.0, -72.0);
                    kernels[24] = compute_kernel(x, -58.0, -89.0, 0.0, 0.0, -75.0);
                    kernels[25] = compute_kernel(x, 0.0, -56.0, -83.0, 0.0, -61.0);
                    kernels[26] = compute_kernel(x, -76.0, -66.0, 0.0, -74.0, 0.0);
                    kernels[27] = compute_kernel(x, -77.0, -65.0, 0.0, -73.0, -49.0);
                    kernels[28] = compute_kernel(x, -76.0, -56.0, -85.0, -82.0, -60.0);
                    kernels[29] = compute_kernel(x, 0.0, -88.0, 0.0, -72.0, -67.0);
                    kernels[30] = compute_kernel(x, -77.0, -65.0, 0.0, 0.0, -74.0);
                    kernels[31] = compute_kernel(x, -76.0, -59.0, 0.0, 0.0, 0.0);
                    kernels[32] = compute_kernel(x, 0.0, -55.0, 0.0, -77.0, -66.0);
                    kernels[33] = compute_kernel(x, -77.0, -64.0, 0.0, -71.0, -62.0);
                    kernels[34] = compute_kernel(x, -76.0, -76.0, 0.0, 0.0, 0.0);
                    kernels[35] = compute_kernel(x, 0.0, -69.0, 0.0, 0.0, -64.0);
                    kernels[36] = compute_kernel(x, -73.0, -57.0, 0.0, -73.0, 0.0);
                    kernels[37] = compute_kernel(x, -77.0, -59.0, 0.0, -72.0, -65.0);
                    kernels[38] = compute_kernel(x, -80.0, -73.0, 0.0, 0.0, -68.0);
                    kernels[39] = compute_kernel(x, 0.0, -60.0, 0.0, -68.0, -54.0);
                    kernels[40] = compute_kernel(x, -77.0, -56.0, 0.0, 0.0, -48.0);
                    kernels[41] = compute_kernel(x, 0.0, -65.0, 0.0, -84.0, -51.0);
                    kernels[42] = compute_kernel(x, -75.0, -58.0, 0.0, 0.0, 0.0);
                    kernels[43] = compute_kernel(x, 0.0, -58.0, 0.0, 0.0, -59.0);
                    float decision = -0.086415241781;
                    decision = decision - (+kernels[0] * -0.220016381087 + kernels[1] * -0.180020293937 + kernels[2] * -0.537153813409 + kernels[3] * -1.0 + kernels[4] * -0.319992401911 + kernels[5] * -1.0 + kernels[6] * -0.966724653227 + kernels[7] * -0.730766453949 + kernels[8] * -0.446243560034 + kernels[9] * -1.0 + kernels[10] * -0.204533753072 + kernels[11] * -1.0 + kernels[12] * -1.0 + kernels[13] * -0.286454976704 + kernels[14] * -0.353799253209 + kernels[15] * -0.358066127632 + kernels[16] * -0.659692643664 + kernels[17] * -1.0 + kernels[18] * -0.145248484026 + kernels[19] * -0.073410161561 + kernels[20] * -0.452888391233 + kernels[21] * -0.311365375949 + kernels[22] * -0.216623526505 + kernels[23] * -0.267139347629 + kernels[24] * -1.0);
                    decision = decision - (+kernels[25] * 1.0 + kernels[26] * 1.0 + kernels[27] * 0.205275036552 + kernels[28] * 1.0 + kernels[29] * 0.748337449748 + kernels[30] * 0.644196467771 + kernels[31] * 1.0 + kernels[32] * 0.475947871127 + kernels[33] * 1.0 + kernels[34] * 1.0 + kernels[35] * 0.772437481145 + kernels[36] * 1.0 + kernels[37] * 0.577311767115 + kernels[38] * 0.951598650801 + kernels[39] * 0.169111144264 + kernels[40] * 0.364025755483 + kernels[41] * 0.349894356391 + kernels[42] * 1.0 + kernels[43] * 0.47200361834);

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