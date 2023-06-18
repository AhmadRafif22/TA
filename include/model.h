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
                    float kernels[99] = {0};
                    float decisions[3] = {0};
                    int votes[3] = {0};
                    kernels[0] = compute_kernel(x, -53.0, -71.0, -72.0, 0.0, 0.0);
                    kernels[1] = compute_kernel(x, -60.0, -70.0, -73.0, 0.0, 0.0);
                    kernels[2] = compute_kernel(x, -54.0, -78.0, -85.0, 0.0, 0.0);
                    kernels[3] = compute_kernel(x, -59.0, -85.0, -79.0, -77.0, -81.0);
                    kernels[4] = compute_kernel(x, -55.0, -87.0, -78.0, 0.0, 0.0);
                    kernels[5] = compute_kernel(x, -44.0, -67.0, 0.0, -78.0, 0.0);
                    kernels[6] = compute_kernel(x, -57.0, -83.0, -77.0, -82.0, -79.0);
                    kernels[7] = compute_kernel(x, -54.0, -89.0, -74.0, -72.0, -72.0);
                    kernels[8] = compute_kernel(x, -59.0, -78.0, -76.0, 0.0, -79.0);
                    kernels[9] = compute_kernel(x, -59.0, -77.0, -81.0, -75.0, 0.0);
                    kernels[10] = compute_kernel(x, -63.0, -84.0, -67.0, -73.0, -83.0);
                    kernels[11] = compute_kernel(x, -62.0, -73.0, 0.0, 0.0, 0.0);
                    kernels[12] = compute_kernel(x, -60.0, -71.0, 0.0, -76.0, 0.0);
                    kernels[13] = compute_kernel(x, -49.0, -84.0, -70.0, -79.0, -75.0);
                    kernels[14] = compute_kernel(x, -54.0, -87.0, -78.0, 0.0, -86.0);
                    kernels[15] = compute_kernel(x, -60.0, -79.0, -81.0, 0.0, 0.0);
                    kernels[16] = compute_kernel(x, -57.0, -73.0, -80.0, 0.0, -76.0);
                    kernels[17] = compute_kernel(x, -51.0, -81.0, -71.0, 0.0, -74.0);
                    kernels[18] = compute_kernel(x, -59.0, -88.0, -76.0, -81.0, -81.0);
                    kernels[19] = compute_kernel(x, -59.0, -89.0, 0.0, -89.0, 0.0);
                    kernels[20] = compute_kernel(x, -54.0, -77.0, -74.0, 0.0, -73.0);
                    kernels[21] = compute_kernel(x, -44.0, -79.0, 0.0, 0.0, -76.0);
                    kernels[22] = compute_kernel(x, -59.0, -80.0, 0.0, -79.0, -81.0);
                    kernels[23] = compute_kernel(x, -61.0, -81.0, -74.0, 0.0, -84.0);
                    kernels[24] = compute_kernel(x, -59.0, -74.0, 0.0, 0.0, 0.0);
                    kernels[25] = compute_kernel(x, -55.0, -80.0, -77.0, -77.0, -83.0);
                    kernels[26] = compute_kernel(x, -58.0, -89.0, 0.0, 0.0, -75.0);
                    kernels[27] = compute_kernel(x, -67.0, -85.0, 0.0, -78.0, -75.0);
                    kernels[28] = compute_kernel(x, -61.0, -68.0, -74.0, -76.0, 0.0);
                    kernels[29] = compute_kernel(x, -53.0, -76.0, -75.0, -73.0, -82.0);
                    kernels[30] = compute_kernel(x, -64.0, -85.0, -72.0, 0.0, -78.0);
                    kernels[31] = compute_kernel(x, -60.0, -83.0, -71.0, -71.0, 0.0);
                    kernels[32] = compute_kernel(x, -70.0, -85.0, 0.0, -75.0, 0.0);
                    kernels[33] = compute_kernel(x, -66.0, -72.0, -83.0, -82.0, -73.0);
                    kernels[34] = compute_kernel(x, -63.0, -90.0, -71.0, 0.0, -80.0);
                    kernels[35] = compute_kernel(x, 0.0, -65.0, 0.0, -84.0, -51.0);
                    kernels[36] = compute_kernel(x, -84.0, -57.0, 0.0, -76.0, -50.0);
                    kernels[37] = compute_kernel(x, -77.0, -53.0, 0.0, 0.0, -69.0);
                    kernels[38] = compute_kernel(x, 0.0, -56.0, 0.0, 0.0, -52.0);
                    kernels[39] = compute_kernel(x, -77.0, -65.0, 0.0, 0.0, -74.0);
                    kernels[40] = compute_kernel(x, -81.0, -65.0, 0.0, -76.0, -53.0);
                    kernels[41] = compute_kernel(x, -77.0, -59.0, 0.0, -72.0, -65.0);
                    kernels[42] = compute_kernel(x, -76.0, -66.0, 0.0, -74.0, 0.0);
                    kernels[43] = compute_kernel(x, -76.0, -56.0, -85.0, -82.0, -60.0);
                    kernels[44] = compute_kernel(x, -75.0, -58.0, 0.0, 0.0, 0.0);
                    kernels[45] = compute_kernel(x, -71.0, -52.0, 0.0, -74.0, -52.0);
                    kernels[46] = compute_kernel(x, -73.0, -62.0, 0.0, 0.0, -57.0);
                    kernels[47] = compute_kernel(x, -76.0, -59.0, 0.0, 0.0, 0.0);
                    kernels[48] = compute_kernel(x, -77.0, -56.0, 0.0, 0.0, -48.0);
                    kernels[49] = compute_kernel(x, 0.0, -60.0, 0.0, -68.0, -54.0);
                    kernels[50] = compute_kernel(x, -77.0, -64.0, 0.0, -71.0, -62.0);
                    kernels[51] = compute_kernel(x, 0.0, -69.0, 0.0, 0.0, -64.0);
                    kernels[52] = compute_kernel(x, -77.0, -55.0, 0.0, -74.0, -54.0);
                    kernels[53] = compute_kernel(x, -73.0, -80.0, 0.0, 0.0, -64.0);
                    kernels[54] = compute_kernel(x, -82.0, -59.0, 0.0, 0.0, 0.0);
                    kernels[55] = compute_kernel(x, 0.0, -55.0, 0.0, -77.0, -66.0);
                    kernels[56] = compute_kernel(x, 0.0, -88.0, 0.0, -72.0, -67.0);
                    kernels[57] = compute_kernel(x, -80.0, -73.0, 0.0, 0.0, -68.0);
                    kernels[58] = compute_kernel(x, -75.0, -57.0, 0.0, 0.0, 0.0);
                    kernels[59] = compute_kernel(x, -77.0, -65.0, 0.0, -73.0, -49.0);
                    kernels[60] = compute_kernel(x, -65.0, -81.0, -74.0, -68.0, 0.0);
                    kernels[61] = compute_kernel(x, -76.0, -61.0, -80.0, 0.0, -57.0);
                    kernels[62] = compute_kernel(x, -58.0, -82.0, 0.0, -66.0, -67.0);
                    kernels[63] = compute_kernel(x, -73.0, -65.0, 0.0, -73.0, -64.0);
                    kernels[64] = compute_kernel(x, -73.0, -87.0, -73.0, -68.0, -72.0);
                    kernels[65] = compute_kernel(x, -58.0, -75.0, -78.0, 0.0, 0.0);
                    kernels[66] = compute_kernel(x, -58.0, -82.0, -74.0, -74.0, 0.0);
                    kernels[67] = compute_kernel(x, -63.0, -82.0, -75.0, -69.0, -79.0);
                    kernels[68] = compute_kernel(x, -78.0, -65.0, -87.0, 0.0, -67.0);
                    kernels[69] = compute_kernel(x, -75.0, -66.0, -76.0, -64.0, 0.0);
                    kernels[70] = compute_kernel(x, -77.0, -63.0, 0.0, -67.0, -62.0);
                    kernels[71] = compute_kernel(x, -73.0, -77.0, 0.0, -68.0, -57.0);
                    kernels[72] = compute_kernel(x, -71.0, -80.0, 0.0, -73.0, -74.0);
                    kernels[73] = compute_kernel(x, -72.0, -74.0, 0.0, -73.0, -78.0);
                    kernels[74] = compute_kernel(x, -77.0, -79.0, 82.0, -67.0, -71.0);
                    kernels[75] = compute_kernel(x, -74.0, -73.0, -73.0, -69.0, -63.0);
                    kernels[76] = compute_kernel(x, -80.0, -82.0, 0.0, -79.0, -80.0);
                    kernels[77] = compute_kernel(x, -64.0, -70.0, -80.0, -75.0, -66.0);
                    kernels[78] = compute_kernel(x, -72.0, -67.0, 0.0, -68.0, -66.0);
                    kernels[79] = compute_kernel(x, -68.0, -78.0, -80.0, 0.0, -69.0);
                    kernels[80] = compute_kernel(x, -69.0, -73.0, 0.0, 0.0, -60.0);
                    kernels[81] = compute_kernel(x, -67.0, -72.0, 0.0, 0.0, -67.0);
                    kernels[82] = compute_kernel(x, -75.0, -74.0, 0.0, -67.0, -58.0);
                    kernels[83] = compute_kernel(x, -89.0, -87.0, -81.0, -74.0, -72.0);
                    kernels[84] = compute_kernel(x, -88.0, -65.0, 0.0, 0.0, -53.0);
                    kernels[85] = compute_kernel(x, -64.0, -80.0, -78.0, 0.0, -76.0);
                    kernels[86] = compute_kernel(x, -59.0, -84.0, -80.0, -77.0, -67.0);
                    kernels[87] = compute_kernel(x, -74.0, -60.0, 0.0, -70.0, -59.0);
                    kernels[88] = compute_kernel(x, -59.0, -79.0, -77.0, 0.0, -76.0);
                    kernels[89] = compute_kernel(x, -58.0, -78.0, -79.0, -70.0, -70.0);
                    kernels[90] = compute_kernel(x, -68.0, -80.0, -75.0, -74.0, -77.0);
                    kernels[91] = compute_kernel(x, -72.0, -72.0, 0.0, -73.0, 71.0);
                    kernels[92] = compute_kernel(x, -66.0, -88.0, -82.0, -70.0, -72.0);
                    kernels[93] = compute_kernel(x, -76.0, -63.0, 0.0, -72.0, -58.0);
                    kernels[94] = compute_kernel(x, -63.0, -82.0, -84.0, 0.0, 0.0);
                    kernels[95] = compute_kernel(x, -72.0, -72.0, -76.0, 0.0, -70.0);
                    kernels[96] = compute_kernel(x, -63.0, -80.0, -84.0, -77.0, -75.0);
                    kernels[97] = compute_kernel(x, -60.0, -72.0, -81.0, -68.0, -77.0);
                    kernels[98] = compute_kernel(x, -62.0, -81.0, -85.0, -72.0, -77.0);
                    decisions[0] = 0.165606230631 + kernels[0] * 0.054392419664 + kernels[1] * 0.390526454793 + kernels[2] * 0.145008789968 + kernels[4] * 0.380804879019 + kernels[5] * 0.365123482113 + kernels[7] * 0.140668319352 + kernels[9] * 0.144237852116 + kernels[10] * 0.197984095394 + kernels[11] + kernels[12] * 0.680140039395 + kernels[13] * 0.153128996233 + kernels[14] * 0.167029642867 + kernels[16] * 0.405333895984 + kernels[17] * 0.074445323387 + kernels[21] * 0.860291297036 + kernels[22] * 0.594366502183 + kernels[24] + kernels[26] + kernels[27] + kernels[28] * 0.425581548349 + kernels[31] * 0.373695098094 + kernels[32] * 0.895164310982 + kernels[33] + kernels[34] * 0.364905280001 + kernels[35] * -0.37326807397 + kernels[37] * -0.048534670811 + kernels[38] * -0.595407963742 + kernels[39] * -0.860442064598 + kernels[41] * -0.524699329844 - kernels[42] - kernels[43] - kernels[44] + kernels[45] * -0.010982992672 - kernels[47] + kernels[48] * -0.361803726835 + kernels[49] * -0.178791646147 - kernels[50] + kernels[51] * -0.835462677442 - kernels[53] + kernels[55] * -0.512655719294 + kernels[56] * -0.80054059624 + kernels[58] * -0.357850888511 + kernels[59] * -0.352387876824;
                    decisions[1] = 0.095299674374 + kernels[1] * 0.876720226374 + kernels[2] * 0.582867105491 + kernels[3] + kernels[4] * 0.620241227268 + kernels[5] * 0.6060175486 + kernels[6] + kernels[7] + kernels[8] + kernels[9] + kernels[10] + kernels[11] * 0.63253028244 + kernels[15] + kernels[16] + kernels[18] + kernels[19] * 0.303235256514 + kernels[20] + kernels[21] + kernels[22] + kernels[23] * 0.172200258776 + kernels[24] * 0.307021989806 + kernels[25] + kernels[26] + kernels[27] + kernels[28] + kernels[29] + kernels[30] + kernels[31] + kernels[32] * 0.48424367939 + kernels[33] + kernels[34] * 0.659566340201 - kernels[60] + kernels[61] * -0.380596643745 - kernels[62] + kernels[64] * -0.348445111103 - kernels[65] - kernels[66] - kernels[67] + kernels[68] * -0.20802542648 - kernels[69] + kernels[72] * -0.649983274787 + kernels[73] * -0.353227022284 - kernels[74] - kernels[76] - kernels[79] + kernels[80] * -0.975611302391 - kernels[81] + kernels[83] * -0.10134646824 + kernels[84] * -0.212828060391 - kernels[85] - kernels[86] + kernels[87] * -0.442818097066 - kernels[88] - kernels[89] - kernels[90] - kernels[91] + kernels[92] * -0.571762508373 - kernels[94] - kernels[95] - kernels[96] - kernels[97] - kernels[98];
                    decisions[2] = -0.079353674394 + kernels[35] * 0.335264187337 + kernels[36] * 0.218016380369 + kernels[37] * 0.17404080079 + kernels[38] * 0.617375492034 + kernels[39] * 0.47843389859 + kernels[40] + kernels[41] + kernels[42] * 0.934157461254 + kernels[43] + kernels[45] + kernels[46] * 0.693246007666 + kernels[48] + kernels[49] * 0.176571515698 + kernels[50] + kernels[51] * 0.630157354604 + kernels[52] + kernels[53] + kernels[54] * 0.613569617939 + kernels[55] * 0.472862955272 + kernels[56] * 0.729081449368 + kernels[57] + kernels[58] * 0.424472240579 + kernels[59] + kernels[61] * -0.5131349928 - kernels[63] + kernels[65] * -0.468711253709 + kernels[66] * -0.599087422409 + kernels[68] * -0.171356367631 + kernels[69] * -0.601472589222 - kernels[70] + kernels[71] * -0.3506170428 + kernels[74] * -0.919036209852 + kernels[75] * -0.176811435397 + kernels[76] * -0.163724988141 - kernels[77] + kernels[78] * -0.717613420405 - kernels[80] - kernels[81] - kernels[82] + kernels[83] * -0.649081924575 - kernels[84] - kernels[87] + kernels[88] * -0.639298697614 + kernels[91] * -0.926296941011 - kernels[93] + kernels[94] * -0.478990733915 + kernels[97] * -0.122015342019;
                    votes[decisions[0] > 0 ? 0 : 1] += 1;
                    votes[decisions[1] > 0 ? 0 : 2] += 1;
                    votes[decisions[2] > 0 ? 1 : 2] += 1;
                    int val = votes[0];
                    int idx = 0;

                    for (int i = 1; i < 3; i++)
                    {
                        if (votes[i] > val)
                        {
                            val = votes[i];
                            idx = i;
                        }
                    }

                    return idx;
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