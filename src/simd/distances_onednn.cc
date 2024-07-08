#ifdef KNOWHERE_WITH_DNNL
#include "distances_onednn.h"

namespace faiss {

thread_local faiss::inner_product_desc inner_product_desc_t;

void fvec_f32bf16f32_inner_product_onednn(size_t xrow, size_t xcol, size_t yrow, size_t ycol,
    float* in_f32_1, float* in_f32_2, float* out_f32) {
    inner_product_desc_t.execute(xrow, xcol, yrow, ycol, in_f32_1, in_f32_2, out_f32);
}
}
#endif
