#ifdef KNOWHERE_WITH_DNNL
#include "distances_onednn.h"

namespace faiss {

thread_local faiss::inner_product_desc inner_product_desc_t;

void fvec_bf16bf16f32_inner_product_onednn(uint32_t xrow, uint32_t xcol, uint32_t yrow, uint32_t ycol,
    uint16_t* in_bf16_1, uint16_t* in_bf16_2, float* out_f32) {
    inner_product_desc_t.execute_bf16(xrow, xcol, yrow, ycol, in_bf16_1, in_bf16_2, out_f32);
}

void fvec_f32bf16f32_inner_product_onednn(uint32_t xrow, uint32_t xcol, uint32_t yrow, uint32_t ycol,
    float* in_f32_1, float* in_f32_2, float* out_f32) {
    inner_product_desc_t.execute(xrow, xcol, yrow, ycol, in_f32_1, in_f32_2, out_f32);
}
}
#endif
