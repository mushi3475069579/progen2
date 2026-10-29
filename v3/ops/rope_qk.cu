// E14: FP32 RoPE Q/K fusion. Compiled to HIP by PyTorch's HIP extension path.
// Apache-2.0. Rotation semantics follow the supplied ProGen2 implementation.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <vector>

struct Strides4 { int64_t b, l, h, d; };

__global__ void rope_qk_kernel(
    const float* q, const float* k, const float* sin, const float* cos,
    float* out_q, float* out_k, int64_t count, int64_t length,
    int64_t heads, int64_t width, int64_t rotary, int64_t offset,
    Strides4 qs, Strides4 ks) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t d = i % width;
        const int64_t h = (i / width) % heads;
        const int64_t l = (i / (width * heads)) % length;
        const int64_t b = i / (width * heads * length);
        const int64_t qi = b * qs.b + l * qs.l + h * qs.h + d * qs.d;
        const int64_t ki = b * ks.b + l * ks.l + h * ks.h + d * ks.d;
        if (d < rotary) {
            const int64_t partner = d ^ 1;
            const float q_partner = q[qi + (partner - d) * qs.d];
            const float k_partner = k[ki + (partner - d) * ks.d];
            const float qr = (d & 1) ? q_partner : -q_partner;
            const float kr = (d & 1) ? k_partner : -k_partner;
            const int64_t position = (offset + l) * rotary + d;
            const float s = sin[position];
            const float c = cos[position];
            // Compile with -ffp-contract=off and without fast math. The original
            // performs two rounded FP32 products, followed by a rounded sum.
            const float q_first = q[qi] * c;
            const float q_second = qr * s;
            const float k_first = k[ki] * c;
            const float k_second = kr * s;
            out_q[i] = q_first + q_second;
            out_k[i] = k_first + k_second;
        } else {
            // Include unrotated channels, avoiding the original output cat.
            out_q[i] = q[qi];
            out_k[i] = k[ki];
        }
    }
}

std::vector<at::Tensor> rope_qk(
    const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& sin, const at::Tensor& cos, int64_t offset) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && sin.is_cuda() && cos.is_cuda(),
                "E14 requires device tensors");
    TORCH_CHECK(q.scalar_type() == at::kFloat && k.scalar_type() == at::kFloat &&
                sin.scalar_type() == at::kFloat && cos.scalar_type() == at::kFloat,
                "E14 requires FP32 inputs and coefficients");
    TORCH_CHECK(q.device() == k.device() && q.device() == sin.device() &&
                q.device() == cos.device(), "E14 tensors must share a device");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && q.sizes() == k.sizes(),
                "E14 Q/K must have equal BLHD shapes");
    TORCH_CHECK(sin.dim() == 4 && cos.sizes() == sin.sizes() && sin.size(0) == 1 &&
                sin.size(2) == 1 && sin.is_contiguous() && cos.is_contiguous(),
                "E14 coefficients must be contiguous (1,capacity,1,rotary)");
    const int64_t length = q.size(1), heads = q.size(2), width = q.size(3);
    const int64_t rotary = sin.size(3);
    TORCH_CHECK(q.size(0) > 0 && length > 0 && heads > 0 && width > 0,
                "E14 does not accept empty tensors");
    TORCH_CHECK(rotary > 0 && rotary <= width && rotary % 2 == 0,
                "E14 requires an even rotary dimension within head width");
    TORCH_CHECK(offset >= 0 && length <= sin.size(1) &&
                offset <= sin.size(1) - length, "E14 position is out of bounds");
    for (int axis = 0; axis < 4; ++axis) {
        TORCH_CHECK(q.stride(axis) >= 0 && k.stride(axis) >= 0,
                    "E14 requires nonnegative input strides");
    }
    const c10::cuda::CUDAGuard guard(q.device());
    auto out_q = at::empty(q.sizes(), q.options());
    auto out_k = at::empty(k.sizes(), k.options());
    const Strides4 qs{q.stride(0), q.stride(1), q.stride(2), q.stride(3)};
    const Strides4 ks{k.stride(0), k.stride(1), k.stride(2), k.stride(3)};
    const int threads = 256;
    const int64_t needed = (q.numel() + threads - 1) / threads;
    const unsigned int blocks = static_cast<unsigned int>(needed < 65535 ? needed : 65535);
    auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
    hipLaunchKernelGGL(rope_qk_kernel, dim3(blocks), dim3(threads), 0, stream.stream(),
        q.data_ptr<float>(), k.data_ptr<float>(), sin.data_ptr<float>(), cos.data_ptr<float>(),
        out_q.data_ptr<float>(), out_k.data_ptr<float>(), q.numel(), length, heads,
        width, rotary, offset, qs, ks);
    const hipError_t error = hipGetLastError();
    TORCH_CHECK(error == hipSuccess, "E14 HIP launch failed: ", hipGetErrorString(error));
    return {out_q, out_k};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("rope_qk", &rope_qk, "Fused FP32 RoPE Q/K, including passthrough channels");
}
