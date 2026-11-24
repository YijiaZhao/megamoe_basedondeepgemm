#pragma once

#include <functional>
#include <pybind11/functional.h>

#if DG_TENSORMAP_COMPATIBLE
#include "../jit/compiler.hpp"
#endif
#include "../jit/device_runtime.hpp"
#include "../jit_kernels/impls/sm90_fp4_mega_moe_h20_fused.hpp"

namespace deep_gemm::mega_fused {

static void validate_stats(const std::optional<torch::Tensor>& stats,
                           int num_experts_per_rank,
                           const torch::Device& device) {
    if (!stats.has_value()) return;
    DG_HOST_ASSERT(stats->scalar_type() == torch::kInt);
    DG_HOST_ASSERT(stats->is_contiguous() and stats->device() == device);
    DG_HOST_ASSERT(stats->numel() == num_experts_per_rank);
}

static void validate_row_scale(const torch::Tensor& scale,
                               int experts, int rows,
                               const torch::Device& device) {
    DG_HOST_ASSERT(scale.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(scale.dim() == 2 and scale.size(0) == experts and scale.size(1) == rows);
    DG_HOST_ASSERT(scale.is_contiguous() and scale.device() == device);
}

static std::tuple<int64_t, std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(const torch::Tensor&)>>
get_symm_buffer_size_for_fused_mega_moe(
        const int& num_ranks, const int& num_experts,
        const int& max_tokens, const int& topk,
        const int& hidden, const int& intermediate) {
    DG_HOST_ASSERT(num_experts % num_ranks == 0);
    const auto workspace = fused_layout::Workspace(nullptr, num_ranks, num_experts, max_tokens, topk);
    const auto token = fused_layout::Data(hidden);
    const auto bf16 = fused_layout::Data(hidden * 2);
    const auto mid = fused_layout::Data(intermediate);
    const auto sf = fused_layout::Data(hidden / 32, false);
    const auto mid_sf = fused_layout::Data(intermediate / 16, false);
    const auto idx = fused_layout::Data(topk * sizeof(int64_t), false);
    const auto weight = fused_layout::Data(topk * sizeof(float), false);
    const auto route_weight = fused_layout::Data(sizeof(float), false);
    const int local_experts = num_experts / num_ranks;
    const int pool = fused_layout::get_num_max_pool_tokens(num_ranks, max_tokens, topk, local_experts);
    int sf_pool = 0;
    for (int bm: fused_layout::kCandidateBlockM)
        sf_pool = std::max(sf_pool, fused_layout::get_num_padded_sf_pool_tokens(pool, bm));
    const auto x = fused_layout::Buffer(token,1,max_tokens,workspace.get_end_ptr());
    const auto xsf = fused_layout::Buffer(sf,1,max_tokens,x.get_end_ptr());
    const auto ti = fused_layout::Buffer(idx,1,max_tokens,xsf.get_end_ptr());
    const auto tw = fused_layout::Buffer(weight,1,max_tokens,ti.get_end_ptr());
    const auto a1 = fused_layout::Buffer(token,1,pool,tw.get_end_ptr());
    const auto a1sf = fused_layout::Buffer(sf,1,sf_pool,a1.get_end_ptr());
    const auto rw = fused_layout::Buffer(route_weight,1,pool,a1sf.get_end_ptr());
    const auto a2 = fused_layout::Buffer(mid,1,pool,rw.get_end_ptr());
    const auto a2sf = fused_layout::Buffer(mid_sf,1,sf_pool,a2.get_end_ptr());
    const auto combine = fused_layout::Buffer(bf16,topk,max_tokens,a2sf.get_end_ptr());
    auto slice=[=](const torch::Tensor& b) {
        auto t=[](void* p,std::vector<int64_t> shape,torch::ScalarType dt,const torch::Tensor& owner){return torch::from_blob(p, shape, [owner](void*) mutable {}, torch::TensorOptions().dtype(dt).device(owner.device()));};
        return std::make_tuple(
            t(math::advance_ptr(b.data_ptr(),reinterpret_cast<int64_t>(x.base)),{max_tokens,hidden},torch::kFloat8_e4m3fn,b),
            t(math::advance_ptr(b.data_ptr(),reinterpret_cast<int64_t>(xsf.base)),{max_tokens,hidden/128},torch::kFloat32,b),
            t(math::advance_ptr(b.data_ptr(),reinterpret_cast<int64_t>(ti.base)),{max_tokens,topk},torch::kInt64,b),
            t(math::advance_ptr(b.data_ptr(),reinterpret_cast<int64_t>(tw.base)),{max_tokens,topk},torch::kFloat32,b),
            t(math::advance_ptr(b.data_ptr(),reinterpret_cast<int64_t>(a1.base)),{pool,hidden},torch::kFloat8_e4m3fn,b),
            t(math::advance_ptr(b.data_ptr(),reinterpret_cast<int64_t>(a1sf.base)),{sf_pool,hidden/128},torch::kFloat32,b),
            t(math::advance_ptr(b.data_ptr(),reinterpret_cast<int64_t>(a2.base)),{pool,intermediate},torch::kFloat8_e4m3fn,b),
            t(math::advance_ptr(b.data_ptr(),reinterpret_cast<int64_t>(a2sf.base)),{sf_pool,intermediate/64},torch::kFloat32,b),
            // per-(topk slot, local token) scattered L2 partials (combine input; diagnostics)
            t(math::advance_ptr(b.data_ptr(),reinterpret_cast<int64_t>(combine.base)),{topk,max_tokens,hidden},torch::kBFloat16,b));
    };
    return {reinterpret_cast<int64_t>(combine.get_end_ptr()),slice};
}

static void run_fused(
        int mode, const torch::Tensor& y,
        const std::tuple<torch::Tensor,torch::Tensor>& l1,
        const std::tuple<torch::Tensor,torch::Tensor>& l2,
        const std::optional<torch::Tensor>& stats,
        const torch::Tensor& l1_scale, const torch::Tensor& l2_scale,
        const torch::Tensor& buffer, const std::vector<int64_t>& ptrs,
        const int& rank, const int& max_tokens, const int& experts, const int& topk,
        const std::optional<float>& clamp, const bool& fast_math,
        const std::optional<torch::Tensor>& phase_stamps = std::nullopt,
        // DG_FE_SELECT_IN_MEGA=1: the Fable cc frontend's compact key array (see the host)
        const std::optional<torch::Tensor>& fe_keys = std::nullopt,
        // Two-node EP16 over NCCL GIN (deep_gemm/mega/ep16.py): y holds the T own rows, the kernel runs 2T
        const std::vector<int64_t>& ep16_gin = {}) {
    const auto [w1,s1]=l1; const auto [w2,s2]=l2;
    const int nr=ptrs.size(), local=experts/nr, h=y.size(1);
    const int inter=s2.size(2)*128;
    validate_stats(stats,local,y.device()); validate_row_scale(l1_scale,local,inter*2,y.device()); validate_row_scale(l2_scale,local,h,y.device());
    const auto [bytes,slice]=get_symm_buffer_size_for_fused_mega_moe(nr,experts,max_tokens,topk,h,inter);
    DG_HOST_ASSERT(buffer.nbytes() >= static_cast<size_t>(bytes));
    const auto [x,xsf,ti,tw,a1,a1sf,a2,a2sf,comb]=slice(buffer);
    sm90_fp4_h20_fused_mega_moe(y,a1,a1sf,a2,a2sf,w1,w2,stats,l1_scale,l2_scale,ptrs,rank,max_tokens,local,static_cast<int>(y.size(0))*(ep16_gin.empty()?1:2),topk,h,inter,clamp.value_or(std::numeric_limits<float>::infinity()),fast_math,mode==1,phase_stamps,mode==2,fe_keys,ep16_gin);
}

// Trailing optional: the Fable cc frontend's compact key array (DG_FE_SELECT_IN_MEGA=1), see the host.
static void mxfp4_mega_moe_fused(const torch::Tensor& y,const std::tuple<torch::Tensor,torch::Tensor>& l1,const std::tuple<torch::Tensor,torch::Tensor>& l2,const std::optional<torch::Tensor>& stats,const torch::Tensor& s1,const torch::Tensor& s2,const torch::Tensor& b,const std::vector<int64_t>& p,const int& r,const int& mt,const int& e,const int& k,const std::optional<float>& c,const bool& f,const std::optional<torch::Tensor>& ps,const std::optional<torch::Tensor>& fe_keys = std::nullopt,const std::vector<int64_t>& ep16_gin = {}){run_fused(1,y,l1,l2,stats,s1,s2,b,p,r,mt,e,k,c,f,ps,fe_keys,ep16_gin);}
static void qoq_mega_moe_fused(const torch::Tensor& y,const std::tuple<torch::Tensor,torch::Tensor>& l1,const std::tuple<torch::Tensor,torch::Tensor>& l2,const std::optional<torch::Tensor>& stats,const torch::Tensor& s1,const torch::Tensor& s2,const torch::Tensor& b,const std::vector<int64_t>& p,const int& r,const int& mt,const int& e,const int& k,const std::optional<float>& c,const bool& f,const std::optional<torch::Tensor>& ps,const std::optional<torch::Tensor>& fe_keys = std::nullopt,const std::vector<int64_t>& ep16_gin = {}){run_fused(2,y,l1,l2,stats,s1,s2,b,p,r,mt,e,k,c,f,ps,fe_keys,ep16_gin);}

static void register_apis(pybind11::module_& m) {
#if DG_TENSORMAP_COMPATIBLE
    m.def("get_symm_buffer_size_for_fused_mega_moe",&get_symm_buffer_size_for_fused_mega_moe);
    m.def("mxfp4_mega_moe_fused",&mxfp4_mega_moe_fused);
    m.def("qoq_mega_moe_fused",&qoq_mega_moe_fused);
#endif
}
} // namespace deep_gemm::mega_fused
