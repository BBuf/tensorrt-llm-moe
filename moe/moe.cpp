#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_fp16.h>

#include <optional>
#include <algorithm>

#include "tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.h"

#include "utils.h"

using torch::Tensor;

int getSMVersion() {
    int device = -1;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    return props.major * 10 + props.minor;
}

template<typename T, typename WeightType>
std::vector<tensorrt_llm::cutlass_extensions::CutlassGemmConfig> getFilteredConfigs(
    tensorrt_llm::kernels::CutlassMoeFCRunner<T, WeightType>& moe_runner, int sm) {
    auto tactics = moe_runner.getTactics();
    if (sm == 89) {
        // Filter some unsupported configs for L40S
        auto it = std::remove_if(tactics.begin(), tactics.end(),
            [&](auto conf) {
                using tensorrt_llm::cutlass_extensions::CutlassTileConfig;
                auto checks = std::vector{
                    // Fail for BF16/FP16
                    conf.tile_config == CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64,
                    conf.tile_config == CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64 && conf.stages == 4,
                    // Fail for FP8
                    false && conf.tile_config == CutlassTileConfig::CtaShape16x256x128_WarpShape16x64x128
                        && conf.stages >= 3,
                };

                return std::any_of(checks.begin(), checks.end(), [](auto v) { return v; });
            });
        tactics.erase(it, tactics.end());
    }

    if (tactics.empty()) {
        throw std::runtime_error("No valid GEMM tactics found");
    }

    return tactics;
}

template<typename T, typename WeightType>
std::pair<tensorrt_llm::cutlass_extensions::CutlassGemmConfig, tensorrt_llm::cutlass_extensions::CutlassGemmConfig> 
selectTacticsForArch(tensorrt_llm::kernels::CutlassMoeFCRunner<T, WeightType>& moe_runner, int sm) {
    bool is_sm90 = sm >= 90;  // 这里移除了INT_QUANT的判断,因为我们在模板中处理类型
    auto tactics = getFilteredConfigs(moe_runner, sm);
    auto it = std::find_if(tactics.begin(), tactics.end(), [is_sm90](auto& c) { return c.is_sm90 == is_sm90; });
    if (it == tactics.end()) {
        // Fall back to any tactic
        std::cout << "WARNING: Could not find config for sm version " << sm << std::endl;
        return std::make_pair(tactics[0], tactics[0]);
    }

    return std::make_pair(*it, *it);
}

tensorrt_llm::ActivationType getActivationType(std::string activation_type_str)
{
    if (activation_type_str == "Gelu" || activation_type_str == "gelu") {
        return tensorrt_llm::ActivationType::Gelu;
    }
    else if (activation_type_str == "Relu" || activation_type_str == "relu") {
        return tensorrt_llm::ActivationType::Relu;
    }
    else if (activation_type_str == "Silu" || activation_type_str == "silu") {
        return tensorrt_llm::ActivationType::Silu;
    }
    else if (activation_type_str == "GeGLU" || activation_type_str == "geglu" || activation_type_str == "gated-gelu") {
        return tensorrt_llm::ActivationType::Geglu;
    }
    else if (activation_type_str == "Swiglu") {
        return tensorrt_llm::ActivationType::Swiglu;
    }
    else {
        std::cout << "Activation Type: " <<  activation_type_str << " not supported !";
    }
    return tensorrt_llm::ActivationType::InvalidType;
}

template<typename T, typename WeightType>
Tensor trt_llm_fused_moe_helper(Tensor                            input_activations, //(num_tokens, hidden_size)
                         Tensor                            gating_output, //(num_tokens, num_experts)
                         Tensor                            fc1_expert_weights, //(num_experts, hidden_size, inter_size)
                         tensorrt_llm::ActivationType fc1_activation_type,
                         Tensor                            fc2_expert_weights, //(num_experts, inter_size, hidden_size)
                         const int                         active_rows,
                         const int                         k)
{

    const int num_rows    = input_activations.size(0); //(num_tokens, hidden_size)
    const int hidden_size = input_activations.size(1);
    const int inter_size  = fc2_expert_weights.size(1); //(num_experts, inter_size, hidden_size)
    const int num_experts = gating_output.size(-1); //(num_tokens, num_experts)
    auto      stream      = at::cuda::getCurrentCUDAStream().stream();

    T* input_act_ptr     = get_ptr<T>(input_activations);
    float* gating_output_ptr = get_ptr<float>(gating_output);

    WeightType*           fc1_expert_weights_ptr = get_ptr<WeightType>(fc1_expert_weights);

    T* fc1_expert_biases_ptr = nullptr;

    WeightType* fc2_expert_weights_ptr = get_ptr<WeightType>(fc2_expert_weights);
    T*          fc2_expert_biases_ptr  = nullptr;

    bool* finished_ptr = nullptr;

    tensorrt_llm::kernels::MOEParallelismConfig moe_parallel_config = tensorrt_llm::kernels::MOEParallelismConfig(1, 0, 1, 0);
    tensorrt_llm::kernels::CutlassMoeFCRunner<T, WeightType> moe_runner;

    int sm = getSMVersion();
    auto [tactic1, tactic2] = selectTacticsForArch(moe_runner, sm);
    moe_runner.setTactic(std::make_optional(tactic1), std::make_optional(tactic2));

    size_t bytes        = moe_runner.getWorkspaceSize(num_rows, hidden_size, inter_size, num_experts, k, fc1_activation_type, tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::RENORMALIZE, moe_parallel_config);

    auto workspace_tensor = torch::empty({bytes}, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));
    char* workspace_ptr   = get_ptr<char>(workspace_tensor);

    const at::ScalarType _st = input_activations.scalar_type();
    auto                 fc2_output =
        torch::empty({k * num_rows, hidden_size}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));

    auto expert_scales     = torch::empty({num_rows, k}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    T*   expert_scales_ptr = get_ptr<T>(expert_scales);

    auto expanded_source_row_to_expanded_dest_row =
        torch::empty({num_rows, k}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    int* expanded_source_row_to_expanded_dest_row_ptr = get_ptr<int>(expanded_source_row_to_expanded_dest_row);

    auto expert_for_source_row =
        torch::empty({num_rows, k}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    int* expert_for_source_row_ptr = get_ptr<int>(expert_for_source_row);

    auto output_tensor =
        torch::empty({num_rows, hidden_size}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    T* output_tensor_ptr = get_ptr<T>(output_tensor);

    tensorrt_llm::kernels::QuantParams quant_params;
    quant_params = tensorrt_llm::kernels::QuantParams::FP8(nullptr, nullptr, nullptr);

    moe_runner.runMoe(input_act_ptr,
                        gating_output_ptr,
                        fc1_expert_weights_ptr,
                        fc1_expert_biases_ptr, // nullptr
                        fc1_activation_type,
                        fc2_expert_weights_ptr,
                        fc2_expert_biases_ptr, // nullptr
                        quant_params,
                        num_rows,
                        hidden_size,
                        inter_size,
                        num_experts,
                        k,
                        workspace_ptr,
                        output_tensor_ptr,
                        finished_ptr, // nullptr
                        active_rows, // original num_rows
                        expert_scales_ptr,
                        expanded_source_row_to_expanded_dest_row_ptr,
                        expert_for_source_row_ptr,
                        0.2f, // sparse_mixer_epsilon
                        moe_parallel_config,
                        tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::RENORMALIZE,
                        stream);

    return output_tensor;
}


Tensor trt_llm_fused_moe(Tensor      input_activations, //(num_tokens, hidden_size)
                  Tensor      gating_output, //(num_tokens, num_experts)
                  Tensor      fc1_expert_weights, //(num_experts, hidden_size, inter_size)
                  std::string fc1_activation_type_str,
                  Tensor      fc2_expert_weights, //(num_experts, inter_size, hidden_size)
                  int64_t     active_rows,
                  int64_t     k)
{

    const at::ScalarType _st = input_activations.scalar_type();

    const int num_rows    = input_activations.size(0);
    const int hidden_size = input_activations.size(1);
    const int num_experts = gating_output.size(-1);

    torch::ScalarType quant_type = fc2_expert_weights.scalar_type();

    CHECK_INPUT(input_activations, _st);
    TORCH_CHECK(input_activations.dim() == 2, "Invalid rank for activations");

    CHECK_INPUT(gating_output, _st);
    TORCH_CHECK(gating_output.dim() == 2, "Invalid rank for gating output");
    TORCH_CHECK(gating_output.size(0) == num_rows, "gating output and activations must have same number of rows");

    CHECK_TH_CUDA(fc1_expert_weights);
    CHECK_CONTIGUOUS(fc1_expert_weights);
    TORCH_CHECK(fc1_expert_weights.dim() == 3, "Invalid rank for fc1 weights");
    TORCH_CHECK(fc1_expert_weights.size(0) == num_experts, "Experts mismatch between gate outputs and fc1 weights");
    TORCH_CHECK(fc1_expert_weights.size(1) == hidden_size,
                "Activation last dim must equal size of dim 1 for fc1 weight");
    
    CHECK_TH_CUDA(fc2_expert_weights);
    CHECK_CONTIGUOUS(fc2_expert_weights);
    TORCH_CHECK(fc2_expert_weights.dim() == 3, "Invalid rank for fc2 weights");
    TORCH_CHECK(fc2_expert_weights.size(0) == gating_output.size(-1),
                "Experts mismatch between gate outputs and fc2 weights");

    Tensor output_tensor;

    tensorrt_llm::ActivationType fc1_activation_type = tensorrt_llm::ActivationType::InvalidType;
    if (fc1_activation_type_str == "identity") {
        fc1_activation_type = tensorrt_llm::ActivationType::Identity;
    }
    else {
        fc1_activation_type = getActivationType(fc1_activation_type_str);
    }

    switch (_st) {
        case at::ScalarType::Float: {

            if (quant_type == _st) {
                output_tensor = trt_llm_fused_moe_helper<float, float>(input_activations,
                                                                gating_output,
                                                                fc1_expert_weights,
                                                                fc1_activation_type,
                                                                fc2_expert_weights,
                                                                active_rows,
                                                                k);
            }
            else {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(quant_type));
                throw std::runtime_error(err_msg);
            }
            break;
        }
        case at::ScalarType::Half: {

            if (quant_type == _st) {
                output_tensor = trt_llm_fused_moe_helper<half, half>(input_activations,
                                                              gating_output,
                                                              fc1_expert_weights,
                                                              fc1_activation_type,
                                                              fc2_expert_weights,
                                                              active_rows,
                                                              k);
            }
            else {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(quant_type));
                throw std::runtime_error(err_msg);
            }
            break;
        }
#ifdef ENABLE_BF16
        case at::ScalarType::BFloat16: {
            if (quant_type == _st) {
                output_tensor = trt_llm_fused_moe_helper<__nv_bfloat16, __nv_bfloat16>(input_activations,
                                                                                gating_output,
                                                                                fc1_expert_weights,
                                                                                fc1_activation_type,
                                                                                fc2_expert_weights,
                                                                                active_rows,
                                                                                k);
            }
            else {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(quant_type));
                throw std::runtime_error(err_msg);
            }
            break;
        }
#endif
        default:
            throw std::runtime_error("Wrong Tensor type.");
    }
    return output_tensor;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("trt_llm_fused_moe", &trt_llm_fused_moe, "moe.");
}
