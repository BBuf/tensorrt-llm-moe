#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <optional>

#include "tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.h"
#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels.h"

#include "utils.h"

using torch::Tensor;

// template<typename T>
// inline T* get_ptr(torch::Tensor& t)
// {
//     return reinterpret_cast<T*>(t.data_ptr());
// }

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
Tensor run_moe_fc_helper(Tensor                            input_activations, //(num_tokens, hidden_size)
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
    T* gating_output_ptr = get_ptr<T>(gating_output);

    WeightType*           fc1_expert_weights_ptr = get_ptr<WeightType>(fc1_expert_weights);
    static constexpr bool is_fp16_or_fp32 =
        std::is_same<WeightType, float>::value || std::is_same<WeightType, half>::value;
#ifdef ENABLE_BF16
    static constexpr bool ignore_scales = is_fp16_or_fp32 || std::is_same<WeightType, __nv_bfloat16>::value;
#else
    static constexpr bool ignore_scales = is_fp16_or_fp32;
#endif

    T* fc1_scales_ptr        = ignore_scales ? nullptr : nullptr;
    T* fc1_expert_biases_ptr = nullptr;

    WeightType* fc2_expert_weights_ptr = get_ptr<WeightType>(fc2_expert_weights);
    T*          fc2_scales_ptr         = ignore_scales ? nullptr : nullptr;
    T*          fc2_expert_biases_ptr  = nullptr;

    // bool* finished_ptr   = get_ptr<bool>(finished);
    bool* finished_ptr = nullptr;

    tensorrt_llm::kernels::MOEParallelismConfig moe_parallel_config = tensorrt_llm::kernels::MOEParallelismConfig::TensorParallelism(1, 0);
    tensorrt_llm::kernels::CutlassMoeFCRunner<T, WeightType> moe_runner;
    long int bytes        = moe_runner.getWorkspaceSize(num_rows, hidden_size, inter_size, num_experts, k, fc1_activation_type, moe_parallel_config);

    auto workspace_tensor = torch::empty({bytes}, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));
    char* workspace_ptr   = get_ptr<char>(workspace_tensor);

    const at::ScalarType _st = input_activations.scalar_type();
    auto                 fc2_output =
        torch::empty({k * num_rows, hidden_size}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    T* fc2_output_ptr = get_ptr<T>(fc2_output);

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

    moe_runner.runMoe(input_act_ptr,
                        gating_output_ptr,
                        fc1_expert_weights_ptr,
                        fc1_scales_ptr, // nullptr
                        fc1_expert_biases_ptr, // nullptr
                        fc1_activation_type,
                        fc2_expert_weights_ptr,
                        fc2_scales_ptr, // nullptr
                        fc2_expert_biases_ptr, // nullptr
                        num_rows,
                        hidden_size,
                        inter_size,
                        num_experts,
                        k,
                        workspace_ptr,
                        output_tensor_ptr,
                        fc2_output_ptr,
                        finished_ptr, // nullptr
                        active_rows, // original num_rows
                        expert_scales_ptr,
                        expanded_source_row_to_expanded_dest_row_ptr,
                        expert_for_source_row_ptr,
                        moe_parallel_config,
                        tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::NONE,
                        stream);

    return output_tensor;
}


Tensor run_moe_fc(Tensor      input_activations, //(num_tokens, hidden_size)
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
    const int inter_size  = fc2_expert_weights.size(1);
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

    const int fc1_num_cols = fc1_expert_weights.size(-1);

    
    CHECK_TH_CUDA(fc2_expert_weights);
    CHECK_CONTIGUOUS(fc2_expert_weights);
    TORCH_CHECK(fc2_expert_weights.dim() == 3, "Invalid rank for fc2 weights");
    TORCH_CHECK(fc2_expert_weights.size(0) == gating_output.size(-1),
                "Experts mismatch between gate outputs and fc2 weights");
    // TORCH_CHECK(fc2_expert_weights.size(1) == fc1_num_cols, "fc1 weight last dim must equal dim 1 of fc2 weights"); 如果是 glu 类，该条件无法满足

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
                output_tensor = run_moe_fc_helper<float, float>(input_activations,
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
                output_tensor = run_moe_fc_helper<half, half>(input_activations,
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
                output_tensor = run_moe_fc_helper<__nv_bfloat16, __nv_bfloat16>(input_activations,
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
  m.def("run_moe_fc", &run_moe_fc, "moe.");
}
