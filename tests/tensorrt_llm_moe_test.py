import torch
import torch.nn.functional as F
import numpy as np
import unittest
from typing import Dict, Any, List, Tuple
from parameterized import parameterized

class TestMoe(unittest.TestCase):
    """Test class for MoE (Mixture of Experts) model"""

    # Test configuration parameters
    DEFAULT_TEST_CONFIGS = {
        'rows': [2, 16, 512, 2048],
        'ks': [2, 4],
        'experts_list': [32],
        'hidden_sizes': [1024, 2048],
        'inter_sizes': [4096],
    }

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize test class setup"""
        import tensorrt_llm_moe
        cls.trt_llm_fused_moe = tensorrt_llm_moe.trt_llm_fused_moe
        torch.manual_seed(1)

    @staticmethod
    def create_random_cuda_tensor(shape: List[int], dtype: torch.dtype, mean: float = 0, std: float = 1) -> torch.Tensor:
        """Create a random CUDA tensor

        Args:
            shape: Tensor shape
            dtype: Data type
            mean: Mean value
            std: Standard deviation

        Returns:
            torch.Tensor: Randomly initialized CUDA tensor
        """
        return torch.empty(shape, dtype=dtype, device="cuda").normal_(mean, std)

    @staticmethod
    def basic_moe_fc(activations: torch.Tensor, expert_for_row: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Basic MoE forward computation

        Args:
            activations: Input activations
            expert_for_row: Expert indices for each row
            weights: Expert weights

        Returns:
            torch.Tensor: Computation result
        """
        res = torch.zeros(size=[activations.shape[0], weights.shape[-1]], 
                         dtype=activations.dtype, device='cuda')
        for row in range(activations.shape[0]):
            row_expert = expert_for_row[row]
            torch.matmul(activations[row], weights[row_expert], 
                        out=res[row : row + 1, :])
        return res

    @staticmethod
    def apply_activation(inp: torch.Tensor, act_str: str) -> torch.Tensor:
        """Apply activation function

        Args:
            inp: Input tensor
            act_str: Activation function name

        Returns:
            torch.Tensor: Activated tensor
        """
        activation_map = {
            "identity": lambda x: x,
            "silu": torch.nn.SiLU(),
            "relu": torch.nn.ReLU(),
            "gelu": lambda x: torch.nn.GELU(approximate="tanh")(x)
        }
        
        if act_str not in activation_map:
            raise ValueError(f"Unsupported activation: {act_str}")
            
        return activation_map[act_str](inp)

    def generate_inputs(self, num_rows: int, active_rows: int, 
                       hidden_size: int, num_experts: int, 
                       dtype: torch.dtype, quant_type: torch.dtype) -> Dict[str, torch.Tensor]:
        """Generate test input data

        Args:
            num_rows: Number of rows
            active_rows: Number of active rows
            hidden_size: Hidden layer size
            num_experts: Number of experts
            dtype: Data type
            quant_type: Quantization type

        Returns:
            Dict[str, torch.Tensor]: Dictionary of input data
        """
        return {
            "input_activations": self.create_random_cuda_tensor([num_rows, hidden_size], dtype, mean=0, std=0.01),
            "gating_output": self.create_random_cuda_tensor([num_rows, num_experts], dtype)
        }

    def generate_weights(self, hidden_size: int, inter_size: int, 
                        num_experts: int, dtype: torch.dtype, 
                        quant_type: torch.dtype) -> Dict[str, torch.Tensor]:
        """Generate weight data

        Args:
            hidden_size: Hidden layer size
            inter_size: Intermediate layer size
            num_experts: Number of experts
            dtype: Data type
            quant_type: Quantization type

        Returns:
            Dict[str, torch.Tensor]: Dictionary of weight data
        """
        weights = {}
        for prefix in ['fc1', 'fc2']:
            if prefix == 'fc1':
                shape = [num_experts, hidden_size, inter_size]
            else:
                shape = [num_experts, inter_size, hidden_size]
                
            ref_weights = self.create_random_cuda_tensor(shape, dtype, mean=0, std=0.01)
            weights[f'{prefix}_expert_weights_for_ref'] = ref_weights
            weights[f'{prefix}_expert_weights_for_ft'] = ref_weights.transpose(1, 2).reshape(*shape)
            
        return weights

    def run_ft_moe(self, input_dict: Dict[str, torch.Tensor], 
                   active_rows: int, k: int, activation_str: str) -> torch.Tensor:
        """Run FastTransformer MoE

        Args:
            input_dict: Input data dictionary
            active_rows: Number of active rows
            k: Top-k value
            activation_str: Activation function name

        Returns:
            torch.Tensor: Computation result
        """
        return self.trt_llm_fused_moe(
            input_dict["input_activations"],
            input_dict["gating_output"],
            input_dict["fc1_expert_weights_for_ft"],
            activation_str,
            input_dict["fc2_expert_weights_for_ft"],
            active_rows,
            k
        )

    def run_ref_moe(self, input_dict: Dict[str, torch.Tensor], 
                    k: int, activation_str: str) -> torch.Tensor:
        """Run reference MoE implementation

        Args:
            input_dict: Input data dictionary
            k: Top-k value
            activation_str: Activation function name

        Returns:
            torch.Tensor: Computation result
        """
        input_dict["gating_output"] = input_dict["gating_output"].to(torch.float32)
        gates = F.softmax(input_dict["gating_output"], dim=-1).to(input_dict["gating_output"].dtype)
        expert_scales, experts_for_row = torch.topk(gates, k, dim=-1)
        expert_scales /= expert_scales.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(input_dict["input_activations"])
        
        for k_idx in range(k):
            current_expert_scales = expert_scales[:, k_idx].unsqueeze(1)
            current_experts_for_row = experts_for_row[:, k_idx]

            fc1_out = self.basic_moe_fc(
                input_dict["input_activations"],
                current_experts_for_row,
                input_dict["fc1_expert_weights_for_ref"]
            )
            
            activated = self.apply_activation(fc1_out, activation_str)
            
            fc2_out = self.basic_moe_fc(
                activated,
                current_experts_for_row,
                input_dict["fc2_expert_weights_for_ref"]
            )
            
            output += current_expert_scales * fc2_out
            
        return output

    def run_moe_test(self, dtype: torch.dtype, quant_type: torch.dtype,
                     rtol: float, atol: float, activation_str: str = "gelu",
                     test_configs: Dict[str, List] = None) -> None:
        """Run MoE test

        Args:
            dtype: Data type
            quant_type: Quantization type
            rtol: Relative tolerance
            atol: Absolute tolerance
            activation_str: Activation function name
            test_configs: Test configuration parameters
        """
        torch.cuda.empty_cache()
        
        if test_configs is None:
            test_configs = self.DEFAULT_TEST_CONFIGS
            
        for hidden_size in test_configs['hidden_sizes']:
            for inter_size in test_configs['inter_sizes']:
                for experts in test_configs['experts_list']:
                    weights = self.generate_weights(hidden_size, inter_size, experts, dtype, quant_type)
                    
                    for row in test_configs['rows']:
                        for k in test_configs['ks']:
                            if k > experts:
                                continue
                                
                            input_dict = self.generate_inputs(row, row, hidden_size, experts, dtype, quant_type)
                            input_dict.update(weights)
                            
                            act_output = self.run_ft_moe(input_dict, row, k, activation_str)
                            ref_output = self.run_ref_moe(input_dict, k, activation_str)
                            
                            msg = f"MoE test failed: rows={row}, experts={experts}, k={k}, hidden_size={hidden_size}, inter_size={inter_size}"
                            torch.testing.assert_close(
                                act_output, ref_output,
                                rtol=rtol, atol=atol,
                                msg=msg,
                                check_dtype=False
                            )

    @parameterized.expand([
        ("fp32_silu", torch.float32, torch.float32, 1e-3, 1e-5),
        ("fp16_silu", torch.float16, torch.float16, 1e-1, 1e-2),
        ("bf16_silu", torch.bfloat16, torch.bfloat16, 1e-1, 1e-2),
    ])
    def test_moe(self, name: str, dtype: torch.dtype, quant_type: torch.dtype,
                 rtol: float, atol: float) -> None:
        """Parameterized MoE test

        Args:
            name: Test name
            dtype: Data type
            quant_type: Quantization type
            rtol: Relative tolerance
            atol: Absolute tolerance
        """
        self.run_moe_test(
            dtype=dtype,
            quant_type=quant_type,
            rtol=rtol,
            atol=atol,
            activation_str="silu"
        )

if __name__ == '__main__':
    unittest.main()
