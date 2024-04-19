import numpy as np

from layers.utils import range_to_bits


class Linear:
    @classmethod
    def layer_from(cls, layer, index: int):
        return cls(layer.in_features, layer.out_features, layer.weight.detach().numpy().T, layer.bias.detach().numpy(),
                   index)

    def __init__(self, in_features: int, out_features: int, weight: np.ndarray, bias: np.ndarray, index: int):
        self.in_features = in_features
        self.out_features = out_features

        self.bias = bias
        self.weight = weight

        self.verify_weights()

        self.name = f'layer_{index}_linear_{str(self.in_features)}_{str(self.out_features)}'
        self.shape = (self.in_features, self.out_features)

        self.in_bits, self.out_bits = None, None

    def __str__(self):
        return f'Linear({self.in_features} -> {self.out_features})'

    def verify_weights(self):
        if self.weight is None:
            raise ValueError('Weight is not defined')

        bias_shape = (self.out_features,)
        weight_shape = (self.in_features, self.out_features)

        if self.weight.shape != weight_shape:
            raise ValueError(f'Weight shape is not correct, expected {weight_shape}, got {self.weight.shape}')

        if self.bias is not None and self.bias.shape != bias_shape:
            raise ValueError(f'Bias shape is not correct, expected {bias_shape}, got {self.bias.shape}')

    def forward_range(self, in_range: np.ndarray):
        out_range = np.array([in_range.T[0] @ self.weight, in_range.T[1] @ self.weight])
        out_range = (out_range + self.bias).T

        self.in_bits = [range_to_bits(*r) for r in in_range]
        self.out_bits = [range_to_bits(*r) for r in out_range]

        return out_range

    def emit(self):
        """
        Emit Verilog code for this layer
        :return: Verilog code
        """

        add_bias = [f'add{i} = mul{i} + {self.bias[i]};\n' for i in range(self.out_features)]
        multiply_weight = []

        for i in range(self.out_features):
            for j in range(self.in_features):
                multiply_weight.append(f"mul{i} = mul{i} + in{j} * {self.weight[j][i]};\n")

        in_params = [f"in{i}" for i in range(self.in_features)]
        out_params = [f"out{i}" for i in range(self.out_features)]

        in_definitions = [f"input [{self.in_bits[i] - 1}:0] {in_params[i]};\n"
                          for i in range(self.in_features)]

        out_definitions = [f"output [{self.out_bits[i] - 1}:0] {out_params[i]};\n"
                           for i in range(self.out_features)]

        mul_definition = [f"reg [{self.out_bits[i] - 1}:0] mul{i};\n" for i in range(self.out_features)]
        add_definition = [f"reg [{self.out_bits[i] - 1}:0] add{i};\n" for i in range(self.out_features)]

        assigns = [f"assign out{i} = add{i};\n" for i in range(self.out_features)]

        return f"""
module {self.name}({",".join(in_params)}, {",".join(out_params)});
    {'    '.join(in_definitions)}
    {'    '.join(out_definitions)}
    
    {'    '.join(mul_definition)}
    {'    '.join(add_definition)}
        
    always @(*)
    begin
        {'        '.join(multiply_weight)}
        {'        '.join(add_bias)}
    end
    {'  '.join(assigns)}
endmodule
"""
