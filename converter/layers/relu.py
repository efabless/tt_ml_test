import numpy as np

from layers.utils import range_to_bits


class ReLU:

    def __init__(self, shape: int, index: int):
        self.shape = (shape,)
        self.name = f'layer_{index}_relu_{shape}'
        self.in_bits, self.out_bits = None, None

    def __str__(self):
        return f'ReLU({self.shape})'

    def forward_range(self, in_range: np.ndarray):
        out_range = np.maximum(in_range, 0)

        self.in_bits = [range_to_bits(*in_range[i]) for i in range(self.shape[0])]
        self.out_bits = [range_to_bits(*out_range[i]) for i in range(self.shape[0])]

        return out_range

    def emit(self):
        """
        Emit the Verilog code for this layer
        :return:
        """

        relu_code = [f'out{i} = in{i} > 0 ? in{i} : 0;\n' for i in range(self.shape[0])]

        in_params = [f"in{i}" for i in range(self.shape[0])]
        out_params = [f"out{i}" for i in range(self.shape[0])]

        in_definitions = [f"input [{self.in_bits[i] - 1}:0] {in_params[i]};\n"
                          for i in range(self.shape[0])]

        out_definitions = [f"output reg [{self.out_bits[i] - 1}:0] {out_params[i]};\n"
                           for i in range(self.shape[0])]

        return f"""
module {self.name}({",".join(in_params)}, {",".join(out_params)});
    {'    '.join(in_definitions)}
    {'    '.join(out_definitions)}
                
    always @(*) 
    begin
        {"        ".join(relu_code)}
    end
endmodule
"""
