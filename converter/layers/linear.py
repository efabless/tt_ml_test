import numpy as np
import torch

from layers.utils import range_to_bits, int_repr


class Linear:
    @classmethod
    def layer_from(cls, layer, index: int):
        return cls(layer.in_features, layer.out_features, layer.weight.detach().numpy().T, layer.bias.detach().numpy(),
        # return cls(layer.in_features, layer.out_features, np.array(torch.int_repr(layer.weight())), np.array(torch.tensor([[0.0, 0.0]])),
                   index)

    @classmethod
    def layer_from_q(cls, layer, index: int):
        # return cls(layer.in_features, layer.out_features, layer.weight.detach().numpy().T, layer.bias.detach().numpy(),
        return cls(layer.in_features, layer.out_features, np.array(torch.int_repr(layer.weight())), layer.bias(),
                   index)

    def __init__(self, in_features: int, out_features: int, weight: np.ndarray, bias: np.ndarray, index: int,
                 scale: float = 0.1, zero = 127):
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.zero = zero
        self.bias = np.array([self.quant(b.detach().numpy()) for b in bias])
        self.weight = weight
        self.verify_weights()
        self.name = f'layer_{index}_linear_{str(self.in_features)}_{str(self.out_features)}'
        self.shape = (self.in_features, self.out_features)
        self.in_bits, self.out_bits = [], []

    def __str__(self):
        return f'Linear({self.in_features} -> {self.out_features})'

    def quant(self, n: float):
        return int( n / self.scale + self.zero )

    def dequant(self, n: int):
        return (n - self.zero) * self.scale

    def verify_weights(self):
        if self.weight is None:
            raise ValueError('Weight is not defined')

        bias_shape = (self.out_features,)
        # weight_shape = (self.in_features, self.out_features)
        weight_shape = (self.out_features, self.in_features)

        if self.weight.shape != weight_shape:
            raise ValueError(f'Weight shape is not correct, expected {weight_shape}, got {self.weight.shape}')

        if self.bias is not None and self.bias.shape != bias_shape:
            raise ValueError(f'Bias shape is not correct, expected {bias_shape}, got {self.bias.shape}')

    def forward_range(self, in_range: np.ndarray):
        if len(in_range[0]) == 1:
            out_range = np.array( [np.sum(self.quant(in_range[0]) * self.weight), np.sum(self.quant(in_range[1]) * self.weight)] )
        else:
            out_range = np.array([[self.quant(i) for i in in_range[0]] @ self.weight.T, [self.quant(i) for i in in_range[1]] @ self.weight.T])
        # out_range = np.array([in_range[0].T @ self.weight, in_range[1].T @ self.weight])
        out_range = [out_range[0] + self.bias, out_range[1] + self.bias]
        # out_range = (out_range + self.bias).T

        self.out_bits = []
        self.in_bits = []
        for r in range(len(in_range[0])):
            self.in_bits.append(range_to_bits(in_range[0][r],in_range[1][r]))
        for r in range(len(out_range[0])):
            self.out_bits.append(range_to_bits(out_range[0][r],out_range[1][r]))

        return out_range

    def emit(self):
        """
        Emit Verilog code for this layer
        :return: Verilog code
        """

        add_bias = [f"add{i} = mul{i} + {self.bias[i]};\n" for i in range(self.out_features)]
        biases = [f"4'd{i} : b = {self.bias[i]};\n" for i in range(self.out_features)]
        multiply_weight = []
        weights = []

        for i in range(self.out_features):
            for j in range(self.in_features):
                # multiply_weight.append(f"mul{i} = mul{i} + in{j} * {self.weight[j][i]};\n")
                multiply_weight.append(f"mul{i} = mul{i} + in{j} * {self.weight[i][j]};\n")
                weights.append(f"4'd{i*self.out_features+j} : w = {self.weight[i][j]};\n")

        in_params = [f"in{i}" for i in range(self.in_features)]
        out_params = [f"out{i}" for i in range(self.out_features)]

        in_definitions = [f"input [{self.in_bits[i] - 1}:0] {in_params[i]};\n"
                          for i in range(self.in_features)]

        out_definitions = [f"reg [{self.out_bits[i] - 1}:0] {out_params[i]};\n"
                           for i in range(self.out_features)]

        mul_definition = [f"reg [{self.out_bits[i] - 1}:0] mul{i};\n" for i in range(self.out_features)]
        add_definition = [f"reg [{self.out_bits[i] - 1}:0] add{i};\n" for i in range(self.out_features)]

        assigns = [f"5'd{i} : out{i} <= mac;\n" for i in range(self.out_features)]

        return f"""
module {self.name}_weights(
    input wire [3:0] indx,
    output reg [7:0] w
);
    always_comb begin
        case (indx)
            {'            '.join(weights)}
        endcase
    end
endmodule
        

module {self.name}_biases(
    input wire [3:0] indx,
    output reg [7:0] b
);
    always_comb begin
        case (indx)
            {'            '.join(biases)}
        endcase
    end
endmodule

module {self.name}(
    clk, 
    rst_n, 
    start, 
    done,
    {",".join(in_params)}, 
    {",".join(out_params)}
);
    input clk;
    input rst_n;
    input start;
    output done;
    
    reg[4:0] seq;
    assign done = (seq == 5'd15);
    wire [7:0] w;
    wire [7:0] b;

    // The CU
    localparam [2:0] ST_IDLE, ST_CALC;
    reg[2:0] state, nstate;
    always_comb begin
        nstate = state;
        case(state)
            ST_IDLE: if(statrt) nstate = ST_CALC;
            ST_CALC: if(done) nstate = ST_IDLE;
        endcase
    end
    always_ff(posedge clk or negedge rst_n)
        if(!rst_n)
            state <= ST_IDLE;
        else
            state <= nstate;

    // The sequencer
    always_ff(posedge clk or negedge rst_n)
        if(!rst_n)
            seq <= 'd0;
        else if (state == ST_IDLE)
            seq <= 'd0;
        else if (state == ST_CALC)
            seq <= seq + 1;

        {self.name}_weights WEIGHTS (.indx(seq), .w(w));
        {self.name}_biases BIASES (.indx(seq), .b(b));

    // The MAC
    wire [9:0] mac = in0 * w + b;

    always_ff(posedge clk)
        case (seq)
            {'            '.join(assigns)}
        endcase
endmodule
"""
