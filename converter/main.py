from typing import List

import numpy as np
import torch
from torch import nn

import layers
from constants import test_bench_template


class Model:
    def __init__(self, model: nn.Sequential):
        self.model = model
        self.layers = []

    def __str__(self):
        return '\n'.join(str(layer) for layer in self.layers)

    def parse_layers(self):
        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.Linear):
                self.layers.append(layers.Linear.layer_from(layer, i))
            elif isinstance(layer, nn.ReLU):
                self.layers.append(layers.ReLU(self.model[i - 1].out_features, i))
            else:
                continue

    def forward_range(self, ranges: List[List[float]]):
        start = np.array(ranges)

        for layer in self.layers:
            start = layer.forward_range(start)

    def get_vars(self):
        in_params = [f"in{i}" for i in range(self.layers[0].shape[0])]
        out_params = [f"out{i}" for i in range(self.layers[-1].shape[-1])]

        in_definitions = [f"    input [{self.layers[0].in_bits[i] - 1}:0] {in_params[i]};"
                          for i in range(self.layers[0].shape[0])]

        out_definitions = [f"    output [{self.layers[-1].out_bits[i] - 1}:0] {out_params[i]};"
                           for i in range(self.layers[-1].shape[-1])]

        return in_params, out_params, in_definitions, out_definitions

    def emit(self):
        out = ["`timescale 1ns / 1ps"]

        in_params, out_params, in_definitions, out_definitions = self.get_vars()

        top = [
            f"module top({','.join(in_params)}, {','.join(out_params)});",
            *in_definitions,
            *out_definitions,
        ]

        in_wires = in_params
        out_wires = []

        for i, layer in enumerate(self.layers):
            out.append(layer.emit())

            out_wires = []
            for j in range(layer.shape[-1]):
                top.append(f"    wire [{layer.out_bits[j]}:0] layer_{i}_out_{j};")
                out_wires.append(f"layer_{i}_out_{j}")

            top.append(f"    {layer.name} layer_{i}({','.join(in_wires)}, {','.join(out_wires)});")

            in_wires = out_wires

        assigns = [f"    assign out{i} = {out_wire};" for i, out_wire in enumerate(out_wires)]

        top.extend(assigns)
        top.append("endmodule")

        out.append('\n'.join(top))

        return '\n'.join(out)

    def emit_test_bench(self):
        in_params, out_params, in_definitions, out_definitions = self.get_vars()

        assigns = [f"        assign {i} = 0;" for i in in_params]

        return test_bench_template.format(
            in_params=', '.join(in_params),
            out_params=', '.join(out_params),
            in_definitions='\n    '.join(in_definitions),
            out_definitions='\n    '.join(out_definitions),
            assignments='\n'.join(assigns),
        )


def test():
    simple_model = nn.Sequential(
        nn.Linear(2, 1),
        nn.ReLU(),
    )

    simple_model[0].weight = nn.Parameter(torch.tensor([[1.0, -1.0]]))
    simple_model[0].bias = nn.Parameter(torch.tensor([1.0]))

    model = Model(simple_model)
    model.parse_layers()
    model.forward_range([[1.0, 100.0], [0.0, 1024.0]])

    print(model)
    code = model.emit()

    with open('test.v', 'w') as f:
        f.write(code)

    with open('test_tb.v', 'w') as f:
        f.write(model.emit_test_bench())
def test_quant():
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 1),
                nn.ReLU()
            )
        def forward(self, x):
            x = self.net[1](self.net[0](x))
            return x

    simple_model = SimpleModel()

    simple_model.net[0].weight = nn.Parameter(torch.tensor([[1.0, -1.0]]))
    simple_model.net[0].bias = nn.Parameter(torch.tensor([1.0]))

    x_data = torch.tensor([[1.0, -1.0], [0.0, 1024.0]])

    class QuantizedSimpleModel(nn.Module):
        def __init__(self):
            super(QuantizedSimpleModel, self).__init__()
            self.quant = torch.quantization.QuantStub(),
            self.net = nn.Sequential(
                nn.Linear(2, 1),
                nn.ReLU()
            )
            self.dequant = torch.quantization.DeQuantStub(),

        def forward(self, x):
            x = self.net[1](self.net[0](x))
            return x

    # simple_model_quantized = nn.Sequential(
    #     torch.quantization.QuantStub(),
    #     nn.Linear(2, 1),
    #     nn.ReLU(),
    #     torch.quantization.DeQuantStub()
    # )

    simple_model_quantized = QuantizedSimpleModel()

    #model_quantized = QuantizedSinePredictor(input_size, hidden_size, output_size)

    # Copy weights from unquantized model
    # simple_model_quantized.load_state_dict(simple_model.state_dict())
    simple_model_quantized.net[0].weight = nn.Parameter(torch.tensor([[1.0, -1.0]]))
    simple_model_quantized.net[0].bias = nn.Parameter(torch.tensor([1.0]))

    simple_model_quantized.eval()

    simple_model_quantized.qconfig = torch.ao.quantization.default_qconfig
    simple_model_quantized = torch.ao.quantization.prepare(simple_model_quantized)

    y_data_pred = simple_model_quantized(x_data)

    simple_model_quantized = torch.ao.quantization.convert(simple_model_quantized)

    model = Model(simple_model_quantized.net)

    model.parse_layers()
    model.forward_range([[1.0, 100.0], [0.0, 1024.0]])

    print(model)
    code = model.emit()

    with open('test.v', 'w') as f:
        f.write(code)

    with open('test_tb.v', 'w') as f:
        f.write(model.emit_test_bench())


if __name__ == '__main__':
    test_quant()
