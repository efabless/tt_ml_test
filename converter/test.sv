`timescale 1ns / 1ps

module layer_0_linear_1_4_weights(
    input wire [3:0] indx,
    output reg [7:0] w
);
    always_comb begin
        case (indx)
            4'd0 : w = 127;
            4'd1 : w = -127;
            4'd2 : w = 127;
            4'd3 : w = -127;

        endcase
    end
endmodule
        

module layer_0_linear_1_4_biases(
    input wire [3:0] indx,
    output reg [7:0] b
);
    always_comb begin
        case (indx)
            4'd0 : b = 0.5;
            4'd1 : b = -0.5;
            4'd2 : b = 0.5;
            4'd3 : b = -0.5;

        endcase
    end
endmodule

module layer_0_linear_1_4(
    clk, 
    rst_n, 
    start, 
    done,
    in0, 
    out0,out1,out2,out3
);
    input clk;
    input rst_n;
    input start;
    output done;
    
    reg[4:0] seq;
    done = (seq == 5'd15);
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

        layer_0_linear_1_4_weights WEIGHTS (.indx(seq), .w(w));
        layer_0_linear_1_4_biases BIASES (.indx(seq), .b(b));

    // The MAC
    wire [9:0] mac = in0 * w + b;

    always_ff(posedge clk)
        case (seq)
            5'd0 : out0 <= mac;
            5'd1 : out0 <= mac;
            5'd2 : out0 <= mac;
            5'd3 : out0 <= mac;

        endcase
endmodule


module layer_1_relu_4(in0,in1,in2,in3, out0,out1,out2,out3);
    input [-1:0] in0;
    input [-1:0] in1;
    input [-1:0] in2;
    input [-1:0] in3;

    output reg [-1:0] out0;
    output reg [-1:0] out1;
    output reg [-1:0] out2;
    output reg [-1:0] out3;

                
    always @(*) 
    begin
        out0 = in0 > 0 ? in0 : 0;
        out1 = in1 > 0 ? in1 : 0;
        out2 = in2 > 0 ? in2 : 0;
        out3 = in3 > 0 ? in3 : 0;

    end
endmodule


module layer_2_linear_4_1_weights(
    input wire [3:0] indx,
    output reg [7:0] w
);
    always_comb begin
        case (indx)
            4'd0 : w = 127;
            4'd0 : w = -127;
            4'd0 : w = 127;
            4'd0 : w = -127;

        endcase
    end
endmodule
        

module layer_2_linear_4_1_biases(
    input wire [3:0] indx,
    output reg [7:0] b
);
    always_comb begin
        case (indx)
            4'd0 : b = 0.5;

        endcase
    end
endmodule

module layer_2_linear_4_1(
    clk, 
    rst_n, 
    start, 
    done,
    in0,in1,in2,in3, 
    out0
);
    input clk;
    input rst_n;
    input start;
    output done;
    
    reg[4:0] seq;
    done = (seq == 5'd15);
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

        layer_2_linear_4_1_weights WEIGHTS (.indx(seq), .w(w));
        layer_2_linear_4_1_biases BIASES (.indx(seq), .b(b));

    // The MAC
    wire [9:0] mac = in0 * w + b;

    always_ff(posedge clk)
        case (seq)
            5'd0 : out0 <= mac;

        endcase
endmodule

module top(in0, out0);
    input [1:0] in0;
    output [-1:0] out0;
    wire [0:0] layer_0_out_0;
    wire [0:0] layer_0_out_1;
    wire [0:0] layer_0_out_2;
    wire [0:0] layer_0_out_3;
    layer_0_linear_1_4 layer_0(in0, layer_0_out_0,layer_0_out_1,layer_0_out_2,layer_0_out_3);
    wire [0:0] layer_1_out_0;
    wire [0:0] layer_1_out_1;
    wire [0:0] layer_1_out_2;
    wire [0:0] layer_1_out_3;
    layer_1_relu_4 layer_1(layer_0_out_0,layer_0_out_1,layer_0_out_2,layer_0_out_3, layer_1_out_0,layer_1_out_1,layer_1_out_2,layer_1_out_3);
    wire [0:0] layer_2_out_0;
    layer_2_linear_4_1 layer_2(layer_1_out_0,layer_1_out_1,layer_1_out_2,layer_1_out_3, layer_2_out_0);
    assign out0 = layer_2_out_0;
endmodule