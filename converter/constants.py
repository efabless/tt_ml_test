test_bench_template = """`timescale 1ns / 1ps

module tb_top;
    {in_definitions}
    {out_definitions}

    top dut(
        {in_params},
        {out_params}
    );

    initial begin
        $dumpfile("tb_top.vcd");
        $dumpvars(0, tb_top);
        
        // Wait a bit before starting simulation
        #2;
        
        {assignments}

        $finish;
    end
endmodule
"""
