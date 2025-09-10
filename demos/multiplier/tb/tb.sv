module tb;

logic [ 7:0] i_A;
logic [ 7:0] i_B;
logic [15:0] o_result;

r4mbe_dadda_multiplier_8bit dut (
    .*
);

initial begin
    $dumpfile("test.vcd");
    $dumpvars(0, dut);
    i_A = 1;
    i_B = 2;
    #10;
    $finish;
end

endmodule : tb
