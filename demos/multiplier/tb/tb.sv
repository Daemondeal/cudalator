module tb;

logic [ 7:0] i_A;
logic [ 7:0] i_B;
logic [15:0] o_result;

r4mbe_dadda_multiplier_8bit dut (
    .*
);

initial begin
    int i;
    // $dumpfile("test.vcd");
    // $dumpvars(0, dut);
    for (i = 0; i < 1000 * 8192; i++) begin
        i_A = $urandom;
        i_B = $urandom;
        #1;
    end
    $finish;
end

endmodule : tb
