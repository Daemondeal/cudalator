`timescale 1ns / 1ps

// Include the adder module itself
`include "adder.sv"

module tb_adder;

    reg  [7:0] a;
    reg  [7:0] b;
    wire [7:0] c;
    wire       cout;

    // Instantiate the device under test (DUT)
    adder dut (
        .a(a),
        .b(b),
        .c(c),
        .cout(cout)
    );

    initial begin
        $display("--- Verilog Adder Test ---");

        // no carry
        a = 8'd10; b = 8'd5; #1;
        $display("TEST a=%d, b=%d -> c=%h, cout=%b", a, b, c, cout);

        // carry out
        a = 8'd255; b = 8'd1; #1;
        $display("TEST a=%d, b=%d -> c=%h, cout=%b", a, b, c, cout);

        // another carry out
        a = 8'h80; b = 8'h80; #1;
        $display("TEST a=%d, b=%d -> c=%h, cout=%b", a, b, c, cout);

        $finish;
    end

endmodule
