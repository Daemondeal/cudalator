// iverilog -o tb test_semantics.v
// vvp tb

module test_semantics;

    parameter a_bit = 64;
    parameter b_bit = 64;

    logic [a_bit-1:0] a;
    logic [b_bit-1:0] b;
    // worst case
    wire [(a_bit+b_bit)-1:0] result;

    assign result = a * b;

    initial begin
        a = 64'hFFFFFFFFFFFFFF;
        b = 64'hFFFFFFFFFFFFFFFF;
        $display("a*b=%h", a * b);
        $display("a = %h | %d", a, a);
        $display("b = %h | %d", b, b);
        $display("result = %h | %d", result, result);
        $display("result length = %0d bits", $clog2(result + 1));

        $finish;
    end
endmodule
