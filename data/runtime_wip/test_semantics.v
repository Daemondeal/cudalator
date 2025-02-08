// iverilog -o tb test_semantics.v
// vvp tb

module test_semantics;

    parameter N = 64;

    reg  [  N-1:0] a;
    reg  [  N-1:0] b;


    wire [2*N-1:0] result;

    assign result = a * b;

    initial begin

        a = 64'h123456789ABCDEF0;
        b = 64'hFEDCBA9876543210;
        // Potrei anche toglierlo
        #10;

        $display("a = %h", a);
        $display("b = %h", b);
        $display("result = %h", result);

        $finish;
    end
endmodule
