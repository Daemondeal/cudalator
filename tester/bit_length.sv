module bit_length;

logic [3:0] a;
logic [5:0] b;
logic [15:0] c;

initial begin
    a = 4'hF;
    b = 6'hA;

    logic [63:0] x = 64'hFFFFFFFFFFFFFF;
    logic [63:0] y = 64'hFFFFFFFFFFFFFFFF;

    $display("ab=%h", x*y);

    $display("L(a*b)=%0d", $size(a*b));

    // $display("a*b=%h", a*b);
    // c = {a**b};
    // $display("a**b=%h", c);
    // c = a**b;
    // $display("c=%h", c);

    $finish();
end

endmodule
