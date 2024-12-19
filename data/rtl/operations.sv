module operations;

    logic [7:0] a;
    logic [7:0] b;
    logic [7:0] c;

    assign c = a + b;
    assign c = a - b;
    assign c = a * b;
    assign c = a / b;
    assign c = a & b;
    assign c = a | b;
    assign c = a ^ b;
    assign c = a << b;
    assign c = a >> b;

    assign c = -a;
    assign c = ~a;
    assign c = &a;
    assign c = |a;
    assign c = ^a;

endmodule : operations
