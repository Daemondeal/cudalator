module double_int_adder(
    input int a,
    input int b,
    input int c,
    output int sum
);

int interm;

assign interm = a + b;
assign sum = interm + c;

endmodule
