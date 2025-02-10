module int_processes(
    input int in,
    output int out
);

int a;
int b;

always @(in, a) begin
    a <= in;
    b <= a + in;
end

always_comb begin
    out = b;
end

endmodule
