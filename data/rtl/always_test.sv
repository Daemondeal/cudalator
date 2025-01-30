module always_test;

logic clk;

logic a;
logic b;
logic c;

logic q;
logic d;

always @(a, b, c) begin
    if (b) begin
        c <= 0;
    end
end

always_ff @(posedge clk)
    q <= d;

always_comb
    d = ~a;

endmodule : always_test
