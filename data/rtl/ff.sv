module ff(
    input clk,
    input rst,
    input d,
    output q
);

always @(posedge clk) begin
    q <= d;
end

endmodule : ff
