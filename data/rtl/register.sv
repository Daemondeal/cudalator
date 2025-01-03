module register(
    input logic i_clk,
    input logic i_rst_n,
    input logic [7:0] d,
    output logic [7:0] q
);

always @(posedge i_clk, negedge i_rst_n) begin
    if (!i_rst_n) begin
        q <= 0;
    end else begin
        q <= d;
    end
end

endmodule : register

