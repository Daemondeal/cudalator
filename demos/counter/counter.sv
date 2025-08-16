module counter (
    input logic i_clk,
    input logic i_rst_n,
    input logic i_en,
    output logic [7:0] o_cnt,
    output logic o_tc
);

assign o_tc = (o_cnt == 8'd17);

always @(posedge i_clk) begin
    if (!i_rst_n) begin
        o_cnt <= '0;
    end else if (i_en) begin
        if (o_tc) begin
            o_cnt <= '0;
        end else begin
            o_cnt <= o_cnt + 8'd1;
        end
    end
end

endmodule : counter
