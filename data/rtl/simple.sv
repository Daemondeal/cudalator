module simple (
    input logic clk,
    input logic rst,
    output reg [7:0] value
);

wire u;

assign u = value[0];
logic x, y, z;

always_ff @(posedge clk) begin
    if (rst) value <= 0;
    else value <= value + 1;
end


always_comb
    y = ~x | value[0];

always_ff @(posedge x) begin
    z <= ~x;
end


// assign value = 12;

endmodule : simple
