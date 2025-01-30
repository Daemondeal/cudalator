module select;

logic [7:0] value;
logic [3:0] nibble;
logic [1:0] couple;

assign couple = nibble[1:0];
assign value[3:0] = nibble;

endmodule : select
