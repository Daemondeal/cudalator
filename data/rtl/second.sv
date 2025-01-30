
module first(input x, output z);
logic x;
logic z;

assign z = ~x;

endmodule

module second;

logic x;
logic z;

first first_i (x, z);
first first_i2 (x, z);

endmodule
