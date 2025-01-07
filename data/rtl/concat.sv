module concat;

logic [7:0] x;
logic [7:0] y;

logic [7:0] z;

assign z = {x[4:0], y[4:0]};

endmodule : concat
