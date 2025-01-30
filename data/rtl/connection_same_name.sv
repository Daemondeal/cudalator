
module sub(input logic x);
endmodule : sub

module top;
logic x;
sub sub_i(.x(x));
endmodule : top
