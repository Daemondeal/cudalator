module sub(input logic x, output logic y);
    assign y = ~x;
endmodule : sub

module top;
logic a;
logic b;

sub sub_i(.x(a), .y(b));
endmodule : top
