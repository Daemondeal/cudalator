module AND (
    input  logic x,
    input  logic y,
    output logic z
);
    assign z = x & y;
endmodule : AND

module OR (
    input  logic x,
    input  logic y,
    output logic z
);
    assign z = x | y;
endmodule : OR

module NOT (
    input  logic x,
    output logic z
);
    assign z = ~x;
endmodule : XOR

module XOR (
    input  logic x,
    input  logic y,
    output logic z
);
    logic net1;
    logic net2;
    logic net3;

    AND _g1 (.x(x), .y(y), .z(net1));
    OR  _g2 (.x(x), .y(y), .z(net2));
    NOT _g3 (.x(net1), .z(net3));
    AND _g4 (.x(net2), .y(net3), .z(z));
endmodule : XOR

module full_adder(
    input logic a,
    input logic b,
    input logic cin,
    output logic s,
    output logic cout
);
    logic net1;
    logic net2;
    logic net3;
    logic net4;

    XOR _g1 (.x(a), .y(b), .z(net1));
    XOR _g2 (.x(net1), .y(cin), .z(s));
    AND _g3 (.x(net1), .y(net2), .z(net3));
    AND _g4 (.x(a), .y(b), .z(net4));
    OR  _g5 (.x(net3), .y(net4), .z(cout));

endmodule : full_adder
