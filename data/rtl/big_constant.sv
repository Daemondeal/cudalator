module big_constant (
    output logic [127:0] o_constant
);

    assign o_constant = 128'hDEADBEEFDEADBEEFDEADBEEF;

endmodule : adder
