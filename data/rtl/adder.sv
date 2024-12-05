module adder (
    input logic [7:0] a,
    input logic [7:0] b,
    output logic [7:0] c,
    output logic cout
);
    logic [8:0] full_sum;

    always_comb begin
        full_sum = a + b;
    end

    assign c = full_sum[7:0];
    assign cout = full_sum[8];

endmodule : adder
