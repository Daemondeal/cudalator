module booth_encoder #(
    parameter int DATAW = 7
) (
    input logic [DATAW-1:0] i_multiplicand,
    input logic [2:0] i_window,
    output logic [DATAW:0] o_encoded,
    output logic o_sign
);

    always_comb begin
        if (i_window == 3'b000) o_encoded = '0;
        if (i_window == 3'b001) o_encoded = {1'b0, i_multiplicand[DATAW-1:0]};
        if (i_window == 3'b010) o_encoded = {1'b0, i_multiplicand[DATAW-1:0]};
        if (i_window == 3'b011) o_encoded = {i_multiplicand[DATAW-1:0], 1'b0};
        if (i_window == 3'b100) o_encoded = ~{i_multiplicand[DATAW-1:0], 1'b0};
        if (i_window == 3'b101) o_encoded = ~{1'b0, i_multiplicand[DATAW-1:0]};
        if (i_window == 3'b110) o_encoded = ~{1'b0, i_multiplicand[DATAW-1:0]};
        if (i_window == 3'b111) o_encoded = '1;
    end

    assign o_sign = i_window[2];

endmodule : booth_encoder
