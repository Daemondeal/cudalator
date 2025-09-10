module r4mbe_dadda_multiplier_8bit (
    input  logic [ 7:0] i_A,
    input  logic [ 7:0] i_B,
    output logic [15:0] o_result
);

    logic [ 8:0] rows_0     ;
    logic        rows_sign_0;

    logic [ 8:0] rows_1     ;
    logic        rows_sign_1;

    logic [ 8:0] rows_2     ;
    logic        rows_sign_2;

    logic [ 8:0] rows_3     ;
    logic        rows_sign_3;

    logic [ 8:0] rows_4     ;
    logic        rows_sign_4;

    logic [10:0] padded_B;

    logic [16:0] dadda_out1;
    logic [16:0] dadda_out2;
    logic [17:0] full_result;

    assign padded_B = {1'b0, 1'b0, i_B, 1'b0};

    booth_encoder #(
        .DATAW(8)
    ) encoder_0 (
        .i_multiplicand(i_A),
        .i_window      (padded_B[2:0]),
        .o_encoded     (rows_0),
        .o_sign        (rows_sign_0)
    );

    booth_encoder #(
        .DATAW(8)
    ) encoder_1 (
        .i_multiplicand(i_A),
        .i_window      (padded_B[4:2]),
        .o_encoded     (rows_1),
        .o_sign        (rows_sign_1)
    );

    booth_encoder #(
        .DATAW(8)
    ) encoder_2 (
        .i_multiplicand(i_A),
        .i_window      (padded_B[6:4]),
        .o_encoded     (rows_2),
        .o_sign        (rows_sign_2)
    );

    booth_encoder #(
        .DATAW(8)
    ) encoder_3 (
        .i_multiplicand(i_A),
        .i_window      (padded_B[8:6]),
        .o_encoded     (rows_3),
        .o_sign        (rows_sign_3)
    );

    booth_encoder #(
        .DATAW(8)
    ) encoder_4 (
        .i_multiplicand(i_A),
        .i_window      (padded_B[10:8]),
        .o_encoded     (rows_4),
        .o_sign        (rows_sign_4)
    );

    dadda_tree dadda_tree_i (
        .i_rows_0(rows_0),
        .i_rows_1(rows_1),
        .i_rows_2(rows_2),
        .i_rows_3(rows_3),
        .i_rows_4(rows_4),

        .i_signs_0(rows_sign_0),
        .i_signs_1(rows_sign_1),
        .i_signs_2(rows_sign_2),
        .i_signs_3(rows_sign_3),
        .i_signs_4(rows_sign_4),

        .o_sums(dadda_out1),
        .o_carries(dadda_out2)
    );

    assign full_result = dadda_out1 + dadda_out2;
    assign o_result = full_result[15:0];


endmodule : r4mbe_dadda_multiplier_8bit
