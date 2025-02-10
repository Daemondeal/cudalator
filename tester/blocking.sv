module blocking;

logic [7:0] x;
logic [7:0] y;

initial
    $monitor("%3t: x: %2h y: %2h", $time, x, y);

always @(x) begin
    // verilator lint_off BLKANDNBLK
    y[7:4] <= 0;
    y = x;
end

initial begin
    x = 8'hFA;
    #10;

    x = 8'hAF;
    #10;

    $finish();
end

endmodule
