module statements;

int x;
int y;
int z;

always_comb begin
    for (x = 0; y < 2; x = x + 1) begin
        x = y;
    end
end

/*
always_comb begin
    x = y + z;
    x <= y + z;

    if (y) begin
        x = y;
        x = z;
    end

    if (y) begin
        x = z;
    end else if (!y) begin
        x = y;
    end else begin
        x = 0;
    end

    while (y) begin
        x = ~y;
    end

    do begin
        x = ~y;
    end while (y);

    repeat(10) begin
        x = y + 1;
    end

    forever begin
        x = y;
        break;
        continue;
    end

    for (x = 0; y < 2; x = x + 1) begin
        x = y;
    end
end
*/

endmodule : statements
