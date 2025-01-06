module for_stmt;

int x;
int y;
int z;

always_comb begin
    for (int i = 0; i < 10; i = i + 1) begin
        x = x + 2;
    end
end

endmodule : for_stmt
