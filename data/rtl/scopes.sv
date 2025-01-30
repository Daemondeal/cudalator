module scopes;

int x;
int y;
int z;

always_comb begin
    x = z + y;

    begin
        int u;
        u = y;
        if (u) begin
            int m = z;
            int n = m;
            x = m + z;
        end
    end
end

endmodule : scopes
