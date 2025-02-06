module nested_scope;

int a;

always_comb begin
    int a = 15;
    if (1) begin
        int a = 16;
    end
end

endmodule : nested_scope
