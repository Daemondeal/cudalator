module int_mux(
    input int a,
    input int b,
    input int sel,
    output int out
);

always_comb begin
    if (sel == 0) begin
        out = a;
    end else begin 
        out = b;
    end
end


endmodule : int_mux
