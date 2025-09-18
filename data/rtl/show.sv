module show;


logic [7:0] a;
logic [7:0] b;
logic [7:0] c;
logic sel;
always_comb begin
    if (sel)
        c = a + b;
    else
        c = a - b;
end

endmodule : simple
