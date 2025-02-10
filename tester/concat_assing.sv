module concat_assign;

logic [7:0] a;
logic [7:0] b;
logic [7:0] c;
logic [7:0] d;

logic [31:0] whole;


initial begin
    whole = 32'hDEADBEEF;

    {{a, b}, c, d} = whole;


    $display("%h %h %h %h", a, b, c, d);
    $finish();
end

endmodule
