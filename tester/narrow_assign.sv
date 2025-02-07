module narrow_assign;
initial begin
    automatic logic [31:0] a = 32'hDEADBEEF;
    automatic logic [15:0] b = 16'hAAAA;

    b = a;
    $display("b: %0h, a: %0h", b, a);
    $finish();
end
endmodule
