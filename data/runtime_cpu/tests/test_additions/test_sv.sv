module test_sv;
int a;
int b;
int c;

initial begin
    a = 1;
    b = 2;

    c = a+b;

    $display("%d", c);
end

endmodule
