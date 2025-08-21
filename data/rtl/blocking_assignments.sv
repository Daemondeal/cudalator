module test;

int a;
int b;

always begin
    int c;
    a <= 2;
    a = a + 2;
    begin
        int b;
        b = 3;
        b = b + 3;
    end

    a = a + 3;
end

endmodule
