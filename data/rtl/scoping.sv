module scoping;

int outer;

always begin
    automatic int inner = 1;
    begin
        int a;
        int inner_inner = 2;
        a = 2;
    end
end

endmodule
