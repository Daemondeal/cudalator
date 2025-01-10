module sub1;
int x;
int y;

always_comb begin
    y = x * 2;
end
endmodule

module sub2;
sub1 sub1_i();
endmodule

module nested;

for (genvar i = 0; i < 2; i++) begin : gen_test
    for (genvar j = 0; j < 2; j++) begin : gen_test_2
        sub2 sub2_i();
    end
end

endmodule
