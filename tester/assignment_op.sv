module assignment_op;


initial begin
    logic [1:0] x;
    logic [1:0] y;
    logic [1:0] z;

    x = 2'b00;
    y = 2'b10;
    z = 2'b00;

    $display("x: %2b y: %2b z: %2b, xyz: %6b", x, y, z, {x, y, z});

    {x, y, z} += 6'b001010;

    $display("x: %2b y: %2b z: %2b, xyz: %6b", x, y, z, {x, y, z});

    $finish();

end

endmodule
