module static_test;

int a;
int b;
int x;
int y;
int u;


initial
    $monitor("[%3t] a: %3d b: %3d x: %3d y: %3d u: %3d", $time, a, b, x, y, u);

always_comb begin
    static int z = 0;

    a = x * 2;
    b = y * 3;

    z += 1;
    u = z;
end

// assign y = x * 2;
always_comb
    y = x * 2;

initial begin
    x = 0;
    #100;

    x = 1;
    #100;

    x = 2;
    #100;

    $finish();
end

endmodule
