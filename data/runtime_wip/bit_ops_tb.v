module bit_ops_tb;
    reg [7:0] a;
    reg [15:0] b;
    reg [7:0] c;
    // 28 bit
    reg [27:0] vec28;
    // 35 bit
    reg [34:0] vec35;
    // 100 bit
    reg [99:0] vec100;
    // 64 bit
    reg [63:0] vec64;
    reg [31:0] vec32_1;
    reg [31:0] vec32_2;
    reg [31:0] vec32_3;
    // altre dimensioni?
    // in realtà dovresti farlo più parametrico ma ci pensiamo poi
    integer fd;

    initial begin
        fd = $fopen("results.txt", "w");
        // [dimension]'[d/h/o][number]
        // ***************** vec28 *****************
        vec28 = 28'h0FFFFFFF;  // Tutti 1
        $fwrite(fd, "vec28\t%h\t%b\t%d\n", vec28, vec28, vec28);

        // Addition
        vec28 = vec28 + 1;
        $fwrite(fd, "vec28+1\t%h\t%b\t%d\n", vec28, vec28, vec28);
        vec28 = 28'h0FFFFFFF;

        // Subtraction
        vec28 = vec28 - vec28;
        $fwrite(fd, "vec28-vec28\t%h\t%b\t%d\n", vec28, vec28, vec28);
        vec28 = 28'h0FFFFFFF;

        // And
        vec28 = vec28 & 28'h0000000;
        $fwrite(fd, "vec28&0000000\t%h\t%b\t%d\n", vec28, vec28, vec28);
        vec28 = 28'h0FFFFFFF;

        // Or
        vec28 = vec28 | vec28;
        $fwrite(fd, "vec28|vec28\t%h\t%b\t%d\n", vec28, vec28, vec28);
        vec28 = 28'h0FFFFFFF;

        // ***************** vec35 *****************
        vec35 = 35'h3FFFFFFFF;  // 011FFFF..
        $fwrite(fd, "vec35\t%h\t%b\t%d\n", vec35, vec35, vec35);

        // Add
        vec35 = vec35 + 1;
        $fwrite(fd, "vec35+1\t%h\t%b\t%d\n", vec35, vec35, vec35);
        vec35 = 35'h3FFFFFFFF;

        // Vorrei testare di nuovo la maschera quindi magari nuova add che vada over
        vec35 = vec35 + 5;
        $fwrite(fd, "vec35+5\t%h\t%b\t%d\n", vec35, vec35, vec35);
        vec35  = 35'h3FFFFFFFF;
        // Sub ..
        // And ..
        // Or ..

        // ***************** vec100 *****************
        vec100 = 100'h123456789ABCDEF;
        $fwrite(fd, "vec100\t%h\t%b\t%d\n", vec100, vec100, vec100);

        // Add
        vec100 = vec28 + vec35;
        $fwrite(fd, "%d+%d\t%h\t%b\t%d\n", vec28, vec35, vec100, vec100, vec100);

        // ***************** vec64 *****************
        vec32_1 = 32'h12345678;
        vec32_2 = 32'h12345678;
        vec32_3 = vec32_1 + vec32_2;
        $fwrite(fd, "%h", vec32_3);

        a = 8'hFF;
        b = 16'hFF00;
        c = a + b;
        $fwrite(fd, "\na=%h\tb=%h\tc=%h\n", a, b, c);
        $fclose(fd);
        $finish;
    end
endmodule
