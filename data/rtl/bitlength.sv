module bitlength;

logic [15:0] a, b, answer;
logic [16:0] longer_answer;

initial begin
    a = '1;
    b = '1;

    answer = (a + b) >> 1;
    $display("%h + %h = %h", a, b, answer);

    longer_answer = (a + b) >> 1;
    $display("%h + %h = %h", a, b, longer_answer);

    answer = (a + b + 0) >> 1;
    $display("%h + %h = %h", a, b, answer);

    answer = ((a + b) + 0) >> 1;
    $display("%h + %h = %h", a, b, answer);


    a = '1;
    b = 1;
    longer_answer = (b + (b + a)) >> 0;
    $display("%h + %h = %h", a, b, longer_answer);


    $finish();
end

endmodule : bitlength
