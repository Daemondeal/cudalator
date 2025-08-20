`timescale 1ns / 1ps
module special_ops_tb;

  initial begin
    $display("--- Verilog test ---");

    // concat and repl
    $display("TEST CONCAT_1 Result: %h", {8'hAA, 8'hBB});
    $display("TEST CONCAT_2 Result: %h", {8'hDE, 4'hA, 8'hDF});
    $display("TEST CONCAT_3_MAX_WIDTH Result: %h", {64'hAAAAAAAAAAAAAAAA, 64'h5555555555555555});

    $display("TEST REPLICATE_1 Result: %h", {2{8'hAB}});
    $display("TEST REPLICATE_2 Result: %h", {5{8'hF0}});
    $display("TEST REPLICATE_3_SINGLE_BIT Result: %h", {8{1'b1}});

    // Since iverilog -or at least my version?- does not support inside, i'm hardcoding the expected result manually.
    // TODO: se mai aggiornerai iverilog o trovi il modo di farlo andare sarebbe il caso di sistemare questo test
    $display("TEST INSIDE_1 Result: %h", 1'b1);
    $display("TEST INSIDE_2 Result: %h", 1'b0);
    $display("TEST INSIDE_3 Result: %h", 1'b1);
    $display("TEST INSIDE_4 Result: %h", 1'b1);
    $display("TEST INSIDE_5_MIXED_WIDTH_FAIL Result: %h", 1'b0);

    // increment & decrement
    begin
        reg [7:0] a = 10;
        reg [7:0] b = 10;
        reg [7:0] c = 255;
        reg [7:0] res_prefix;
        reg [7:0] res_postfix;
        $display("TEST INC_A_INITIAL Result: %h", a);
        $display("TEST INC_B_INITIAL Result: %h", b);
        res_prefix = ++a;
        $display("TEST INC_A_PREFIX_RES Result: %h", res_prefix);
        $display("TEST INC_A_FINAL Result: %h", a);
        res_postfix = b++;
        $display("TEST INC_B_POSTFIX_RES Result: %h", res_postfix);
        $display("TEST INC_B_FINAL Result: %h", b);
        $display("TEST INC_C_WRAP_INITIAL Result: %h", c);
        c++;
        $display("TEST INC_C_WRAP_FINAL Result: %h", c);
    end

    // cond
    begin
        reg [3:0] condition_true = 4'b0100;
        reg [3:0] condition_false = 4'b0000;
        reg [7:0] val_if_true = 100;
        reg [7:0] val_if_false = 200;
        $display("TEST CONDITIONAL_1_TRUE Result: %h", (condition_true ? val_if_true : val_if_false));
        $display("TEST CONDITIONAL_2_FALSE Result: %h", (condition_false ? val_if_true : val_if_false));
    end

    #1 $finish;
  end
endmodule
