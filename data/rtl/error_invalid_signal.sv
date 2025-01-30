module error_invalid_signal;

int x;
int y;

always_comb  begin
    y = x + z;
end

endmodule : error_invalid_signal
