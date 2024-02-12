// Code your testbench here
// or browse Examples
module __adler32__add_two_f32_tb;
  reg clk;
  always #5 clk = ~clk;
  
  reg [31:0] a;
  reg [31:0] b;
  wire [31:0] out;
  
  __adler32__add_two_f32 dut (.clk(clk), .a(a), .b(b), .out(out));
  
  initial begin
    
    clk = 0;
    
    @(negedge clk);
    assign a = 'h4019999a; // 2.4
    assign b = 'h40966666; // 4.7
    
    @(posedge clk);
    @(posedge clk);
    @(posedge clk);
    $display("%h", out); // 7.1
    
    $finish;
  end
  
endmodule