`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: IITDH
// Engineer: Dixshant Jha 
// 
// Create Date: 01/22/2025 07:36:35 PM
// Design Name: 
// Module Name: top_level
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module top_level();
reg clk, rst_n;
reg [7:0] pixels [0:783];
reg [9:0] img_idx;
reg [7:0] data_in;

wire signed [11:0] conv_out_1, conv_out_2, conv_out_3;
wire signed [13:0] conv2_out_1, conv2_out_2, conv2_out_3 , conv2_out_4 , conv2_out_5 , conv2_out_6 , conv2_out_7 , conv2_out_8 , conv2_out_9 ;
wire signed [11:0] max_value_1, max_value_2, max_value_3;
wire signed [11:0] max2_value_1, max2_value_2, max2_value_3, max2_value_4 , max2_value_5 , max2_value_6 , max2_value_7 , max2_value_8 , max2_value_9 ;
wire signed [11:0] fc_out_data;
wire [3:0] decision;
wire valid_out_1, valid_out_2, valid_out_3, valid_out_4, valid_out_5, valid_out_6,valid_out_7 , valid_out_8;

conv1_layer conv1_layer(
  .clk(clk),
  .rst_n(rst_n),
  .data_in(data_in),
  .conv_out_1(conv_out_1),
  .conv_out_2(conv_out_2),
  .conv_out_3(conv_out_3),
  .valid_out_conv(valid_out_1)
);

maxpool_relu #(.CONV_BIT(12), .HALF_WIDTH(12), .HALF_HEIGHT(12), .HALF_WIDTH_BIT(4))
maxpool_relu_1(
  .clk(clk),
  .rst_n(rst_n),
  .valid_in(valid_out_1),
  .conv_out_1(conv_out_1),
  .conv_out_2(conv_out_2),
  .conv_out_3(conv_out_3),
  .max_value_1(max_value_1),
  .max_value_2(max_value_2),
  .max_value_3(max_value_3),
  .valid_out_relu(valid_out_2)
);

conv2_layer conv2_layer(
  .clk(clk),
  .rst_n(rst_n),
  .valid_in(valid_out_2),
  .data_in_0(max_value_1),
  .data_in_1(max_value_2),
  .data_in_2(max_value_3),
  .conv1_out(conv2_out_1),
  .conv2_out(conv2_out_2),
  .conv3_out(conv2_out_3),
  .conv4_out(conv2_out_4),
  .conv5_out(conv2_out_5),
  .conv6_out(conv2_out_6),
  .conv7_out(conv2_out_7),
  .conv8_out(conv2_out_8),
  .conv9_out(conv2_out_9),
  
  .conv_valid_out(valid_out_3)
);

maxpool_relu #(.CONV_BIT(12), .HALF_WIDTH(4), .HALF_HEIGHT(4), .HALF_WIDTH_BIT(3))
maxpool_relu_20(
  .clk(clk),
  .rst_n(rst_n),
  .valid_in(valid_out_3),
  .conv_out_1(conv2_out_1),
  .conv_out_2(conv2_out_2),
  .conv_out_3(conv2_out_3),
  .max_value_1(max2_value_1),
  .max_value_2(max2_value_2),
  .max_value_3(max2_value_3),
  .valid_out_relu(valid_out_4)
);

maxpool_relu #(.CONV_BIT(12), .HALF_WIDTH(4), .HALF_HEIGHT(4), .HALF_WIDTH_BIT(3))
maxpool_relu_21(
  .clk(clk),
  .rst_n(rst_n),
  .valid_in(valid_out_3),
  .conv_out_1(conv2_out_4),
  .conv_out_2(conv2_out_5),
  .conv_out_3(conv2_out_6),
  .max_value_1(max2_value_4),
  .max_value_2(max2_value_5),
  .max_value_3(max2_value_6),
  .valid_out_relu(valid_out_5)
);

maxpool_relu #(.CONV_BIT(12), .HALF_WIDTH(4), .HALF_HEIGHT(4), .HALF_WIDTH_BIT(3))
maxpool_relu_22(
  .clk(clk),
  .rst_n(rst_n),
  .valid_in(valid_out_3),
  .conv_out_1(conv2_out_7),
  .conv_out_2(conv2_out_8),
  .conv_out_3(conv2_out_9),
  .max_value_1(max2_value_7),
  .max_value_2(max2_value_8),
  .max_value_3(max2_value_9),
  .valid_out_relu(valid_out_6)
);

fully_connected #(.INPUT_NUM(144), .OUTPUT_NUM(10), .DATA_BITS(8))
fully_connected(
  .clk(clk),
  .rst_n(rst_n),
  .valid_in(valid_out_6),
  .data_in_1(max2_value_1),
  .data_in_2(max2_value_2),
  .data_in_3(max2_value_3),
  .data_in_4(max2_value_4),
  .data_in_5(max2_value_5),
  .data_in_6(max2_value_6),
  .data_in_7(max2_value_7),
  .data_in_8(max2_value_8),
  .data_in_9(max2_value_9),
  .data_out(fc_out_data),
  .valid_out_fc(valid_out_7)
);

comparator comparator(
  .clk(clk),
  .rst_n(rst_n),
  .valid_in(valid_out_7),
  .data_in(fc_out_data),
  .decision(decision),
  .valid_out(valid_out_8)
);

always #5 clk = ~clk;

// Read image text file
initial begin
  $readmemh("/media/harsh/c98020b5-0d6a-4055-87fd-8c8b7a337914/MTP_PROJECT/verilog_codes/output_image.mem", pixels);
  clk <= 1'b0;
  rst_n <= 1'b1;
  #3
  rst_n <= 1'b0;
  #3
  rst_n <= 1'b1;
end

always @(posedge clk) begin
  if(~rst_n) begin
    img_idx <= 0;
  end else begin
    if(img_idx < 10'd784) begin
      img_idx <= img_idx + 1'b1;
    end
    data_in <= pixels[img_idx];
  end
end


endmodule
