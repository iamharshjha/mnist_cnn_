`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: IITDH
// Engineer: Dixshant 
// 
// Create Date: 01/22/2025 01:33:50 PM
// Design Name: 
// Module Name: conv2_layer
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


module conv2_layer (
    input clk,
    input rst_n,
    input valid_in,
    input signed [11:0] data_in_0, data_in_1, data_in_2,  // 3 channels, 12-bit input data
    output signed [13:0] conv1_out, conv2_out, conv3_out,conv4_out , conv5_out , conv6_out , conv7_out , conv8_out , conv9_out,   // 9 outputs from pointwise convolution
    output conv_valid_out    // Valid signal for output
);

    // Buffer signals for the 3 input channels
    wire signed [11:0] buffer_0_out [0:24]; // 25 data points for channel 0
    wire signed [11:0] buffer_1_out [0:24]; // 25 data points for channel 1
    wire signed [11:0] buffer_2_out [0:24]; // 25 data points for channel 2
    wire valid_out_buf_0, valid_out_buf_1, valid_out_buf_2;

    // Instantiate buffer modules for each channel
    conv2_buf #( .WIDTH(12), .HEIGHT(12), .DATA_BITS(12)) buffer_0 (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .data_in(data_in_0),
        .data_out_0(buffer_0_out[0]), .data_out_1(buffer_0_out[1]), .data_out_2(buffer_0_out[2]),
        .data_out_3(buffer_0_out[3]), .data_out_4(buffer_0_out[4]), .data_out_5(buffer_0_out[5]),
        .data_out_6(buffer_0_out[6]), .data_out_7(buffer_0_out[7]), .data_out_8(buffer_0_out[8]),
        .data_out_9(buffer_0_out[9]), .data_out_10(buffer_0_out[10]), .data_out_11(buffer_0_out[11]),
        .data_out_12(buffer_0_out[12]), .data_out_13(buffer_0_out[13]), .data_out_14(buffer_0_out[14]),
        .data_out_15(buffer_0_out[15]), .data_out_16(buffer_0_out[16]), .data_out_17(buffer_0_out[17]),
        .data_out_18(buffer_0_out[18]), .data_out_19(buffer_0_out[19]), .data_out_20(buffer_0_out[20]),
        .data_out_21(buffer_0_out[21]), .data_out_22(buffer_0_out[22]), .data_out_23(buffer_0_out[23]),
        .data_out_24(buffer_0_out[24]),
        .valid_out_buf(valid_out_buf_0)
    );

    conv2_buf #( .WIDTH(12), .HEIGHT(12), .DATA_BITS(12)) buffer_1 (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .data_in(data_in_1),
        .data_out_0(buffer_1_out[0]), .data_out_1(buffer_1_out[1]), .data_out_2(buffer_1_out[2]),
        .data_out_3(buffer_1_out[3]), .data_out_4(buffer_1_out[4]), .data_out_5(buffer_1_out[5]),
        .data_out_6(buffer_1_out[6]), .data_out_7(buffer_1_out[7]), .data_out_8(buffer_1_out[8]),
        .data_out_9(buffer_1_out[9]), .data_out_10(buffer_1_out[10]), .data_out_11(buffer_1_out[11]),
        .data_out_12(buffer_1_out[12]), .data_out_13(buffer_1_out[13]), .data_out_14(buffer_1_out[14]),
        .data_out_15(buffer_1_out[15]), .data_out_16(buffer_1_out[16]), .data_out_17(buffer_1_out[17]),
        .data_out_18(buffer_1_out[18]), .data_out_19(buffer_1_out[19]), .data_out_20(buffer_1_out[20]),
        .data_out_21(buffer_1_out[21]), .data_out_22(buffer_1_out[22]), .data_out_23(buffer_1_out[23]),
        .data_out_24(buffer_1_out[24]),
        .valid_out_buf(valid_out_buf_1)
    );

    conv2_buf #( .WIDTH(12), .HEIGHT(12), .DATA_BITS(12)) buffer_2 (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .data_in(data_in_2),
        .data_out_0(buffer_2_out[0]), .data_out_1(buffer_2_out[1]), .data_out_2(buffer_2_out[2]),
        .data_out_3(buffer_2_out[3]), .data_out_4(buffer_2_out[4]), .data_out_5(buffer_2_out[5]),
        .data_out_6(buffer_2_out[6]), .data_out_7(buffer_2_out[7]), .data_out_8(buffer_2_out[8]),
        .data_out_9(buffer_2_out[9]), .data_out_10(buffer_2_out[10]), .data_out_11(buffer_2_out[11]),
        .data_out_12(buffer_2_out[12]), .data_out_13(buffer_2_out[13]), .data_out_14(buffer_2_out[14]),
        .data_out_15(buffer_2_out[15]), .data_out_16(buffer_2_out[16]), .data_out_17(buffer_2_out[17]),
        .data_out_18(buffer_2_out[18]), .data_out_19(buffer_2_out[19]), .data_out_20(buffer_2_out[20]),
        .data_out_21(buffer_2_out[21]), .data_out_22(buffer_2_out[22]), .data_out_23(buffer_2_out[23]),
        .data_out_24(buffer_2_out[24]),
        .valid_out_buf(valid_out_buf_2)
    );

    // Instantiate depthwise convolution for each channel using the given depthwise_2 module
    wire signed [13:0] dep1_out, dep2_out, dep3_out;   // Outputs from depthwise convolution
    wire valid_out_calc;

    depthwise_2 depwise_inst (
        .clk(clk),
        .rst_n(rst_n),
        .valid_out_buf(valid_out_buf_0),
        .data_out1_0(buffer_0_out[0]), .data_out1_1(buffer_0_out[1]), .data_out1_2(buffer_0_out[2]),
        .data_out1_3(buffer_0_out[3]), .data_out1_4(buffer_0_out[4]), .data_out1_5(buffer_0_out[5]),
        .data_out1_6(buffer_0_out[6]), .data_out1_7(buffer_0_out[7]), .data_out1_8(buffer_0_out[8]),
        .data_out1_9(buffer_0_out[9]), .data_out1_10(buffer_0_out[10]), .data_out1_11(buffer_0_out[11]),
        .data_out1_12(buffer_0_out[12]), .data_out1_13(buffer_0_out[13]), .data_out1_14(buffer_0_out[14]),
        .data_out1_15(buffer_0_out[15]), .data_out1_16(buffer_0_out[16]), .data_out1_17(buffer_0_out[17]),
        .data_out1_18(buffer_0_out[18]), .data_out1_19(buffer_0_out[19]), .data_out1_20(buffer_0_out[20]),
        .data_out1_21(buffer_0_out[21]), .data_out1_22(buffer_0_out[22]), .data_out1_23(buffer_0_out[23]),
        .data_out1_24(buffer_0_out[24]),
        
        .data_out2_0(buffer_1_out[0]), .data_out2_1(buffer_1_out[1]), .data_out2_2(buffer_1_out[2]),
        .data_out2_3(buffer_1_out[3]), .data_out2_4(buffer_1_out[4]), .data_out2_5(buffer_1_out[5]),
        .data_out2_6(buffer_1_out[6]), .data_out2_7(buffer_1_out[7]), .data_out2_8(buffer_1_out[8]),
        .data_out2_9(buffer_1_out[9]), .data_out2_10(buffer_1_out[10]), .data_out2_11(buffer_1_out[11]),
        .data_out2_12(buffer_1_out[12]), .data_out2_13(buffer_1_out[13]), .data_out2_14(buffer_1_out[14]),
        .data_out2_15(buffer_1_out[15]), .data_out2_16(buffer_1_out[16]), .data_out2_17(buffer_1_out[17]),
        .data_out2_18(buffer_1_out[18]), .data_out2_19(buffer_1_out[19]), .data_out2_20(buffer_1_out[20]),
        .data_out2_21(buffer_1_out[21]), .data_out2_22(buffer_1_out[22]), .data_out2_23(buffer_1_out[23]),
        .data_out2_24(buffer_1_out[24]),
        
        .data_out3_0(buffer_2_out[0]), .data_out3_1(buffer_2_out[1]), .data_out3_2(buffer_2_out[2]),
        .data_out3_3(buffer_2_out[3]), .data_out3_4(buffer_2_out[4]), .data_out3_5(buffer_2_out[5]),
        .data_out3_6(buffer_2_out[6]), .data_out3_7(buffer_2_out[7]), .data_out3_8(buffer_2_out[8]),
        .data_out3_9(buffer_2_out[9]), .data_out3_10(buffer_2_out[10]), .data_out3_11(buffer_2_out[11]),
        .data_out3_12(buffer_2_out[12]), .data_out3_13(buffer_2_out[13]), .data_out3_14(buffer_2_out[14]),
        .data_out3_15(buffer_2_out[15]), .data_out3_16(buffer_2_out[16]), .data_out3_17(buffer_2_out[17]),
        .data_out3_18(buffer_2_out[18]), .data_out3_19(buffer_2_out[19]), .data_out3_20(buffer_2_out[20]),
        .data_out3_21(buffer_2_out[21]), .data_out3_22(buffer_2_out[22]), .data_out3_23(buffer_2_out[23]),
        .data_out3_24(buffer_2_out[24]),
        
        .dep_out_1(dep1_out),
        .dep_out_2(dep2_out),
        .dep_out_3(dep3_out),
        .valid_out_calc(valid_out_calc)
    );

    // Pointwise convolution after depthwise convolution
    wire signed [15:0] conv1_out, conv2_out, conv3_out,
                             conv4_out, conv5_out, conv6_out,
                             conv7_out, conv8_out, conv9_out;
    pointwise_conv2 pointwise_inst (
        .clk(clk),
        .rst_n(rst_n),
        .valid_calc_out(valid_out_calc),
        .dep_out_1(dep1_out),
        .dep_out_2(dep2_out),
        .dep_out_3(dep3_out),
        .conv1_out(conv1_out),
        .conv2_out(conv2_out),
        .conv3_out(conv3_out),
        .conv4_out(conv4_out),
        .conv5_out(conv5_out),
        .conv6_out(conv6_out),
        .conv7_out(conv7_out),
        .conv8_out(conv8_out),
        .conv9_out(conv9_out),
        .valid_out_calc(conv_valid_out)
    );

endmodule

