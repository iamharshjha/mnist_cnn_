`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 01/19/2025 10:41:39 AM
// Design Name: 
// Module Name: depthwise_conv1_tb
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

module depthwise_conv1_tb;

    // Testbench signals
    logic valid_out_buf;
    logic signed [7:0] data_out_0, data_out_1, data_out_2, data_out_3, data_out_4;
    logic signed [7:0] data_out_5, data_out_6, data_out_7, data_out_8, data_out_9;
    logic signed [7:0] data_out_10, data_out_11, data_out_12, data_out_13, data_out_14;
    logic signed [7:0] data_out_15, data_out_16, data_out_17, data_out_18, data_out_19;
    logic signed [7:0] data_out_20, data_out_21, data_out_22, data_out_23, data_out_24;

    wire signed [20:0] conv_out_1, conv_out_2, conv_out_3 ;
    wire valid_out_calc;

    // Instantiate the module under test (MUT)
    depthwise_separable_conv mut (
        .valid_out_buf(valid_out_buf),
        .data_out_0(data_out_0), .data_out_1(data_out_1), .data_out_2(data_out_2), .data_out_3(data_out_3), .data_out_4(data_out_4),
        .data_out_5(data_out_5), .data_out_6(data_out_6), .data_out_7(data_out_7), .data_out_8(data_out_8), .data_out_9(data_out_9),
        .data_out_10(data_out_10), .data_out_11(data_out_11), .data_out_12(data_out_12), .data_out_13(data_out_13), .data_out_14(data_out_14),
        .data_out_15(data_out_15), .data_out_16(data_out_16), .data_out_17(data_out_17), .data_out_18(data_out_18), .data_out_19(data_out_19),
        .data_out_20(data_out_20), .data_out_21(data_out_21), .data_out_22(data_out_22), .data_out_23(data_out_23), .data_out_24(data_out_24),
        .conv_out_1(conv_out_1),
        .conv_out_2(conv_out_2),
        .conv_out_3(conv_out_3),
        .valid_out_calc(valid_out_calc)
    );

    // Initialize signals and apply stimulus
    initial begin
        // Initialize inputs
        valid_out_buf = 0;
        data_out_0 = 1; 
        data_out_1 = 1; 
        data_out_2 = 1; 
        data_out_3 = 0; 
        data_out_4 = 0;
        data_out_5 = 0; 
        data_out_6 = 0; 
        data_out_7 = 0; 
        data_out_8 = 0; 
        data_out_9 = 0;
        data_out_10 = 0; 
        data_out_11 = 0; 
        data_out_12 = 0; 
        data_out_13 = 0; 
        data_out_14 = 0;
        data_out_15 = 0; 
        data_out_16 = 0; 
        data_out_17 = 0; 
        data_out_18 = 0; 
        data_out_19 = 0;
        data_out_20 = 0; 
        data_out_21 = 0; 
        data_out_22 = 0; 
        data_out_23 = 0; 
        data_out_24 = 0;


        // Wait for memory initialization
        #10;
        
        // Apply valid signal and observe output
        valid_out_buf = 1;
        #10;
        valid_out_buf = 0;
        
        // Wait for processing and display outputs
        #50;
        $display("conv_out_1 = %d", conv_out_1);
        $display("conv_out_2 = %d", conv_out_2);
        $display("conv_out_3 = %d", conv_out_3);
        $display("valid_out_calc = %b", valid_out_calc);

        // End simulation
        $finish;
    end

endmodule

