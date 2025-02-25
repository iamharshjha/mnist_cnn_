`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: IITDH
// Engineer: Dixshant Jha 
// 
// Create Date: 01/19/2025 09:50:15 AM
// Design Name: 
// Module Name: depthwise_separable_conv1
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
//   Depthwise separable convolution with saturation logic to ensure the
//   final output is a 15-bit signed number.
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
//
//////////////////////////////////////////////////////////////////////////////////

module depthwise_separable_conv1 (
    input  valid_out_buf,
    input  signed [7:0] data_out_0, data_out_1, data_out_2, data_out_3, data_out_4,
                         data_out_5, data_out_6, data_out_7, data_out_8, data_out_9,
                         data_out_10, data_out_11, data_out_12, data_out_13, data_out_14,
                         data_out_15, data_out_16, data_out_17, data_out_18, data_out_19,
                         data_out_20, data_out_21, data_out_22, data_out_23, data_out_24,
    output signed [14:0] conv_out_1, conv_out_2, conv_out_3,
    output reg valid_out_calc
);

    // Declare memory arrays for the weights and biases
    logic signed [7:0] depthwise_kernel [0:4][0:4];
    logic signed [7:0] pointwise_kernels [0:2];
    logic signed [7:0] biases [0:2];

    // Flatten data for reading memory
    logic [7:0] flat_data [0:30];

    // Read weights and biases from memory
    initial begin
        $readmemh("layer0_conv2d.mem", flat_data);
        
        // Assign flat_data to depthwise_kernel
        for (int i = 0; i < 5; i++) begin
            for (int j = 0; j < 5; j++) begin
                depthwise_kernel[i][j] = flat_data[i * 5 + j];
            end
        end

        // Assign to pointwise_kernels
        pointwise_kernels[0] = flat_data[25];
        pointwise_kernels[1] = flat_data[26];
        pointwise_kernels[2] = flat_data[27];
        
        // Assign to biases
        biases[0] = flat_data[28];
        biases[1] = flat_data[29];
        biases[2] = flat_data[30];
    end

    // Compute depthwise convolution result (24-bit signed)
    wire signed [23:0] depthwise_result_1;
    assign depthwise_result_1 = 
      ((data_out_0  * depthwise_kernel[0][0] + 8) / 32) +
      ((data_out_1  * depthwise_kernel[0][1] + 8) / 32) +
      ((data_out_2  * depthwise_kernel[0][2] + 8) / 32) +
      ((data_out_3  * depthwise_kernel[0][3] + 8) / 32) +
      ((data_out_4  * depthwise_kernel[0][4] + 8) / 32) +
      ((data_out_5  * depthwise_kernel[1][0] + 8) / 32) +
      ((data_out_6  * depthwise_kernel[1][1] + 8) / 32) +
      ((data_out_7  * depthwise_kernel[1][2] + 8) / 32) +
      ((data_out_8  * depthwise_kernel[1][3] + 8) / 32) +
      ((data_out_9  * depthwise_kernel[1][4] + 8) / 32) +
      ((data_out_10 * depthwise_kernel[2][0] + 8) / 32) +
      ((data_out_11 * depthwise_kernel[2][1] + 8) / 32) +
      ((data_out_12 * depthwise_kernel[2][2] + 8) / 32) +
      ((data_out_13 * depthwise_kernel[2][3] + 8) / 32) +
      ((data_out_14 * depthwise_kernel[2][4] + 8) / 32) +
      ((data_out_15 * depthwise_kernel[3][0] + 8) / 32) +
      ((data_out_16 * depthwise_kernel[3][1] + 8) / 32) +
      ((data_out_17 * depthwise_kernel[3][2] + 8) / 32) +
      ((data_out_18 * depthwise_kernel[3][3] + 8) / 32) +
      ((data_out_19 * depthwise_kernel[3][4] + 8) / 32) +
      ((data_out_20 * depthwise_kernel[4][0] + 8) / 32) +
      ((data_out_21 * depthwise_kernel[4][1] + 8) / 32) +
      ((data_out_22 * depthwise_kernel[4][2] + 8) / 32) +
      ((data_out_23 * depthwise_kernel[4][3] + 8) / 32) +
      ((data_out_24 * depthwise_kernel[4][4] + 8) / 32);

    // Full precision (32-bit) pointwise convolution results
    wire signed [31:0] conv1_full, conv2_full, conv3_full;
    assign conv1_full = (depthwise_result_1 * pointwise_kernels[0]) / 32 + biases[0];
    assign conv2_full = (depthwise_result_1 * pointwise_kernels[1]) / 32 + biases[1];
    assign conv3_full = (depthwise_result_1 * pointwise_kernels[2]) / 32 + biases[2];

    // Saturate the 32-bit results to 15 bits.
    // 15-bit signed range: -16384 to 16383.
    reg signed [14:0] conv_out_1_reg, conv_out_2_reg, conv_out_3_reg;
    always @(*) begin
        // Saturation for conv1 result
        if (conv1_full > 32'sd16383)
            conv_out_1_reg = 15'sd16383;
        else if (conv1_full < -32'sd16384)
            conv_out_1_reg = -15'sd16384;
        else
            conv_out_1_reg = conv1_full[14:0];
            
        // Saturation for conv2 result
        if (conv2_full > 32'sd16383)
            conv_out_2_reg = 15'sd16383;
        else if (conv2_full < -32'sd16384)
            conv_out_2_reg = -15'sd16384;
        else
            conv_out_2_reg = conv2_full[14:0];
            
        // Saturation for conv3 result
        if (conv3_full > 32'sd16383)
            conv_out_3_reg = 15'sd16383;
        else if (conv3_full < -32'sd16384)
            conv_out_3_reg = -15'sd16384;
        else
            conv_out_3_reg = conv3_full[14:0];
    end

    // Connect the saturated values to the outputs.
    assign conv_out_1 = conv_out_1_reg;
    assign conv_out_2 = conv_out_2_reg;
    assign conv_out_3 = conv_out_3_reg;

    // Pass through the valid signal.
    always @(*) begin
        valid_out_calc = valid_out_buf;
    end

endmodule
