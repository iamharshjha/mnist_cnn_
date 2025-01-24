`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: IITDH
// Engineer: Dixshant Jha 
// 
// Create Date: 01/19/2025 09:50:15 AM
// Design Name: 
// Module Name: depthwise_separable_conv
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


module depthwise_separable_conv1 (
    input valid_out_buf,
    input signed [7:0] data_out_0, data_out_1, data_out_2, data_out_3, data_out_4,
                  data_out_5, data_out_6, data_out_7, data_out_8, data_out_9,
                  data_out_10, data_out_11, data_out_12, data_out_13, data_out_14,
                  data_out_15, data_out_16, data_out_17, data_out_18, data_out_19,
                  data_out_20, data_out_21, data_out_22, data_out_23, data_out_24,
    output signed [31:0] conv_out_1, conv_out_2,conv_out_3 , 
    output reg valid_out_calc
);

    // Declare memory arrays for the weights and biases
    logic signed [7:0] depthwise_kernel [0:4][0:4];
    logic signed [7:0] pointwise_kernels [0:2];
    logic signed [7:0] biases [0:2];

    // Flatten data for reading memory
    logic [7:0] flat_data [0:30];

    // Initial block to read weights and biases from memory
    initial begin
        $readmemh("layer0_conv2d.mem",flat_data);
        
        // Assign flat_data to depthwise_kernel
//        $display("Flat Data:");
//        for (int i = 0; i < 31; i++) begin
//            $display("flat_data[%0d] = %0h", i, flat_data[i]); // Print each value in hexadecimal format
//        end
        
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
        

        
//         $display("Depthwise Kernel:");
//           for (int k = 0; k < 5; k++) begin
//               for (int j = 0; j < 5; j++) begin
//                   $display("depthwise_kernel[%0d][%0d] = %0d", k, j, depthwise_kernel[k][j]);
//               end
//           end
       
//           $display("Pointwise Kernels:");
//           $display("pointwise_kernels[0] = %0d", pointwise_kernels[0]);
//           $display("pointwise_kernels[1] = %0d", pointwise_kernels[1]);
//           $display("pointwise_kernels[2] = %0d", pointwise_kernels[2]);
//           $display("Biases:");
//           $display("biases[0] = %0d", biases[0]);
//           $display("biases[1] = %0d", biases[1]);
//           $display("biases[2] = %0d", biases[2]);
        
    end

    // Local variables for depthwise convolution results
    wire  signed [31:0] depthwise_result_1;

    // Perform depthwise convolution
    assign depthwise_result_1 = (data_out_0 * depthwise_kernel[0][0] + data_out_1 * depthwise_kernel[0][1] + 
                                 data_out_2 * depthwise_kernel[0][2] + data_out_3 * depthwise_kernel[0][3] + 
                                 data_out_4 * depthwise_kernel[0][4] + data_out_5 * depthwise_kernel[1][0] + 
                                 data_out_6 * depthwise_kernel[1][1] + data_out_7 * depthwise_kernel[1][2] + 
                                 data_out_8 * depthwise_kernel[1][3] + data_out_9 * depthwise_kernel[1][4] + 
                                 data_out_10 * depthwise_kernel[2][0] + data_out_11 * depthwise_kernel[2][1] + 
                                 data_out_12 * depthwise_kernel[2][2] + data_out_13 * depthwise_kernel[2][3] + 
                                 data_out_14 * depthwise_kernel[2][4] + data_out_15 * depthwise_kernel[3][0] + 
                                 data_out_16 * depthwise_kernel[3][1] + data_out_17 * depthwise_kernel[3][2] + 
                                 data_out_18 * depthwise_kernel[3][3] + data_out_19 * depthwise_kernel[3][4] + 
                                 data_out_20 * depthwise_kernel[4][0] + data_out_21 * depthwise_kernel[4][1] + 
                                 data_out_22 * depthwise_kernel[4][2] + data_out_23 * depthwise_kernel[4][3] + 
                                 data_out_24 * depthwise_kernel[4][4]);

    // Pointwise convolution using depthwise results
   

    assign conv_out_1 = (depthwise_result_1 * pointwise_kernels[0]   )  + biases[0];
    assign conv_out_2 = (depthwise_result_1 * pointwise_kernels[1]   )  + biases[1];
    assign conv_out_3 = (depthwise_result_1 * pointwise_kernels[2]   )  + biases[2];
    
    // Set the valid output signal
   always @(posedge valid_out_buf) begin
        valid_out_calc <= 1;
        $display("depthwise_result_1 = %0d", depthwise_result_1);
    end

endmodule

