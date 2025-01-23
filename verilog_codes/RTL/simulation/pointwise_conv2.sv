`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 01/22/2025 10:14:09 AM
// Design Name: 
// Module Name: pointwise_conv2
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


module pointwise_conv2(
    input clk, 
    input rst_n, 
    input signed [12:0] dep_out_1, dep_out_2, dep_out_3, 
    input valid_calc_out, 
    output reg signed [15:0] conv1_out, conv2_out, conv3_out,
                             conv4_out, conv5_out, conv6_out,
                             conv7_out, conv8_out, conv9_out,
    output reg valid_out_calc
);

    // Declare the weight and bias arrays (9 filters, each with 3 weights and 1 bias)
    reg signed [7:0] weights_flat[0:110];  // 36 values (27 weights + 9 biases)
    reg signed [7:0] weights [0:8][0:2];  // 9 filters with 3 weights each
    reg signed [7:0] biases [0:8];         // 9 biases (one for each filter)

    // Internal calculation output array (for debugging or storing intermediate results)
    reg signed [19:0] calc_out [0:8];

    // Initial block for simulation to read weights and biases
    initial begin
        $readmemh("layer2_conv2d.mem", weights_flat);
        
        // Populate the weights and biases arrays from the flat memory
        for (int i = 0; i < 9; i++) begin
            weights[i][0] = weights_flat[75 + i * 3];      // Weight for channel 1 of filter i
            weights[i][1] = weights_flat[75 + i * 3 + 1];  // Weight for channel 2 of filter i
            weights[i][2] = weights_flat[75 + i * 3 + 2];  // Weight for channel 3 of filter i
            biases[i] = weights_flat[102 + i];             // Bias for filter i
        end
        
        // Display weights and biases for debugging (simulation)
        $display("Weights and Biases:");
        for (int i = 0; i < 9; i++) begin
            $display("Filter %0d:", i);
            $display("  Channel 1: %0d", weights[i][0]);
            $display("  Channel 2: %0d", weights[i][1]);
            $display("  Channel 3: %0d", weights[i][2]);
            $display("  Bias: %0d", biases[i]);
        end
    end

    // Process the convolution and apply the filters to the inputs
    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            // Reset the outputs and valid signal
            conv1_out <= 0;
            conv2_out <= 0;
            conv3_out <= 0;
            conv4_out <= 0;
            conv5_out <= 0;
            conv6_out <= 0;
            conv7_out <= 0;
            conv8_out <= 0;
            conv9_out <= 0;
            valid_out_calc <= 0;
        end else if (valid_calc_out) begin
            // Apply the filters on the 3 input channels (pointwise convolution)

            // Filter 1: Apply the weights and bias
            conv1_out <= dep_out_1 * weights[0][0] + dep_out_2 * weights[0][1] + dep_out_3 * weights[0][2] + biases[0];
            
            // Filter 2
            conv2_out <= dep_out_1 * weights[1][0] + dep_out_2 * weights[1][1] + dep_out_3 * weights[1][2] + biases[1];
            
            // Filter 3
            conv3_out <= dep_out_1 * weights[2][0] + dep_out_2 * weights[2][1] + dep_out_3 * weights[2][2] + biases[2];
            
            // Filter 4
            conv4_out <= dep_out_1 * weights[3][0] + dep_out_2 * weights[3][1] + dep_out_3 * weights[3][2] + biases[3];
            
            // Filter 5
            conv5_out <= dep_out_1 * weights[4][0] + dep_out_2 * weights[4][1] + dep_out_3 * weights[4][2] + biases[4];
            
            // Filter 6
            conv6_out <= dep_out_1 * weights[5][0] + dep_out_2 * weights[5][1] + dep_out_3 * weights[5][2] + biases[5];
            
            // Filter 7
            conv7_out <= dep_out_1 * weights[6][0] + dep_out_2 * weights[6][1] + dep_out_3 * weights[6][2] + biases[6];
            
            // Filter 8
            conv8_out <= dep_out_1 * weights[7][0] + dep_out_2 * weights[7][1] + dep_out_3 * weights[7][2] + biases[7];
            
            // Filter 9
            conv9_out <= dep_out_1 * weights[8][0] + dep_out_2 * weights[8][1] + dep_out_3 * weights[8][2] + biases[8];
            
            // Set valid signal
            valid_out_calc <= 1;
        end else begin
            valid_out_calc <= 0;
        end
    end

endmodule
