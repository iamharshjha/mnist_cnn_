`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 01/22/2025 02:45:48 PM
// Design Name: 
// Module Name: fully_connected_layer
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
//   Fully connected layer that buffers 144 inputs (from 9 channels and 16 cycles)
//   and computes dot products for 10 output neurons. Each multiplication is divided
//   by 32, the input is 15 bits, and the output is 18 bits wide.
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module fully_connected #(
    parameter INPUT_NUM  = 144,  // 9 channels * 16 cycles = 144 inputs
    parameter OUTPUT_NUM = 10,
    parameter DATA_BITS  = 8    // weights and biases are 8 bits
)(
    input clk,
    input rst_n,
    input valid_in,
    // Inputs are 15 bits wide.
    input signed [14:0] data_in_1, data_in_2, data_in_3, data_in_4, 
                        data_in_5, data_in_6, data_in_7, data_in_8, data_in_9,
    // Output is now 18 bits wide.
    output reg [17:0] data_out,
    output reg valid_out_fc
);

  localparam INPUT_WIDTH = 16; // number of cycles per channel

  reg state;                         // 0: buffering, 1: computing
  reg [INPUT_WIDTH-1:0] buf_idx;       // index for each cycle (0 to 15)
  reg [3:0] out_idx;                 // index for output neuron (0 to 9)
  // Update buffer to store 15-bit inputs.
  reg signed [14:0] buffer [0:INPUT_NUM-1];  // 144-element buffer
  
  // Weights and biases (8-bit each)
  reg signed [DATA_BITS-1:0] weight [0:INPUT_NUM*OUTPUT_NUM-1];
  reg signed [DATA_BITS-1:0] bias [0:OUTPUT_NUM-1];

  // Load weights and biases from a single flat memory file.
  reg signed [DATA_BITS-1:0] weight_bias [0:(INPUT_NUM*OUTPUT_NUM+OUTPUT_NUM)-1];
  integer i;
  initial begin
    $readmemh("layer5_dense.mem", weight_bias);
    for(i = 0; i < INPUT_NUM * OUTPUT_NUM; i = i + 1) begin
      weight[i] = weight_bias[i];
    end
    for(i = 0; i < OUTPUT_NUM; i = i + 1) begin
      bias[i] = weight_bias[INPUT_NUM * OUTPUT_NUM + i];
    end
  end

  // Since the inputs are already 15 bits, we simply pass them through.
  wire signed [14:0] data1, data2, data3, data4, data5, data6, data7, data8, data9;
  assign data1 = data_in_1;
  assign data2 = data_in_2;
  assign data3 = data_in_3;
  assign data4 = data_in_4;
  assign data5 = data_in_5;
  assign data6 = data_in_6;
  assign data7 = data_in_7;
  assign data8 = data_in_8;
  assign data9 = data_in_9;

  // State Machine: Buffer inputs and then compute outputs.
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      valid_out_fc <= 0;
      buf_idx      <= 0;
      out_idx      <= 0;
      state        <= 0;
    end else begin
      // Clear valid flag after each cycle.
      if (valid_out_fc)
        valid_out_fc <= 0;
      
      if (valid_in) begin
        if (!state) begin
          // BUFFERING PHASE:
          // Store one sample from each of the 9 inputs per valid cycle.
          buffer[buf_idx]                   <= data1;
          buffer[INPUT_WIDTH + buf_idx]     <= data2;
          buffer[INPUT_WIDTH*2 + buf_idx]   <= data3;
          buffer[INPUT_WIDTH*3 + buf_idx]   <= data4;
          buffer[INPUT_WIDTH*4 + buf_idx]   <= data5;
          buffer[INPUT_WIDTH*5 + buf_idx]   <= data6;
          buffer[INPUT_WIDTH*6 + buf_idx]   <= data7;
          buffer[INPUT_WIDTH*7 + buf_idx]   <= data8;
          buffer[INPUT_WIDTH*8 + buf_idx]   <= data9;
          buf_idx <= buf_idx + 1;
          if (buf_idx == INPUT_WIDTH - 1) begin
            buf_idx <= 0;
            state   <= 1; // Switch to computation phase after 144 inputs are collected.
            valid_out_fc <= 1;
          end
        end else begin
          // COMPUTATION PHASE:
          // Cycle through each output neuron to compute the dot product.
          out_idx <= out_idx + 1;
          if (out_idx == OUTPUT_NUM - 1)
            out_idx <= 0;
          valid_out_fc <= 1;
        end
      end
    end
  end

  // Dot Product Computation:
  // For the selected output neuron (indexed by out_idx), multiply each buffered input
  // by its corresponding weight (each product divided by 32), sum the results, then add the bias.
  reg signed [19:0] sum;
  integer j;
  always @(*) begin
    sum = 20'd0;
    for(j = 0; j < INPUT_NUM; j = j + 1) begin
      // Divide each multiplication by 32.
      sum = sum + ((weight[out_idx * INPUT_NUM + j] * buffer[j]) / 32);
    end
    sum = sum + bias[out_idx];
  end
  wire signed [19:0] calc_out = sum;

  // The accumulated sum is 20 bits. We now produce an 18-bit output.
  // We slice bits [19:2] from calc_out to form the final 18-bit output.
  reg [17:0] data_out_reg;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      data_out_reg <= 18'd0;
    else
      data_out_reg <= calc_out[19:2];
  end
  
  // Update the output port.
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      data_out <= 0;
    else
      data_out <= data_out_reg;
  end

endmodule
