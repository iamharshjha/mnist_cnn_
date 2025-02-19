module maxpool_relu #(
  parameter CONV_BIT      = 14,
  parameter HALF_WIDTH    = 24,  // Updated: input image width is 24 pixels
  parameter HALF_HEIGHT   = 24,  // Updated: input image height is 24 pixels
  parameter HALF_WIDTH_BIT = 5   // 5 bits needed to count from 0 to 23
)(
  input clk,
  input rst_n,  // asynchronous reset, active low
  input valid_in,
  input signed [CONV_BIT-1:0] conv_out_1,  // channel 1 pixel
  input signed [CONV_BIT-1:0] conv_out_2,  // channel 2 pixel
  input signed [CONV_BIT-1:0] conv_out_3,  // channel 3 pixel
  output reg [CONV_BIT-1:0] max_value_1,     // pooled output for ch1
  output reg [CONV_BIT-1:0] max_value_2,     // pooled output for ch2
  output reg [CONV_BIT-1:0] max_value_3,     // pooled output for ch3
  output reg valid_out_relu
);

  // Pooling is performed on 2x2 windows.
  localparam POOL_SIZE = 2;
  // Output image dimensions after 2x2 pooling:
  localparam OUT_WIDTH  = HALF_WIDTH  / POOL_SIZE;   // 24/2 = 12
  localparam OUT_HEIGHT = HALF_HEIGHT / POOL_SIZE;    // 24/2 = 12

  // Pixel position counters for the 24x24 image.
  // Use HALF_WIDTH_BIT bits (now 5 bits) to count 0 to 23.
  reg [HALF_WIDTH_BIT-1:0] col;  // current column (0 to 23)
  reg [HALF_WIDTH_BIT-1:0] row;  // current row (0 to 23)

  // Pooling-window column counter for output (0 to OUT_WIDTH-1, i.e. 0 to 11)
  reg [3:0] pool_col;

  // Temporary registers to hold the first pixel of a 2-pixel group in a row.
  reg signed [CONV_BIT-1:0] temp1, temp2, temp3;

  // Line buffers to store the row max from the first row of each 2x2 pooling window.
  // These are indexed by the output column (0 to OUT_WIDTH-1).
  reg signed [CONV_BIT-1:0] row_max_buf1 [0:OUT_WIDTH-1];
  reg signed [CONV_BIT-1:0] row_max_buf2 [0:OUT_WIDTH-1];
  reg signed [CONV_BIT-1:0] row_max_buf3 [0:OUT_WIDTH-1];

  // Internal signals for the computed row max (per 2-pixel group) and final 2x2 window max.
  reg signed [CONV_BIT-1:0] row_max1, row_max2, row_max3;
  reg signed [CONV_BIT-1:0] final_max1, final_max2, final_max3;

  integer i;
  // Sequential process: reset and processing pixels.
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      // Reset counters and registers.
      col <= 0;
      row <= 0;
      pool_col <= 0;
      valid_out_relu <= 0;
      max_value_1 <= 0;
      max_value_2 <= 0;
      max_value_3 <= 0;
      temp1 <= 0;
      temp2 <= 0;
      temp3 <= 0;
      for (i = 0; i < OUT_WIDTH; i = i + 1) begin
        row_max_buf1[i] <= 0;
        row_max_buf2[i] <= 0;
        row_max_buf3[i] <= 0;
      end
    end else if (valid_in) begin
      valid_out_relu <= 0;  // Default: no valid output until produced below

      // Process the pixel at the current (row, col) position.
      // If the column is even, capture the pixel values into temporary registers.
      if (col[0] == 1'b0) begin
        temp1 <= conv_out_1;
        temp2 <= conv_out_2;
        temp3 <= conv_out_3;
      end else begin
        // For an odd column, we now have a pair of pixels in the same row.
        // Compute the row max for this 2-pixel group.
        row_max1 = (temp1 > conv_out_1) ? temp1 : conv_out_1;
        row_max2 = (temp2 > conv_out_2) ? temp2 : conv_out_2;
        row_max3 = (temp3 > conv_out_3) ? temp3 : conv_out_3;

        // For an even-numbered row (row[0] == 0), store the row max in the line buffer.
        if (row[0] == 1'b0) begin
          row_max_buf1[pool_col] <= row_max1;
          row_max_buf2[pool_col] <= row_max2;
          row_max_buf3[pool_col] <= row_max3;
        end else begin
          // For an odd-numbered row, perform the final 2x2 comparison:
          // Compare the stored row max (from the previous row) with the current row's max.
          final_max1 = (row_max_buf1[pool_col] > row_max1) ? row_max_buf1[pool_col] : row_max1;
          final_max2 = (row_max_buf2[pool_col] > row_max2) ? row_max_buf2[pool_col] : row_max2;
          final_max3 = (row_max_buf3[pool_col] > row_max3) ? row_max_buf3[pool_col] : row_max3;
          // Apply ReLU: if the final max is negative, output zero.
          max_value_1 <= (final_max1 > 0) ? final_max1 : 0;
          max_value_2 <= (final_max2 > 0) ? final_max2 : 0;
          max_value_3 <= (final_max3 > 0) ? final_max3 : 0;
          valid_out_relu <= 1;  // Indicate that the pooling output is valid.
        end

        // Update the pooling window's column counter.
        if (col == HALF_WIDTH - 1)
          pool_col <= 0;
        else
          pool_col <= pool_col + 1;
      end

      // Increment the column counter.
      if (col == HALF_WIDTH - 1) begin
        col <= 0;
        // End of the row: increment the row counter.
        if (row == HALF_HEIGHT - 1)
          row <= 0;
        else
          row <= row + 1;
      end else begin
        col <= col + 1;
      end
    end
  end

endmodule

