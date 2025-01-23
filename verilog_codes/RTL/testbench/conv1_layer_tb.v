module conv1_layer_tb;

   // Testbench signals
   reg clk;
   reg rst_n;
   reg [7:0] data_in;  // Input data for the module
   wire [11:0] conv_out_1, conv_out_2, conv_out_3;  // Convolution outputs
   wire valid_out_conv; // Valid output for convolution

   // Instantiate the top-level module
   conv1_layer uut (
      .clk(clk),
      .rst_n(rst_n),
      .data_in(data_in),
      .conv_out_1(conv_out_1),
      .conv_out_2(conv_out_2),
      .conv_out_3(conv_out_3),
      .valid_out_conv(valid_out_conv)
   );

   // Memory to hold the data read from the .mem file
   reg [7:0] pixels [0:783];  // Assuming 784 pixels for a 28x28 image
   
   reg [9:0] img_idx;  // Image index
   
   // Clock generation
   always #5 clk = ~clk;  // 100 MHz clock

   // Read image .mem file and apply data
   initial begin
      // Initialize signals
      clk = 0;
      rst_n = 0;
      data_in = 8'b0;
      img_idx = 0;

      // Read the .mem file into the pixels array
      $readmemh("/media/harsh/c98020b5-0d6a-4055-87fd-8c8b7a337914/MTP_PROJECT/verilog_codes/output_image.mem", pixels);  

      // Debugging the values to check if the file is read
      

      // Apply reset signal
      #3 rst_n <= 1'b0;
      #3 rst_n <= 1'b1;  // Deassert reset after 6 time units

   end

   // Driving the input data to the module on every clock cycle
   always @(posedge clk) begin
      if (~rst_n) begin
         img_idx <= 0;
      end else begin
         if (img_idx < 784) begin
            data_in <= pixels[img_idx];  // Load data from the pixels array
            img_idx <= img_idx + 1;  // Increment the index to read the next pixel
            $display("input_img: %h", data_in );  // Display the input data
         end
      end
   end

   // Finish simulation after all data is applied
   initial begin
      #7840;  // Wait for 784 cycles (enough time for all pixels to be applied)
      $finish;  // End the simulation
   end

endmodule
