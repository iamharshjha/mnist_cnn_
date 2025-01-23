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
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module fully_connected #(parameter INPUT_NUM = 144, OUTPUT_NUM = 10, DATA_BITS = 8) (
    input clk,
    input rst_n,
    input valid_in,
    input signed [11:0] data_in_1, data_in_2, data_in_3, data_in_4, data_in_5,
                       data_in_6, data_in_7, data_in_8, data_in_9,
    output reg [11:0] data_out,
    output reg valid_out_fc
);

    localparam INPUT_WIDTH = 16; // Total number of inputs per data_in (9 inputs x 16 cycles = 144 inputs)
    localparam INPUT_NUM_DATA_BITS = 8;

    reg state;
    reg [3:0] out_idx; // Index for output neurons
    reg [INPUT_WIDTH - 1:0] buf_idx; // Index for input neurons
    reg signed [13:0] buffer [0:INPUT_NUM - 1]; // Buffer to hold all inputs (144 elements)
    reg signed [DATA_BITS - 1:0] weight [0:INPUT_NUM * OUTPUT_NUM - 1]; // Weight array
    reg [DATA_BITS-1:0] weight_bias [0:1449];
    reg signed [DATA_BITS - 1:0] bias [0:OUTPUT_NUM - 1]; // Bias array

    wire signed [19:0] calc_out;
    wire signed [13:0] data [1:9]; // Data wires for inputs

    // Initialize weights and biases from files
    integer i;
    initial begin
        $readmemh("layer5_dense.mem", weight_bias);
        //$readmemh("media/harsh/c98020b5-0d6a-4055-87fd-8c8b7a337914/MTP_PROJECT/verilog_codes/layer5_conv2d.mem", bias,1440,1449);
        
   for(i = 0; i < 1440; i = i + 1) begin
     weight[i] = weight_bias[i];
   end

   // Assign the last 10 values to the bias array
   for(i = 0; i < 10; i = i + 1) begin
     bias[i] = weight_bias[1440 + i];
   end
        
    end

    // Sign-extend inputs to 14 bits
    genvar j;
    generate
        for (j = 1; j <= 9; j = j + 1) begin
            assign data[j] = (data_in_1[11] == 1) ? {2'b11, data_in_1} : {2'b00, data_in_1};
        end
    endgenerate

    // State machine for filling buffer and computing output
    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            valid_out_fc <= 0;
            buf_idx <= 0;
            out_idx <= 0;
            state <= 0;
        end else begin
            if (valid_out_fc == 1)
                valid_out_fc <= 0;

            if (valid_in) begin
                if (!state) begin // Fill the buffer with inputs
                    buffer[buf_idx] <= data[1];
                    buffer[INPUT_WIDTH + buf_idx] <= data[2];
                    buffer[INPUT_WIDTH * 2 + buf_idx] <= data[3];
                    buffer[INPUT_WIDTH * 3 + buf_idx] <= data[4];
                    buffer[INPUT_WIDTH * 4 + buf_idx] <= data[5];
                    buffer[INPUT_WIDTH * 5 + buf_idx] <= data[6];
                    buffer[INPUT_WIDTH * 6 + buf_idx] <= data[7];
                    buffer[INPUT_WIDTH * 7 + buf_idx] <= data[8];
                    buffer[INPUT_WIDTH * 8 + buf_idx] <= data[9];
                    buf_idx <= buf_idx + 1;
                    if (buf_idx == INPUT_WIDTH - 1) begin
                        buf_idx <= 0;
                        state <= 1;
                        valid_out_fc <= 1;
                    end
                end else begin // Compute output for each neuron
                    out_idx <= out_idx + 1;
                    if (out_idx == OUTPUT_NUM - 1) begin
                        out_idx <= 0;
                    end
                    valid_out_fc <= 1;
                end
            end
        end
    end
    
assign calc_out = weight[out_idx * INPUT_NUM + 0] * buffer[0] +
                  weight[out_idx * INPUT_NUM + 1] * buffer[1] +
                  weight[out_idx * INPUT_NUM + 2] * buffer[2] +
                  weight[out_idx * INPUT_NUM + 3] * buffer[3] +
                  weight[out_idx * INPUT_NUM + 4] * buffer[4] +
                  weight[out_idx * INPUT_NUM + 5] * buffer[5] +
                  weight[out_idx * INPUT_NUM + 6] * buffer[6] +
                  weight[out_idx * INPUT_NUM + 7] * buffer[7] +
                  weight[out_idx * INPUT_NUM + 8] * buffer[8] +
                  weight[out_idx * INPUT_NUM + 9] * buffer[9] +
                  weight[out_idx * INPUT_NUM + 10] * buffer[10] +
                  weight[out_idx * INPUT_NUM + 11] * buffer[11] +
                  weight[out_idx * INPUT_NUM + 12] * buffer[12] +
                  weight[out_idx * INPUT_NUM + 13] * buffer[13] +
                  weight[out_idx * INPUT_NUM + 14] * buffer[14] +
                  weight[out_idx * INPUT_NUM + 15] * buffer[15] +
                  weight[out_idx * INPUT_NUM + 16] * buffer[16] +
                  weight[out_idx * INPUT_NUM + 17] * buffer[17] +
                  weight[out_idx * INPUT_NUM + 18] * buffer[18] +
                  weight[out_idx * INPUT_NUM + 19] * buffer[19] +
                  weight[out_idx * INPUT_NUM + 20] * buffer[20] +
                  weight[out_idx * INPUT_NUM + 21] * buffer[21] +
                  weight[out_idx * INPUT_NUM + 22] * buffer[22] +
                  weight[out_idx * INPUT_NUM + 23] * buffer[23] +
                  weight[out_idx * INPUT_NUM + 24] * buffer[24] +
                  weight[out_idx * INPUT_NUM + 25] * buffer[25] +
                  weight[out_idx * INPUT_NUM + 26] * buffer[26] +
                  weight[out_idx * INPUT_NUM + 27] * buffer[27] +
                  weight[out_idx * INPUT_NUM + 28] * buffer[28] +
                  weight[out_idx * INPUT_NUM + 29] * buffer[29] +
                  weight[out_idx * INPUT_NUM + 30] * buffer[30] +
                  weight[out_idx * INPUT_NUM + 31] * buffer[31] +
                  weight[out_idx * INPUT_NUM + 32] * buffer[32] +
                  weight[out_idx * INPUT_NUM + 33] * buffer[33] +
                  weight[out_idx * INPUT_NUM + 34] * buffer[34] +
                  weight[out_idx * INPUT_NUM + 35] * buffer[35] +
                  weight[out_idx * INPUT_NUM + 36] * buffer[36] +
                  weight[out_idx * INPUT_NUM + 37] * buffer[37] +
                  weight[out_idx * INPUT_NUM + 38] * buffer[38] +
                  weight[out_idx * INPUT_NUM + 39] * buffer[39] +
                  weight[out_idx * INPUT_NUM + 40] * buffer[40] +
                  weight[out_idx * INPUT_NUM + 41] * buffer[41] +
                  weight[out_idx * INPUT_NUM + 42] * buffer[42] +
                  weight[out_idx * INPUT_NUM + 43] * buffer[43] +
                  weight[out_idx * INPUT_NUM + 44] * buffer[44] +
                  weight[out_idx * INPUT_NUM + 45] * buffer[45] +
                  weight[out_idx * INPUT_NUM + 46] * buffer[46] +
                  weight[out_idx * INPUT_NUM + 47] * buffer[47] +
                  weight[out_idx * INPUT_NUM + 48] * buffer[48] +
                  weight[out_idx * INPUT_NUM + 49] * buffer[49] +
                  weight[out_idx * INPUT_NUM + 50] * buffer[50] +
                  weight[out_idx * INPUT_NUM + 51] * buffer[51] +
                  weight[out_idx * INPUT_NUM + 52] * buffer[52] +
                  weight[out_idx * INPUT_NUM + 53] * buffer[53] +
                  weight[out_idx * INPUT_NUM + 54] * buffer[54] +
                  weight[out_idx * INPUT_NUM + 55] * buffer[55] +
                  weight[out_idx * INPUT_NUM + 56] * buffer[56] +
                  weight[out_idx * INPUT_NUM + 57] * buffer[57] +
                  weight[out_idx * INPUT_NUM + 58] * buffer[58] +
                  weight[out_idx * INPUT_NUM + 59] * buffer[59] +
                  weight[out_idx * INPUT_NUM + 60] * buffer[60] +
                  weight[out_idx * INPUT_NUM + 61] * buffer[61] +
                  weight[out_idx * INPUT_NUM + 62] * buffer[62] +
                  weight[out_idx * INPUT_NUM + 63] * buffer[63] +
                  weight[out_idx * INPUT_NUM + 64] * buffer[64] +
                  weight[out_idx * INPUT_NUM + 65] * buffer[65] +
                  weight[out_idx * INPUT_NUM + 66] * buffer[66] +
                  weight[out_idx * INPUT_NUM + 67] * buffer[67] +
                  weight[out_idx * INPUT_NUM + 68] * buffer[68] +
                  weight[out_idx * INPUT_NUM + 69] * buffer[69] +
                  weight[out_idx * INPUT_NUM + 70] * buffer[70] +
                  weight[out_idx * INPUT_NUM + 71] * buffer[71] +
                  weight[out_idx * INPUT_NUM + 72] * buffer[72] +
                  weight[out_idx * INPUT_NUM + 73] * buffer[73] +
                  weight[out_idx * INPUT_NUM + 74] * buffer[74] +
                  weight[out_idx * INPUT_NUM + 75] * buffer[75] +
                  weight[out_idx * INPUT_NUM + 76] * buffer[76] +
                  weight[out_idx * INPUT_NUM + 77] * buffer[77] +
                  weight[out_idx * INPUT_NUM + 78] * buffer[78] +
                  weight[out_idx * INPUT_NUM + 79] * buffer[79] +
                  weight[out_idx * INPUT_NUM + 80] * buffer[80] +
                  weight[out_idx * INPUT_NUM + 81] * buffer[81] +
                  weight[out_idx * INPUT_NUM + 82] * buffer[82] +
                  weight[out_idx * INPUT_NUM + 83] * buffer[83] +
                  weight[out_idx * INPUT_NUM + 84] * buffer[84] +
                  weight[out_idx * INPUT_NUM + 85] * buffer[85] +
                  weight[out_idx * INPUT_NUM + 86] * buffer[86] +
                  weight[out_idx * INPUT_NUM + 87] * buffer[87] +
                  weight[out_idx * INPUT_NUM + 88] * buffer[88] +
                  weight[out_idx * INPUT_NUM + 89] * buffer[89] +
                  weight[out_idx * INPUT_NUM + 90] * buffer[90] +
                  weight[out_idx * INPUT_NUM + 91] * buffer[91] +
                  weight[out_idx * INPUT_NUM + 92] * buffer[92] +
                  weight[out_idx * INPUT_NUM + 93] * buffer[93] +
                  weight[out_idx * INPUT_NUM + 94] * buffer[94] +
                  weight[out_idx * INPUT_NUM + 95] * buffer[95] +
                  weight[out_idx * INPUT_NUM + 96] * buffer[96] +
                  weight[out_idx * INPUT_NUM + 97] * buffer[97] +
                  weight[out_idx * INPUT_NUM + 98] * buffer[98] +
                  weight[out_idx * INPUT_NUM + 99] * buffer[99] +
                  weight[out_idx * INPUT_NUM + 100] * buffer[100] +
                  weight[out_idx * INPUT_NUM + 101] * buffer[101] +
                  weight[out_idx * INPUT_NUM + 102] * buffer[102] +
                  weight[out_idx * INPUT_NUM + 103] * buffer[103] +
                  weight[out_idx * INPUT_NUM + 104] * buffer[104] +
                  weight[out_idx * INPUT_NUM + 105] * buffer[105] +
                  weight[out_idx * INPUT_NUM + 106] * buffer[106] +
                  weight[out_idx * INPUT_NUM + 107] * buffer[107] +
                  weight[out_idx * INPUT_NUM + 108] * buffer[108] +
                  weight[out_idx * INPUT_NUM + 109] * buffer[109] +
                  weight[out_idx * INPUT_NUM + 110] * buffer[110] +
                  weight[out_idx * INPUT_NUM + 111] * buffer[111] +
                  weight[out_idx * INPUT_NUM + 112] * buffer[112] +
                  weight[out_idx * INPUT_NUM + 113] * buffer[113] +
                  weight[out_idx * INPUT_NUM + 114] * buffer[114] +
                  weight[out_idx * INPUT_NUM + 115] * buffer[115] +
                  weight[out_idx * INPUT_NUM + 116] * buffer[116] +
                  weight[out_idx * INPUT_NUM + 117] * buffer[117] +
                  weight[out_idx * INPUT_NUM + 118] * buffer[118] +
                  weight[out_idx * INPUT_NUM + 119] * buffer[119] +
                  weight[out_idx * INPUT_NUM + 120] * buffer[120] +
                  weight[out_idx * INPUT_NUM + 121] * buffer[121] +
                  weight[out_idx * INPUT_NUM + 122] * buffer[122] +
                  weight[out_idx * INPUT_NUM + 123] * buffer[123] +
                  weight[out_idx * INPUT_NUM + 124] * buffer[124] +
                  weight[out_idx * INPUT_NUM + 125] * buffer[125] +
                  weight[out_idx * INPUT_NUM + 126] * buffer[126] +
                  weight[out_idx * INPUT_NUM + 127] * buffer[127] +
                  weight[out_idx * INPUT_NUM + 128] * buffer[128] +
                  weight[out_idx * INPUT_NUM + 129] * buffer[129] +
                  weight[out_idx * INPUT_NUM + 130] * buffer[130] +
                  weight[out_idx * INPUT_NUM + 131] * buffer[131] +
                  weight[out_idx * INPUT_NUM + 132] * buffer[132] +
                  weight[out_idx * INPUT_NUM + 133] * buffer[133] +
                  weight[out_idx * INPUT_NUM + 134] * buffer[134] +
                  weight[out_idx * INPUT_NUM + 135] * buffer[135] +
                  weight[out_idx * INPUT_NUM + 136] * buffer[136] +
                  weight[out_idx * INPUT_NUM + 137] * buffer[137] +
                  weight[out_idx * INPUT_NUM + 138] * buffer[138] +
                  weight[out_idx * INPUT_NUM + 139] * buffer[139] +
                  weight[out_idx * INPUT_NUM + 140] * buffer[140] +
                  weight[out_idx * INPUT_NUM + 141] * buffer[141] +
                  weight[out_idx * INPUT_NUM + 142] * buffer[142] +
                  weight[out_idx * INPUT_NUM + 143] * buffer[143] +
                  bias[out_idx];

always @(posedge clk) begin
  if (~rst_n) begin
    data_out <= 12'b0;  // Reset the output to 0
  end else begin
    data_out <= calc_out[18:7];  // Take the slice of calc_out
  end
end


endmodule