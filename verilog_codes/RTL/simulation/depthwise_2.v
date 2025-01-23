`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 01/22/2025 09:17:29 AM
// Design Name: 
// Module Name: depthwise_2
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


module depthwise_2(
input clk , 
input rst_n , 
input valid_out_buf , 
input signed [11:0] data_out1_0, data_out1_1, data_out1_2, data_out1_3, data_out1_4,
	  data_out1_5, data_out1_6, data_out1_7, data_out1_8, data_out1_9,
	  data_out1_10, data_out1_11, data_out1_12, data_out1_13, data_out1_14,
	  data_out1_15, data_out1_16, data_out1_17, data_out1_18, data_out1_19,
	  data_out1_20, data_out1_21, data_out1_22, data_out1_23, data_out1_24,
	
	  data_out2_0, data_out2_1, data_out2_2, data_out2_3, data_out2_4,
	  data_out2_5, data_out2_6, data_out2_7, data_out2_8, data_out2_9,
	  data_out2_10, data_out2_11, data_out2_12, data_out2_13, data_out2_14,
	  data_out2_15, data_out2_16, data_out2_17, data_out2_18, data_out2_19,
	  data_out2_20, data_out2_21, data_out2_22, data_out2_23, data_out2_24,
	
	  data_out3_0, data_out3_1, data_out3_2, data_out3_3, data_out3_4,
	  data_out3_5, data_out3_6, data_out3_7, data_out3_8, data_out3_9,
	  data_out3_10, data_out3_11, data_out3_12, data_out3_13, data_out3_14,
	  data_out3_15, data_out3_16, data_out3_17, data_out3_18, data_out3_19,
	  data_out3_20, data_out3_21, data_out3_22, data_out3_23, data_out3_24,
	  
	  output reg [13:0] dep_out_1 , dep_out_2 , dep_out_3 , 
	  output reg valid_out_calc
    );
    
wire [19:0] calc_out_1 , calc_out_2 , calc_out_3;
reg signed [7:0] weights[0:74];
initial begin
    $readmemh("layer2_conv2d.mem", weights);
end


assign calc_out_1 = data_out1_0*weights[0] + data_out1_1*weights[1] + data_out1_2*weights[2] + data_out1_3*weights[3] + data_out1_4*weights[4] + 
                    data_out1_5*weights[5] + data_out1_6*weights[6] + data_out1_7*weights[7] + data_out1_8*weights[8] + data_out1_9*weights[9] + 
                    data_out1_10*weights[10] + data_out1_11*weights[11] + data_out1_12*weights[12] + data_out1_13*weights[13] + data_out1_14*weights[14] + 
                    data_out1_15*weights[15] + data_out1_16*weights[16] + data_out1_17*weights[17] + data_out1_18*weights[18] + data_out1_19*weights[19] + 
                    data_out1_20*weights[20] + data_out1_21*weights[21] + data_out1_22*weights[22] + data_out1_23*weights[23] + data_out1_24*weights[24];
                    
                    
assign calc_out_2 = data_out2_0*weights[25] + data_out2_1*weights[26] + data_out2_2*weights[27] + data_out2_3*weights[28] + data_out2_4*weights[29] + 
                    data_out2_5*weights[30] + data_out2_6*weights[31] + data_out2_7*weights[32] + data_out2_8*weights[33] + data_out2_9*weights[34] + 
                    data_out2_10*weights[35] + data_out2_11*weights[36] + data_out2_12*weights[37] + data_out2_13*weights[38] + data_out2_14*weights[39] + 
                    data_out2_15*weights[40] + data_out2_16*weights[41] + data_out2_17*weights[42] + data_out2_18*weights[43] + data_out2_19*weights[44] + 
                    data_out2_20*weights[45] + data_out2_21*weights[46] + data_out2_22*weights[47] + data_out2_23*weights[48] + data_out2_24*weights[49];

                    
                   
assign calc_out_3 = data_out3_0*weights[50] + data_out3_1*weights[51] + data_out3_2*weights[52] + data_out3_3*weights[53] + data_out3_4*weights[54] + 
                    data_out3_5*weights[55] + data_out3_6*weights[56] + data_out3_7*weights[57] + data_out3_8*weights[58] + data_out3_9*weights[59] + 
                    data_out3_10*weights[60] + data_out3_11*weights[61] + data_out3_12*weights[62] + data_out3_13*weights[63] + data_out3_14*weights[64] + 
                    data_out3_15*weights[65] + data_out3_16*weights[66] + data_out3_17*weights[67] + data_out3_18*weights[68] + data_out3_19*weights[69] + 
                    data_out3_20*weights[70] + data_out3_21*weights[71] + data_out3_22*weights[72] + data_out3_23*weights[73] + data_out3_24*weights[74];


always @ (posedge clk) begin
	if(~rst_n) begin
		valid_out_calc <= 0;
    dep_out_1 <=0 ;
    dep_out_2 <=0 ;
    dep_out_3 <=0 ; 
	end
	else begin
		// Toggling Valid Output Signal
		if(valid_out_buf == 1) begin
			if(valid_out_calc == 1)
				valid_out_calc <= 0;
			else
				valid_out_calc <= 1;
				dep_out_1 <= calc_out_1[19:6];
				dep_out_2 <= calc_out_2[19:6];
				dep_out_3 <= calc_out_3[19:6];
		end
	end
end


endmodule
