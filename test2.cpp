#include <stdio.h>

#include <algorithm>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>


void tile_multi_parrarel(size_t data_cnt, size_t start_col, size_t end_col, float* A, float* B, float* C,float* D, int tile_size);
 

void
fc_layer(size_t data_cnt, size_t input_dim, size_t output_dim, float* matrix, float* bias, float* input, float* output, int threads) {

	std::size_t num_thread = threads;
	std::vector<std::thread> thread;
	thread.reserve(num_thread);

	size_t n_cols = data_cnt / num_thread;

	size_t start_col = 0;
	for (size_t i = 0; i < num_thread; i++) {
		auto end_col = start_col + n_cols;
		thread.emplace_back([&] {
			tile_multi_parrarel(data_cnt, start_col, end_col, matrix, bias, input, output, num_thread);
		});
		start_col += n_cols;
	}

	for (auto& t : thread) t.join();

	thread.clear();


}


void tile_multi_parrarel(size_t data_cnt, size_t start_col, size_t end_col, float* A, float* B, float* C,float* D, size_t tile_size) {

	// bias를 미리 output에 다 더해놓은 상태에서
	// 가중치 * input 누적 합을 output에 해주면 됨

	// 그리고 최종에서 relu 한번으로 계산하면 됨.

	// tile이기 때문에 col_chunck를 i0로 둬서, 다시 for loop을 제작해서 하면 됨.
	float overlap = data_cnt / tile_size;
	for (size_t col_chunk = start_col; col_chunk < end_col; col_chunk += tile_size) {
		for (size_t row_chunck = 0; row_chunck < data_cnt; row_chunck += tile_size) {
			// output에 우선적으로 bias를 더해주는 연산
			for(size_t col0=col_chunk; col0 <col_chunk+tile_size; col0++){
				for(size_t row0 = row_chunck; row0 < row_chunck+tile_size; row0++){
					for(size_t ix=0; ix<data_cnt;ix++){
						D[(row0+row_chunck)*data_cnt + ix] += 
						C[row0*data_cnt + ix];
					}
				
				}
			}


			for (size_t tile = 0; tile < data_cnt; tile += tile_size) {
				for (size_t row = 0; row < tile_size; row++) {
					for (size_t tile_row = 0; tile_row < tile_size; tile_row++) {
						for (size_t idx = 0; idx < tile_size; idx++) {
							D[(row + row_chunck) * data_cnt + col_chunk + idx] +=
								A[(row + row_chunck) * data_cnt + tile + tile_row] *
								B[tile * data_cnt + tile_row * data_cnt + col_chunk + idx];

							if (D[(row + row_chunck) * data_cnt + col_chunk + idx] < 0) D[(row + row_chunck)* data_cnt + col_chunk + idx] = 0;
							// ReLu function도 수정해야함. 
						}
					}
				}
			}
		}
	}




}
