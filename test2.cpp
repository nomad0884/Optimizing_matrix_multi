#include <stdio.h>

#include <algorithm>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>


void tile_multi_parrarel(size_t data_cnt, size_t start_col, size_t end_col, float* A, float* B, float* C,float* D, size_t tile_size);
void fixed_tile_multi_parallel(float* matrix, float*bias, float* input, float*output, size_t tile_size, size_t data_cnt, size_t input_dim, size_t output_dim, size_t start, size_t end);

void
fc_layer(size_t data_cnt, size_t input_dim, size_t output_dim, float* matrix, float* bias, float* input, float* output, int threads) {

	std::size_t num_thread = threads;
	std::vector<std::thread> thread;
	thread.reserve(num_thread);

	size_t n_cols = data_cnt / num_thread;

	size_t start_col = 0;

	size_t TILE_SIZE = 16;
	for (size_t i = 0; i < num_thread; i++) {
		auto end_col = start_col + n_cols;
		thread.emplace_back([=] {
			// tile_multi_parrarel(data_cnt, start_col, end_col, matrix, bias, input, output, TILE_SIZE);
			fixed_tile_multi_parallel(matrix, bias, input, output, TILE_SIZE, data_cnt, input_dim, output_dim, start_col, end_col);
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
	for (size_t col_chunk = start_col; col_chunk < end_col; col_chunk += tile_size) {
		size_t j_max = std::min(col_chunk + tile_size , end_col);

		for (size_t row_chunck = 0; row_chunck < data_cnt; row_chunck += tile_size) {
			size_t i_max = std::min(row_chunck+tile_size , data_cnt);
			
			for (size_t tile = 0; tile < data_cnt; tile += tile_size) {

				for (size_t row = 0; row < tile_size; row++) {
					for (size_t tile_row = 0; tile_row < tile_size; tile_row++) {
						for (size_t idx = 0; idx < tile_size; idx++) {
							D[(row + row_chunck) * data_cnt + col_chunk + idx] +=
								A[(row + row_chunck) * data_cnt + tile + tile_row] *
								B[(tile + tile_row)*data_cnt + col_chunk + idx];
						}
					}
				}

			}

			// 누적합 연산이 완료 된 후, bias를 더해주고 relu 함수 적용.

			// 정확도는 더 떨어짐...
			for(size_t col0=col_chunk; col0 <i_max; col0++){
				for(size_t row0 = row_chunck; row0 < j_max; row0++){
					D[col0*data_cnt + row0] += C[row0];
					D[col0*data_cnt + row0] =  (D[col0*data_cnt + row0] < 0.0f ) ? 0.0f : D[col0*data_cnt + row0];
				
				}
			}

		}
	}




}
// 입력은 (1,4096) 가중치는 (4096,4096) 편향은 (1,4096) 출력은 (1,4096)

void fixed_tile_multi_parallel(float* matrix, float*bias, float* input, float*output, size_t tile_size, size_t data_cnt, size_t input_dim, size_t output_dim, size_t start, size_t end){
	for( size_t col_chunck = start ; col_chunck < end; col_chunck += tile_size){
		for (size_t row_chunck=0; row_chunck < data_cnt; row_chunck += tile_size){

			for(size_t j = col_chunck ; j < col_chunck + tile_size ; j++){
				for(size_t i = row_chunck; i < row_chunck+tile_size ; i++){
					for(size_t p=0; p<tile_size;p++){
						output[j*output_dim + i] +=
							input[i*input_dim + j] *
							matrix[i*output_dim + p ];
					}
				}
			}

			for(size_t bj = col_chunck; bj < tile_size; bj++){
				for (size_t bi = row_chunck; bi < output_dim ; bi++){
					output[bj*output_dim + bi] += bias[bi*data_cnt + bj];
					output[bj*output_dim + bi] = ( output[bj*output_dim + bi] <  0.0f ) ? 0.0f : output[bj*output_dim + bi];
				}
			}

		}
	}
}
