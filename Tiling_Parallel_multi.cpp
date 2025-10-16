	#include <stdio.h>

	#include <algorithm>
	#include <cstdlib>
	#include <random>
	#include <thread>
	#include <vector>


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
				fixed_tile_multi_parallel(matrix, bias, input, output, TILE_SIZE, data_cnt, input_dim, output_dim, start_col, end_col);
			});
			start_col += n_cols;
		}

		for (auto& t : thread) t.join();

		thread.clear();


	}


	// 입력은 (1,4096) 가중치는 (4096,4096) 편향은 (1,4096) 출력은 (1,4096)

	void fixed_tile_multi_parallel(float* matrix, float*bias, float* input, float*output, size_t tile_size, size_t data_cnt, size_t input_dim, size_t output_dim, size_t start, size_t end){
		for( size_t col_chunck = start ; col_chunck < end; col_chunck += tile_size){
			size_t j_max = std::min(col_chunck+tile_size, end);
			for (size_t row_chunck=0; row_chunck < data_cnt; row_chunck += tile_size){
				size_t i_max = std::min(row_chunck+tile_size, data_cnt);
				
				for(size_t t = row_chunck; t < i_max; t++){
					for(size_t h = col_chunck; h < j_max; h++){
						output[t*output_dim + h] = 0.0f;
					}
				}
				for(size_t k_chunck=0; k_chunck < input_dim ; k_chunck+=tile_size){
					size_t k_max = std::min(k_chunck+tile_size, input_dim);
					
					for(size_t i = row_chunck ; i< i_max;i++){
						for(size_t k = k_chunck; k< k_max;k++){
							for(size_t j = col_chunck; j< j_max; j++){
								output[i*output_dim + j] +=
									input[i*input_dim + k] *
									matrix[k*output_dim+j];
							}
						}
					}
				}
				

				for(size_t i = row_chunck; i< i_max; i++){
					for(size_t j = col_chunck; j < j_max; j++){
						output[i*output_dim + j] += bias[j];
						( output[i*output_dim + j] <0.0f ) ? 0.0f : output[i*output_dim+j];
					}
				}


				

			}
		}
	}
