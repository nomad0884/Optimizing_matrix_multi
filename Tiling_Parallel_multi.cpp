	#include <stdio.h>

	#include <algorithm>
	#include <cstdlib>
	#include <random>
	#include <thread>
	#include <vector>
	#include <immintrin.h>  

	void fixed_tile_multi_parallel(float* matrix, float*bias, float* input, float*output, size_t tile_size, size_t data_cnt, size_t input_dim, size_t output_dim, size_t start, size_t end);
	void fixed2_tile_multi_parallel(float* matrix, float*bias, float* input, float*output, size_t tile_size, size_t data_cnt, size_t input_dim, size_t output_dim, size_t start, size_t end);

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
				fixed2_tile_multi_parallel(matrix, bias, input, output, TILE_SIZE, data_cnt, input_dim, output_dim, start_col, end_col);
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
						output[i*output_dim + j ] = std::max(output[i*output_dim + j], 0.0f);
					}
				}


				

			}
		}
	}


	__attribute__((target("avx2,fma")))
	void fixed2_tile_multi_parallel(float* matrix, float*bias, float* input, float*output, size_t tile_size, size_t data_cnt, size_t input_dim, size_t output_dim, size_t start, size_t end){
        // transpose and fmadd를 하는게 낫나
        __m256 z = _mm256_setzero_ps();
        for( size_t col_chunck = start ; col_chunck < end ; col_chunck += tile_size){
            size_t j_max = std::min(col_chunck+tile_size, end);

            for(size_t row_chunck = 0; row_chunck < data_cnt; row_chunck+= tile_size){
                size_t i_max = std::min(row_chunck+tile_size, data_cnt);

                for(size_t k_chunck=0; k_chunck < input_dim ; k_chunck+=tile_size){
                    size_t k_max = std::min(k_chunck+tile_size , input_dim);

                    for(size_t k = k_chunck ; k<k_max;k++){
                        for(size_t i=row_chunck; i < i_max; i++){
                            __m256 temp = _mm256_set1_ps(input[i*input_dim + k]);
                            
                            size_t j = col_chunck;
                            for(;j+8 <= j_max; j+=8){
                                __m256 y = _mm256_load_ps(&output[i*output_dim + j]);
                                __m256 w = _mm256_load_ps(&matrix[k*output_dim + j]);

                                y = _mm256_fmadd_ps(temp,w,y);
                                _mm256_store_ps(&output[i*output_dim +j],y);
                            }
							for(; j<j_max;j++){
								output[i*output_dim + j] += 
									input[i*input_dim + k] * matrix[k*output_dim + j];
							}
                        }
                    }
                }


                for(size_t i = row_chunck; i< i_max; i++){
                    size_t j = col_chunck;
					for(; j+8<= j_max;j+=8){
						__m256 y = _mm256_load_ps(&output[i*output_dim + j]);
						__m256 b = _mm256_load_ps(&bias[j]);
						y = _mm256_add_ps(y,b);
						y = _mm256_max_ps(y,z);
						_mm256_storeu_ps(&output[i*output_dim + j], y);
					}
					for(;j<j_max;j++){
						float v = output[i*output_dim + j] + bias[j];
						output[i*output_dim + j] = std::max(v,0.0f);
					}

                }
            }
        }
	}
