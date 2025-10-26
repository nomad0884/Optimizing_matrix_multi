	#include <stdio.h>

	#include <algorithm>
	#include <cstdlib>
	#include <random>
	#include <thread>
	#include <vector>
	#include <immintrin.h>  

	void fixed2_tile_multi_parallel(float* matrix, float*bias, float* input, float*output, size_t tile_size, size_t data_cnt, size_t input_dim, size_t output_dim, size_t start, size_t end);

	void
	fc_layer(size_t data_cnt, size_t input_dim, size_t output_dim, float* matrix, float* bias, float* input, float* output, int threads) {

		std::size_t num_thread = threads;
		std::vector<std::thread> thread;
		thread.reserve(num_thread);

		size_t n_cols = data_cnt / num_thread;

		size_t start_col = 0;

		size_t TILE_SIZE = 256;
		for (size_t i = 0; i < num_thread; i++) {
			auto end_col = start_col + n_cols;
			thread.emplace_back([=] {
				fixed2_tile_multi_parallel(matrix, bias, input, output, TILE_SIZE, data_cnt, input_dim, output_dim, start_col, end_col);
			});
			start_col += n_cols;
		}

		for (auto& t : thread) t.join();

		thread.clear();


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
								__m256 b = _mm256_load_ps(&bias[j]);
                                y = _mm256_fmadd_ps(temp,w,b);
                                _mm256_store_ps(&output[i*output_dim +j],y);
								//_mm256_stream_ps(&output[i*output_dim + j],y);
                            }
							for(; j<j_max;j++){  // 처리 되지 않은 index들을 마저 처리.
								output[i*output_dim + j] += 
									input[i*input_dim + k] * matrix[k*output_dim + j] + bias[j];
							}
                        }
                    }
                }


                for(size_t i = row_chunck; i< i_max; i++){
                    size_t j = col_chunck;
					for(; j+8<= j_max;j+=8){
						__m256 y = _mm256_load_ps(&output[i*output_dim + j]);
						// __m256 b = _mm256_load_ps(&bias[j]);
						// y = _mm256_add_ps(y,b);
						y = _mm256_max_ps(y,z);
						// _mm256_storeu_ps(&output[i*output_dim + j], y);
						_mm256_stream_ps(&output[i*output_dim+j],y);
					}
					for(;j<j_max;j++){  // 남은 index 처리.
						float v = output[i*output_dim + j] + bias[j];
						output[i*output_dim + j] = std::max(v,0.0f);
					}

                }
            }
        }
	}
