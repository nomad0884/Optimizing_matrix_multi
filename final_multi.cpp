	#include <stdio.h>

	#include <algorithm>
	#include <cstdlib>
	#include <random>
	#include <thread>
	#include <vector>
	#include <immintrin.h>  

	// GitHub Link : https://github.com/nomad0884/Optimizing_matrix_multi 
	// Notion Link : https://tranquil-diplodocus-d1c.notion.site/Lab-1-Optimized-Nueral-Network-Software-289ebcfb1de180a887cfe33eb83c1e36?pvs=73



	// 최적의 tile size는 256으로 산정됨.
	// thread 개수는 8개가 최적이었음. thread는 2의 n 승으로 넣어야 새그멘테이션 오류가 안남. 입력 data를 고려함. 
	// 남은 열 연산 방식이 tile size, thread가 2의 n 승이어야 최적의 상태임. 

	void tile_multi_parallel(float* matrix, float*bias, float* input, float*output, size_t tile_size, size_t data_cnt, size_t input_dim, size_t output_dim, size_t start, size_t end);

	void
	fc_layer(size_t data_cnt, size_t input_dim, size_t output_dim, float* matrix, float* bias, float* input, float* output, int threads) {

		std::size_t num_thread = threads;
		std::vector<std::thread> thread;
		thread.reserve(num_thread);

		size_t n_cols = data_cnt / num_thread;  

		size_t start_col = 0;

		size_t TILE_SIZE =  128;   // 여러 연산 수행 끝에 128이 최적의 tile size. 
		for (size_t i = 0; i < num_thread; i++) {
			auto end_col = start_col + n_cols;
			thread.emplace_back([=] {
				tile_multi_parallel(matrix, bias, input, output, TILE_SIZE, data_cnt, input_dim, output_dim, start_col, end_col);
			});
			start_col += n_cols;
		}

		for (auto& t : thread) t.join();

		thread.clear();


	}


	


__attribute__((target("avx2,fma")))
void tile_multi_parallel(float*  matrix, float* bias, float*  input, float* output, size_t tile_size, size_t data_cnt, size_t input_dim, size_t output_dim, size_t start, size_t end) 
{
    __m256 z = _mm256_setzero_ps();

	for(size_t i = start; i< end; i++){   // first-touch 초기화 . 각자 쓰레드 할당 영역에 대해서 먼저 작성하여 인근 cache , memory로 불러오는 것.  ( 미세한 성능 향상 ).
		for(size_t j = 0; j< data_cnt; j+=8){
			_mm256_store_ps(&output[i * output_dim + j], z);
		}
	}

    for (size_t col_chunk = start; col_chunk < end; col_chunk += tile_size) {
        size_t j_max = std::min(col_chunk + tile_size, end);

        for (size_t row_chunk = 0; row_chunk < data_cnt; row_chunk += tile_size) {
            size_t i_max = std::min(row_chunk + tile_size, data_cnt);

            for (size_t k_chunk = 0; k_chunk < input_dim; k_chunk += tile_size) {
                size_t k_max = std::min(k_chunk + tile_size, input_dim);

                for (size_t i = row_chunk; i < i_max; ++i) {
                    for (size_t k = k_chunk; k + 4 <= k_max; k += 4) {
						// input의 K~ 열의 value들을 받아와서 8개의 float vector로 구현.   (8 * 4)
                        __m256 a0 = _mm256_set1_ps(input[i * input_dim + (k + 0)]);
                        __m256 a1 = _mm256_set1_ps(input[i * input_dim + (k + 1)]);
                        __m256 a2 = _mm256_set1_ps(input[i * input_dim + (k + 2)]);
                        __m256 a3 = _mm256_set1_ps(input[i * input_dim + (k + 3)]);

                        size_t j = col_chunk;
                        for (; j + 8 <= j_max; j += 8) {
                            __m256 y  = _mm256_load_ps(&output[i * output_dim + j]);
							
							// input과 matrix k행 j열 계산. 8개씩 계산하기 때문에 동일. (4* 8)
                            __m256 w0 = _mm256_load_ps(&matrix[(k + 0) * output_dim + j]);
                            __m256 w1 = _mm256_load_ps(&matrix[(k + 1) * output_dim + j]);
                            __m256 w2 = _mm256_load_ps(&matrix[(k + 2) * output_dim + j]);
                            __m256 w3 = _mm256_load_ps(&matrix[(k + 3) * output_dim + j]);
							
							// output i행 j열에 중복 연산을 store. 
                            y = _mm256_fmadd_ps(a0, w0, y);  // 누적 sum. 
                            y = _mm256_fmadd_ps(a1, w1, y);
                            y = _mm256_fmadd_ps(a2, w2, y);
                            y = _mm256_fmadd_ps(a3, w3, y);

                            _mm256_store_ps(&output[i * output_dim + j], y);
                        }

                        // 잔여 열 처리
                        for (; j < j_max; ++j) {
                            output[i * output_dim + j] +=
                                input[i * input_dim + (k + 0)] * matrix[(k + 0) * output_dim + j] +
                                input[i * input_dim + (k + 1)] * matrix[(k + 1) * output_dim + j] +
                                input[i * input_dim + (k + 2)] * matrix[(k + 2) * output_dim + j] +
                                input[i * input_dim + (k + 3)] * matrix[(k + 3) * output_dim + j];
                        }
                    }

                    // 남은 K 처리하는 loop
                    for (size_t k = (k_max & ~3UL); k < k_max; ++k) {   // 4개씩 덩어리로 처리 안되는 것들에 대한 나머지 연산.  tile size가 2의 배수면 연산이 없을 듯.
                        __m256 a = _mm256_set1_ps(input[i * input_dim + k]);
                        size_t j = col_chunk;
                        for (; j + 8 <= j_max; j += 8) {
                            __m256 y = _mm256_load_ps(&output[i * output_dim + j]);
                            __m256 w = _mm256_load_ps(&matrix[k * output_dim + j]);
                            y = _mm256_fmadd_ps(a, w, y);
                            _mm256_store_ps(&output[i * output_dim + j], y);
                        }
                        for (; j < j_max; ++j) {  // 남은 열 연산. 
                            output[i * output_dim + j] += input[i * input_dim + k] * matrix[k * output_dim + j];
                        }
                    }
                }
            }

			// bias 더하기, ReLU 연산.
            for (size_t i = row_chunk; i < i_max; ++i) {
                size_t j = col_chunk;
                for (; j + 8 <= j_max; j += 8) {
                    __m256 y = _mm256_load_ps(&output[i * output_dim + j]);
                    __m256 b = _mm256_load_ps(&bias[j]);
                    y = _mm256_add_ps(y, b);
                    y = _mm256_max_ps(y, z);
                    _mm256_store_ps(&output[i * output_dim + j], y);   // loss 구하는 과정에서 다시 접근하기 때문에 store로 변경. 
                }
                for (; j < j_max; ++j) {   // 남은 열 연산. 
                    float v = output[i * output_dim + j] + bias[j];
                    output[i * output_dim + j] = (v > 0.0f) ? v : 0.0f;
                }
            }
        }
    }
}
