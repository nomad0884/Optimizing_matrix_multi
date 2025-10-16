#include <stdio.h>

#include <algorithm>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

void
fc_layer(size_t data_cnt, size_t input_dim, size_t output_dim, float* matrix, float* bias, float* input, float* output, int threads) {
	// loop over input instances
	//for ( size_t iidx = 0; iidx < data_cnt; iidx++ ) {
	//	// loop over weight columns
	//	for ( size_t oidx = 0; oidx < output_dim; oidx++ ) {
	//		float outv = 0;
	//		// loop over each input's activation values
	//		for ( size_t aidx = 0; aidx < input_dim; aidx++ ) {
	//			float inv = input[input_dim*iidx+aidx];
	//			float weight = matrix[output_dim*aidx+oidx];
	//			outv += inv*weight;
	//		}
	//		outv += bias[oidx];
	//		if ( outv < 0 ) outv = 0;

	//		output[iidx*output_dim+oidx] = outv;
	//	}
	//}
	


	std::size_t num_thread = threads;
	std::vector<std::thread> thread_pool;
	thread_pool.reserve(num_thread);

	std::size_t n_cols = data_cnt / num_thread;

	size_t start_col = 0;
	for (size_t i = 0; i < num_thread; i++) {
		size_t end_col = start_col + n_cols;

		thread_pool.emplace_back([&, start_col, end_col]() {
			tile_multi_parallel(data_cnt, input_dim, output_dim, matrix, bias, input, output, start_col, end_col, 16);
			});

		start_col += n_cols;
	}

	for (auto& t : thread_pool)
		t.join();

	thread_pool.clear();

}

// input -> n , output -> m
// matrix -> N * M , bias -> 1,M   , input -> 1* N
// convolution ( (1,N) * (N,M) ) 's shape

void tile_multi(size_t data_cnt, size_t input_dim, size_t output_dim, float* A, float* B, float* C,float*D size_t start_col, size_t end_col, int tile_size) {
	float overlap = data_cnt / tile_size;
	for (size_t col_chunk = 0; col_chunck < data_cnt; col_chunck += tile_size) { // +16 은 타일 개수
		for (size_t row_chunck = 0; row_chunck < output_dim; row_chunck += tile_size) {
			for (size_t aidx = 0; aidx < input_dim; aidx += tile_size) {

				for (size_t row = 0; row < tile_size; row++) {
					for (size_t tile_row = 0; tile_row < tile_size; tile_row++) {
						for (size_t idx = 0; idx < tile_size; idx++) {
							D(row + row_chunck) * data_cnt + col_chunck + idx] +=
								A[(row + row_chunck) * data_cnt + tile + tile_row] *
								B[tile * data_cnt + tile_row * data_cnt + col_chunk + idx] +
								C[row_chunck * data_cnt + tile_row] / overlap;


							if (D(row + row_chunck) * data_cnt + col_chunck + idx] < 0) D(row + row_chunck)* data_cnt + col_chunck + idx] = 0;

							// 너무 연산을 많이 하는 거 같음. instruction set이 많아져서 clock cycle이 커질 거 같음..
							 
						}

						
					}
				}


			}

		 }
	}
}

void tile_multi_parallel(size_t data_cnt, size_t input_dim, size_t output_dim,
	float* A, float* B, float* C, float* D,
	size_t start_col, size_t end_col, int tile_size)
{
	float overlap = static_cast<float>(data_cnt) / tile_size;

	for (size_t col_chunk = start_col; col_chunk < end_col; col_chunk += tile_size) {
		for (size_t row_chunk = 0; row_chunk < output_dim; row_chunk += tile_size) {
			for (size_t tile = 0; tile < input_dim; tile += tile_size) {
				for (size_t row = 0; row < tile_size; row++) {
					for (size_t tile_row = 0; tile_row < tile_size; tile_row++) {
						for (size_t idx = 0; idx < tile_size; idx++) {
							size_t d_idx = (row + row_chunk) * data_cnt + col_chunk + idx;

							D[d_idx] +=
								A[(row + row_chunk) * data_cnt + tile + tile_row] *
								B[tile * data_cnt + tile_row * data_cnt + col_chunk + idx] +
								C[row_chunk * data_cnt + tile_row] / overlap;

							if (D[d_idx] < 0)
								D[d_idx] = 0;
						}
					}
				}
			}
		}
	}
}
