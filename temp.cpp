__attribute__((target("avx2,fma")))
void fixed2_tile_multi_parallel(float* __restrict__ matrix,
                                float* __restrict__ bias,
                                float* __restrict__ input,
                                float* __restrict__ output,
                                size_t tile_size,
                                size_t data_cnt,
                                size_t input_dim,
                                size_t output_dim,
                                size_t start,
                                size_t end) 
{
    __m256 z = _mm256_setzero_ps();

    for (size_t col_chunk = start; col_chunk < end; col_chunk += tile_size) {
        size_t j_max = std::min(col_chunk + tile_size, end);

        for (size_t row_chunk = 0; row_chunk < data_cnt; row_chunk += tile_size) {
            size_t i_max = std::min(row_chunk + tile_size, data_cnt);

            for (size_t k_chunk = 0; k_chunk < input_dim; k_chunk += tile_size) {
                size_t k_max = std::min(k_chunk + tile_size, input_dim);

                // ---- 핵심 구간: K루프 4×언롤 ----
                for (size_t i = row_chunk; i < i_max; ++i) {
                    for (size_t k = k_chunk; k + 4 <= k_max; k += 4) {
                        __m256 a0 = _mm256_set1_ps(input[i * input_dim + (k + 0)]);
                        __m256 a1 = _mm256_set1_ps(input[i * input_dim + (k + 1)]);
                        __m256 a2 = _mm256_set1_ps(input[i * input_dim + (k + 2)]);
                        __m256 a3 = _mm256_set1_ps(input[i * input_dim + (k + 3)]);

                        size_t j = col_chunk;
                        for (; j + 8 <= j_max; j += 8) {
                            __m256 y  = _mm256_load_ps(&output[i * output_dim + j]);

                            __m256 w0 = _mm256_load_ps(&matrix[(k + 0) * output_dim + j]);
                            __m256 w1 = _mm256_load_ps(&matrix[(k + 1) * output_dim + j]);
                            __m256 w2 = _mm256_load_ps(&matrix[(k + 2) * output_dim + j]);
                            __m256 w3 = _mm256_load_ps(&matrix[(k + 3) * output_dim + j]);

                            y = _mm256_fmadd_ps(a0, w0, y);
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

                    // 남은 k (mod 4) 처리
                    for (size_t k = (k_max & ~3UL); k < k_max; ++k) {
                        __m256 a = _mm256_set1_ps(input[i * input_dim + k]);
                        size_t j = col_chunk;
                        for (; j + 8 <= j_max; j += 8) {
                            __m256 y = _mm256_load_ps(&output[i * output_dim + j]);
                            __m256 w = _mm256_load_ps(&matrix[k * output_dim + j]);
                            y = _mm256_fmadd_ps(a, w, y);
                            _mm256_store_ps(&output[i * output_dim + j], y);
                        }
                        for (; j < j_max; ++j) {
                            output[i * output_dim + j] += input[i * input_dim + k] * matrix[k * output_dim + j];
                        }
                    }
                }
            }

            // ---- Fused bias + ReLU ----
            for (size_t i = row_chunk; i < i_max; ++i) {
                size_t j = col_chunk;
                for (; j + 8 <= j_max; j += 8) {
                    __m256 y = _mm256_load_ps(&output[i * output_dim + j]);
                    __m256 b = _mm256_load_ps(&bias[j]);
                    y = _mm256_add_ps(y, b);
                    y = _mm256_max_ps(y, z);
                    _mm256_stream_ps(&output[i * output_dim + j], y);
                }
                for (; j < j_max; ++j) {
                    float v = output[i * output_dim + j] + bias[j];
                    output[i * output_dim + j] = (v > 0.0f) ? v : 0.0f;
                }
            }
        }
    }
}
