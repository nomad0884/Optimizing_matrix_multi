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
