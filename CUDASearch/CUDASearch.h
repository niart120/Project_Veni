#pragma once

using namespace System;
#include "cuda_runtime.h"

namespace CUDASearch {
	public ref class VMSearcherGPU
	{
	private:
		UInt64* seeds_;
		size_t bsize_;
		UInt64 mask_;
		int range_;

		void allocate(size_t new_bsize);
		void deallocate() {
			cudaFree(seeds_);
			seeds_ = nullptr;
		}
	public:
		VMSearcherGPU(int n, int range);
		~VMSearcherGPU();
		!VMSearcherGPU();
		void calculate(cli::array<UInt64>^ seeds);
	};

	public ref class SeedGenerator
	{
	private:
		UInt32* params_;
		UInt64* initseeds_;
		UInt64* seeds_;
		__constant__ UInt32* base_;

		UInt64 initmul_;
		UInt64 initadd_;
		size_t bsize_;

		void allocate(size_t new_bsize);
		void deallocate() {
			cudaFree(params_);
			cudaFree(initseeds_);
			cudaFree(seeds_);
			cudaFree(base_);
			params_ = nullptr;
			initseeds_ = nullptr;
			seeds_ = nullptr;
			base_ = nullptr;
		}

	public:
		SeedGenerator(UInt64 initmul, UInt64 initadd);
		~SeedGenerator();
		!SeedGenerator();
		void calculate(cli::array<UInt32>^ base, cli::array<UInt32>^ params, cli::array<UInt64>^ initseeds, cli::array<UInt64>^ seeds, Int32 length);
	};
}
