#include <stdint.h>
#include "CUDASearch.h"

static const int msgsize = 13;
static const int paramsize = 4;
void CUDASearch::VMSearcherGPU::allocate(size_t new_bsize)
{
	if (this->bsize_ < new_bsize) {
		deallocate();
		this->bsize_ = new_bsize;
		UInt64* uptr;
		cudaMalloc(&uptr, bsize_ * sizeof(UInt64));
		this->seeds_ = uptr;
	}
}


CUDASearch::VMSearcherGPU::VMSearcherGPU(int n, int range)
{
	this->mask_ = (1ULL << (5 * n)) - 1;
	this->range_ = range + n;
}

CUDASearch::VMSearcherGPU::~VMSearcherGPU()
{
	this->!VMSearcherGPU();
}

CUDASearch::VMSearcherGPU::!VMSearcherGPU()
{
	cudaFree(this->seeds_);
}

void CUDASearch::VMSearcherGPU::calculate(cli::array<UInt64>^ seeds)
{
	extern void device_vending_search(uint64_t* seeds, uint64_t const mask, uint32_t const size, int32_t const range);
	UInt32 size = seeds->Length;
	allocate((size_t)size);
	//host -> device
	pin_ptr<UInt64> hseeds = &seeds[0];
	cudaMemcpy(this->seeds_, hseeds, size * sizeof(UInt64), cudaMemcpyHostToDevice);
	//kernel call
	device_vending_search(this->seeds_, this->mask_, size, this->range_);
	// device -> host
	cudaMemcpy(hseeds, this->seeds_, size * sizeof(UInt64), cudaMemcpyDeviceToHost);
}

void CUDASearch::SeedGenerator::allocate(size_t new_bsize)
{
	if (this->bsize_ < new_bsize) {
		deallocate();
		this->bsize_ = new_bsize;
		UInt64* iptr;
		UInt64* sptr;
		UInt32* ptr;

		cudaMalloc(&iptr, bsize_ * sizeof(UInt64));
		this->initseeds_ = iptr;
		cudaMalloc(&sptr, bsize_ * sizeof(UInt64));
		this->seeds_ = sptr;
		cudaMalloc(&ptr, paramsize * bsize_ * sizeof(UInt32));
		this->params_ = ptr;
		cudaMalloc(&ptr, msgsize * sizeof(UInt32));
		this->base_ = ptr;
	}
}


CUDASearch::SeedGenerator::SeedGenerator(UInt64 initmul, UInt64 initadd)
{
	this->initmul_ = initmul;
	this->initadd_ = initadd;
}


CUDASearch::SeedGenerator::~SeedGenerator()
{
	this->!SeedGenerator();
}

CUDASearch::SeedGenerator::!SeedGenerator()
{
	cudaFree(this->params_);
	cudaFree(this->initseeds_);
	cudaFree(this->seeds_);
	cudaFree(this->base_);
}

void CUDASearch::SeedGenerator::calculate(cli::array<UInt32>^ m, cli::array<UInt32>^ params, cli::array<UInt64>^ initseeds, cli::array<UInt64>^ seeds, Int32 length)
{
	extern void device_generate_initseed(uint32_t * params, uint64_t * initseeds, uint64_t * seeds, uint32_t const size, uint64_t const initmul, uint64_t const initadd, uint32_t const m0, uint32_t const m1, uint32_t const m2, uint32_t const m3, uint32_t const m4, uint32_t const m5, uint32_t const m6, uint32_t const m7);
	UInt32 size = length;
	allocate(size);

	pin_ptr<UInt32> hparams = &params[0];
	pin_ptr<UInt64> hinitseeds = &initseeds[0];
	pin_ptr<UInt64> hseeds = &seeds[0];
	cudaMemcpy(this->params_, hparams, paramsize * size * sizeof(UInt32), cudaMemcpyHostToDevice);
	device_generate_initseed(this->params_, this->initseeds_, this->seeds_, size, this->initmul_, this->initadd_, m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7]);
	cudaMemcpy(hinitseeds, this->initseeds_, size * sizeof(UInt64), cudaMemcpyDeviceToHost);
	cudaMemcpy(hseeds, this->seeds_, size * sizeof(UInt64), cudaMemcpyDeviceToHost);
}
